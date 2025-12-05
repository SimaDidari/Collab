import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import importlib
from collections import defaultdict
from transformers import HfArgumentParser
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from igraph import Graph
import igraph as ig
import numpy as np
from collections import defaultdict
import re
import time
import ipdb

# Topic modeling imports
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from corextopic import corextopic as ct  # CorEx is now required

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from .information_extraction.openie_transformers_offline import TransformersOfflineOpenIE
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter
from .utils.misc_utils import *
from .utils.misc_utils import NerRawOutput, TripleRawOutput
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig

logger = logging.getLogger(__name__)


class TopicExtractor:
    """
    Topic extraction using CorEx (Correlation Explanation) algorithm.
    CorEx finds maximally informative topics by maximizing total correlation.
    """
    
    def __init__(self, n_topics: int = 50, n_top_words: int = 10, anchor_strength: float = 2.0):
        """
        Initialize the TopicExtractor.
        
        Args:
            n_topics: Number of topics to extract
            n_top_words: Number of top words to use for topic representation
            anchor_strength: Strength of anchor words (if using semi-supervised)
            llm_model: LLM model for generating topic sentence representations
            embedding_model: Embedding model for computing topic-query similarity
        """
        self.n_topics = n_topics
        self.n_top_words = n_top_words
        self.anchor_strength = anchor_strength
        self.vectorizer = None
        self.model = None
        self.topic_words = None
        self.vocab = None
        self.topic_sentences = None
        self.topic_sentence_embeddings = None
    
    def _generate_topic_sentence(self, words: List[str],  llm_model: BaseLLM = None) -> str:
        """
        Generate a concise sentence representing the topic given a list of words.
        
        Args:
            words: List of top words for the topic
            
        Returns:
            A concise sentence representing the topic
        """
        if llm_model is None:
            # Fallback: just join the words
            return " ".join(words)
        
        words_str = ", ".join(words)
        prompt = f"Given these topic words: [{words_str}], generate a single concise sentence (max 15 words) that captures the essence of this topic. Output only the sentence, nothing else."
        
        messages = [{"role": "user", "content": prompt}]
        try:
            response, _, _ = llm_model.infer(messages)
            return response.strip()
        except Exception as e:
            logger.warning(f"Error generating topic sentence: {e}. Falling back to word concatenation.")
            return " ".join(words)
        
    def fit(self, documents: List[str], topic_sentences_path: str = None,  llm_model: BaseLLM = None, embedding_model: BaseEmbeddingModel = None) -> Tuple[np.ndarray, List[str]]:
        """
        Extract topics from documents using CorEx.
        
        Args:
            documents: List of document strings
            topic_sentences_path: Optional path to save/load topic sentences JSON
            
        Returns:
            Tuple of (topic_document_matrix, topic_labels)
            - topic_document_matrix: shape (n_topics, n_documents) with relevance scores
            - topic_labels: List of topic label strings
        """
        # Create document-term matrix
        self.vectorizer = CountVectorizer(
            max_df=0.95, 
            min_df=2, 
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        self.vocab = list(self.vectorizer.get_feature_names_out())
        
        # Initialize and fit CorEx model
        # Note: words parameter goes in fit(), not __init__()
        self.model = ct.Corex(n_hidden=self.n_topics, seed=42)
        self.model.fit(doc_term_matrix, words=self.vocab)
        
        # Get topic-document relevance scores using predict_proba
        # predict_proba returns tuple: (probabilities, log_odds)
        # probabilities shape: (n_docs, n_topics)
        probs_tuple = self.model.predict_proba(doc_term_matrix)
        topic_doc_matrix = np.exp(probs_tuple[0].T)  # shape: (n_topics, n_docs)
        
        # # Normalize to [0, 1] range
        # topic_doc_matrix = topic_doc_matrix / (topic_doc_matrix.max(axis=1, keepdims=True) + 1e-10)
        
        # Get topic labels from top words - ensure uniqueness by appending index
        topic_labels = []
        self.topic_words = []
        self.topic_sentences = []
        seen_labels = set()
        
        for topic_idx in range(self.n_topics):
            # 1) Extract the list of words for the topic
            top_words = self.model.get_topics(topic=topic_idx, n_words=self.n_top_words)
            if top_words:
                words = [word for word, _, _ in top_words]
                self.topic_words.append(words)
                base_label = "_".join(words[:3])  # Use top 3 words as label
            else:
                self.topic_words.append([])
                base_label = f"topic_{topic_idx}"
            
            # Ensure uniqueness by appending index if duplicate
            label = base_label
            if label in seen_labels:
                label = f"{base_label}_{topic_idx}"
            seen_labels.add(label)
            topic_labels.append(label)
            
            # 2) Generate a sentence representation using LLM
            if self.topic_words[topic_idx]:
                topic_sentence = self._generate_topic_sentence(self.topic_words[topic_idx], llm_model)
            else:
                topic_sentence = f"Topic {topic_idx}"
            self.topic_sentences.append(topic_sentence)
            logger.info(f"Topic {topic_idx}: words={self.topic_words[topic_idx][:5]}, sentence='{topic_sentence}'")
        
        # Store topic sentences in JSON
        if topic_sentences_path:
            topic_sentences_data = {
                "topic_sentences": self.topic_sentences,
                "topic_words": self.topic_words,
                "topic_labels": topic_labels
            }
            with open(topic_sentences_path, 'w') as f:
                json.dump(topic_sentences_data, f, indent=2)
            logger.info(f"Topic sentences saved to {topic_sentences_path}")
        
        # Pre-compute topic sentence embeddings if embedding model is available
        if embedding_model is not None and self.topic_sentences:
            logger.info("Computing topic sentence embeddings...")
            self.topic_sentence_embeddings = np.array(
                embedding_model.batch_encode(self.topic_sentences, norm=True)
            )
        
        return topic_doc_matrix, topic_labels
    
    def load_topic_sentences(self, topic_sentences_path: str, embedding_model: BaseEmbeddingModel = None):
        """
        Load topic sentences from JSON file.
        
        Args:
            topic_sentences_path: Path to the topic sentences JSON file
            embedding_model: Embedding model for computing embeddings
        """
        if os.path.exists(topic_sentences_path):
            with open(topic_sentences_path, 'r') as f:
                data = json.load(f)
            self.topic_sentences = data.get("topic_sentences", [])
            self.topic_words = data.get("topic_words", [])
            logger.info(f"Loaded {len(self.topic_sentences)} topic sentences from {topic_sentences_path}")
            
            # Re-compute embeddings if embedding model is available
            if embedding_model is not None and self.topic_sentences:
                logger.info("Computing topic sentence embeddings...")
                self.topic_sentence_embeddings = np.array(
                    embedding_model.batch_encode(self.topic_sentences, norm=True)
                )
    
    def get_query_topic_scores(self, query: str, embedding_model: BaseEmbeddingModel = None) -> np.ndarray:
        """
        Extract topic relevance scores for a query based on cosine similarity
        between the query embedding and topic sentence embeddings.
        
        Args:
            query: Query string
            
        Returns:
            Array of shape (n_topics,) with topic relevance scores for the query
        """
        # Use cosine similarity with topic sentence embeddings if available
        if embedding_model is not None and self.topic_sentence_embeddings is not None:
            # Compute query embedding
            query_embedding = np.array(embedding_model.batch_encode([query], norm=True))
            query_embedding = query_embedding.squeeze()  # shape: (embedding_dim,)
            
            # Compute cosine similarity with all topic sentence embeddings
            # topic_sentence_embeddings shape: (n_topics, embedding_dim)
            # Since embeddings are normalized, dot product = cosine similarity
            scores = np.dot(self.topic_sentence_embeddings, query_embedding)  # shape: (n_topics,)
            
            return scores
        
        # Fallback to original CorEx-based method if no embedding model
        if self.vectorizer is None or self.model is None:
            raise ValueError("TopicExtractor must be fit before extracting query topics")
        
        # Transform query to document-term representation
        query_term_matrix = self.vectorizer.transform([query])
        
        # Get topic probabilities for the query using predict_proba
        # predict_proba returns tuple: (probabilities, log_odds)
        probs_tuple = self.model.predict_proba(query_term_matrix)
        query_topic_scores = np.exp(probs_tuple[0].squeeze())  # shape: (n_topics,)

        
        return query_topic_scores
    
    def get_topic_embedding_text(self, topic_idx: int) -> str:
        """
        Get a text representation of a topic for embedding.
        
        Args:
            topic_idx: Index of the topic
            
        Returns:
            String representation of the topic (sentence if available, otherwise words)
        """
        # Return topic sentence if available
        if self.topic_sentences and topic_idx < len(self.topic_sentences):
            return self.topic_sentences[topic_idx]
        # Fallback to words
        if self.topic_words and topic_idx < len(self.topic_words):
            return " ".join(self.topic_words[topic_idx])
        return f"topic {topic_idx}"


class ClusteredTopicExtractor:
    """
    Wrapper that clusters documents using Faiss K-means before applying TopicExtractor
    to each cluster for more focused topic extraction.
    """
    
    def __init__(self, n_clusters: int = 10, n_topics_per_cluster: int = 50, n_top_words: int = 10):
        """
        Initialize the ClusteredTopicExtractor.
        
        Args:
            n_clusters: Number of document clusters to create
            n_topics_per_cluster: Number of topics to extract per cluster
            n_top_words: Number of top words per topic
        """
        self.n_clusters = n_clusters
        self.n_topics_per_cluster = n_topics_per_cluster
        self.n_top_words = n_top_words
        
        # Per-cluster topic extractors
        self.cluster_topic_extractors: Dict[int, TopicExtractor] = {}
        self.cluster_assignments: np.ndarray = None
        self.cluster_centroids: np.ndarray = None
        
        # Aggregated topic data
        self.topic_sentences: List[str] = []
        self.topic_words: List[List[str]] = []
        self.topic_sentence_embeddings: np.ndarray = None
        self.topic_to_cluster: Dict[int, int] = {}  # Maps global topic idx to cluster idx
        
    def fit(self, documents: List[str], topic_sentences_path: str = None, 
            llm_model: BaseLLM = None, embedding_model: BaseEmbeddingModel = None) -> Tuple[np.ndarray, List[str]]:
        """
        Cluster documents and extract topics from each cluster.
        
        Args:
            documents: List of document strings
            topic_sentences_path: Optional path to save/load topic sentences JSON
            llm_model: LLM model for generating topic sentences
            embedding_model: Embedding model for clustering and similarity
            
        Returns:
            Tuple of (topic_document_matrix, topic_labels)
        """
        import faiss
        
        if embedding_model is None:
            raise ValueError("embedding_model is required for clustering")
        
        logger.info(f"Clustering {len(documents)} documents into {self.n_clusters} clusters using Faiss K-means")
        
        # Get document embeddings for clustering
        doc_embeddings = np.array(embedding_model.batch_encode(documents, norm=True)).astype('float32')
        embedding_dim = doc_embeddings.shape[1]
        
        # Faiss K-means clustering
        kmeans = faiss.Kmeans(embedding_dim, self.n_clusters, niter=20, verbose=True, seed=42)
        kmeans.train(doc_embeddings)
        
        # Get cluster assignments
        _, self.cluster_assignments = kmeans.index.search(doc_embeddings, 1)
        self.cluster_assignments = self.cluster_assignments.squeeze()
        self.cluster_centroids = kmeans.centroids
        
        # Log cluster sizes
        cluster_sizes = {i: int(np.sum(self.cluster_assignments == i)) for i in range(self.n_clusters)}
        logger.info(f"Cluster sizes: {cluster_sizes}")
        
        # Extract topics for each cluster
        all_topic_labels = []
        self.topic_sentences = []
        self.topic_words = []
        global_topic_idx = 0
        
        # Initialize topic-document matrix (will be filled per cluster)
        total_topics = self.n_clusters * self.n_topics_per_cluster
        topic_doc_matrix = np.zeros((total_topics, len(documents)))
        
        for cluster_idx in range(self.n_clusters):
            cluster_mask = self.cluster_assignments == cluster_idx
            cluster_doc_indices = np.where(cluster_mask)[0]
            cluster_docs = [documents[i] for i in cluster_doc_indices]
            
            if len(cluster_docs) < 3:
                logger.warning(f"Cluster {cluster_idx} has only {len(cluster_docs)} documents, skipping topic extraction")
                # Add placeholder topics for this cluster
                for local_idx in range(self.n_topics_per_cluster):
                    all_topic_labels.append(f"cluster_{cluster_idx}_topic_{local_idx}")
                    self.topic_sentences.append(f"Cluster {cluster_idx} topic {local_idx}")
                    self.topic_words.append([])
                    self.topic_to_cluster[global_topic_idx] = cluster_idx
                    global_topic_idx += 1
                continue
            
            logger.info(f"Extracting topics for cluster {cluster_idx} with {len(cluster_docs)} documents")
            
            # Create and fit topic extractor for this cluster
            cluster_extractor = TopicExtractor(
                n_topics=self.n_topics_per_cluster,
                n_top_words=self.n_top_words
            )
            
            cluster_topic_doc_matrix, cluster_topic_labels = cluster_extractor.fit(
                cluster_docs,
                llm_model=llm_model,
                embedding_model=embedding_model
            )
            self.cluster_topic_extractors[cluster_idx] = cluster_extractor
            
            # Map cluster topics to global topics
            for local_topic_idx, label in enumerate(cluster_topic_labels):
                global_label = f"c{cluster_idx}_{label}"
                all_topic_labels.append(global_label)
                self.topic_to_cluster[global_topic_idx] = cluster_idx
                
                # Copy topic sentences and words
                if cluster_extractor.topic_sentences and local_topic_idx < len(cluster_extractor.topic_sentences):
                    self.topic_sentences.append(cluster_extractor.topic_sentences[local_topic_idx])
                else:
                    self.topic_sentences.append(global_label)
                    
                if cluster_extractor.topic_words and local_topic_idx < len(cluster_extractor.topic_words):
                    self.topic_words.append(cluster_extractor.topic_words[local_topic_idx])
                else:
                    self.topic_words.append([])
                
                # Fill in topic-document scores for documents in this cluster
                for local_doc_idx, global_doc_idx in enumerate(cluster_doc_indices):
                    topic_doc_matrix[global_topic_idx, global_doc_idx] = cluster_topic_doc_matrix[local_topic_idx, local_doc_idx]
                
                global_topic_idx += 1
        
        # Store topic sentences in JSON
        if topic_sentences_path:
            topic_sentences_data = {
                "topic_sentences": self.topic_sentences,
                "topic_words": self.topic_words,
                "topic_labels": all_topic_labels,
                "n_clusters": self.n_clusters,
                "topic_to_cluster": self.topic_to_cluster
            }
            with open(topic_sentences_path, 'w') as f:
                json.dump(topic_sentences_data, f, indent=2)
            logger.info(f"Topic sentences saved to {topic_sentences_path}")
        
        # Pre-compute topic sentence embeddings
        if embedding_model is not None and self.topic_sentences:
            logger.info("Computing topic sentence embeddings...")
            self.topic_sentence_embeddings = np.array(
                embedding_model.batch_encode(self.topic_sentences, norm=True)
            )
        
        return topic_doc_matrix, all_topic_labels
    
    def load_topic_sentences(self, topic_sentences_path: str, embedding_model: BaseEmbeddingModel = None):
        """Load topic sentences from JSON file."""
        if os.path.exists(topic_sentences_path):
            with open(topic_sentences_path, 'r') as f:
                data = json.load(f)
            self.topic_sentences = data.get("topic_sentences", [])
            self.topic_words = data.get("topic_words", [])
            self.topic_to_cluster = {int(k): v for k, v in data.get("topic_to_cluster", {}).items()}
            logger.info(f"Loaded {len(self.topic_sentences)} topic sentences from {topic_sentences_path}")
            
            if embedding_model is not None and self.topic_sentences:
                logger.info("Computing topic sentence embeddings...")
                self.topic_sentence_embeddings = np.array(
                    embedding_model.batch_encode(self.topic_sentences, norm=True)
                )
    
    def get_query_topic_scores(self, query: str, embedding_model: BaseEmbeddingModel = None) -> np.ndarray:
        """
        Get topic relevance scores for a query using cosine similarity with topic sentences.
        
        Args:
            query: Query string
            embedding_model: Embedding model for computing similarity
            
        Returns:
            Array of shape (n_total_topics,) with topic relevance scores
        """
        if embedding_model is not None and self.topic_sentence_embeddings is not None:
            query_embedding = np.array(embedding_model.batch_encode([query], norm=True))
            query_embedding = query_embedding.squeeze()
            scores = np.dot(self.topic_sentence_embeddings, query_embedding)
            return scores
        
        raise ValueError("ClusteredTopicExtractor requires embedding_model for query scoring")
    
    def get_topic_embedding_text(self, topic_idx: int) -> str:
        """Get text representation of a topic for embedding."""
        if self.topic_sentences and topic_idx < len(self.topic_sentences):
            return self.topic_sentences[topic_idx]
        if self.topic_words and topic_idx < len(self.topic_words):
            return " ".join(self.topic_words[topic_idx])
        return f"topic {topic_idx}"


class HippoRAG:

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None):
        """
        Initializes an instance of the class and its related components.
        
        Enhanced with topic node support for improved multi-hop QA performance.
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        # Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"HippoRAG init with config:\n  {_print_config}\n")

        # LLM and embedding model specific working directories
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        if self.global_config.openie_mode == 'online':
            self.openie = OpenIE(llm_model=self.llm_model)
        elif self.global_config.openie_mode == 'offline':
            self.openie = VLLMOfflineOpenIE(self.global_config)
        elif self.global_config.openie_mode == 'Transformers-offline':
            self.openie = TransformersOfflineOpenIE(self.global_config)

        self.graph = self.initialize_graph()

        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name)(
                    global_config=self.global_config,
                    embedding_model_name=self.global_config.embedding_model_name)
        
        self.chunk_embedding_store = EmbeddingStore(
            self.embedding_model,
            os.path.join(self.working_dir, "chunk_embeddings"),
            self.global_config.embedding_batch_size, 'chunk')
        self.entity_embedding_store = EmbeddingStore(
            self.embedding_model,
            os.path.join(self.working_dir, "entity_embeddings"),
            self.global_config.embedding_batch_size, 'entity')
        self.fact_embedding_store = EmbeddingStore(
            self.embedding_model,
            os.path.join(self.working_dir, "fact_embeddings"),
            self.global_config.embedding_batch_size, 'fact')
        
        # NEW: Topic embedding store
        self.topic_embedding_store = EmbeddingStore(
            self.embedding_model,
            os.path.join(self.working_dir, "topic_embeddings"),
            self.global_config.embedding_batch_size, 'topic')

        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.openie_results_path = os.path.join(
            self.global_config.save_dir,
            f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')

        self.rerank_filter = DSPyFilter(self)

        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0

        self.ent_node_to_chunk_ids = None
        
        # NEW: Topic-related attributes
        self.topic_extractor: Union[TopicExtractor, ClusteredTopicExtractor] = None
        self.topic_node_keys: List = []
        self.topic_node_idxs: List = []
        self.topic_embeddings: np.ndarray = None
        self.topic_to_chunk_weights: Dict[str, Dict[str, float]] = {}
        self.topic_labels: List[str] = []
        
        # Topic configuration
        self.n_topics = getattr(self.global_config, 'n_topics', 10)
        self.n_clusters = getattr(self.global_config, 'n_clusters', 50)
        self.n_topics_per_cluster = getattr(self.global_config, 'n_topics_per_cluster', 20)
        self.topic_node_weight = getattr(self.global_config, 'topic_node_weight', 0.5)
        self.topic_edge_threshold = getattr(self.global_config, 'topic_edge_threshold', 0.2)


    def initialize_graph(self):
        """Initialize graph from pickle file or create new."""
        self._graph_pickle_filename = os.path.join(self.working_dir, f"graph.pickle")
        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges")
            return preloaded_graph

    def pre_openie(self, docs: List[str]):
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")
        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)
        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k: chunks[k] for k in chunk_keys_to_process}
        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)
        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)
        assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.')

    def index(self, docs: List[str]):
        """Index documents with topic enhancement."""
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE")

        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        new_openie_rows = {k: chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict), \
            f"len(chunk_to_rows): {len(chunk_to_rows)}, len(ner_results_dict): {len(ner_results_dict)}, len(triple_results_dict): {len(triple_results_dict)}"

        chunk_ids = list(chunk_to_rows.keys())
        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

        # NEW: Extract and encode topics from corpus
        logger.info(f"Extracting Topics using Clustered CorEx")
        self._extract_and_encode_topics(docs, chunk_ids)

        logger.info(f"Constructing Graph")
        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)
        
        # NEW: Add topic edges
        self._add_topic_edges(chunk_ids)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()
            self.augment_graph()
            self.save_igraph()

    def _extract_and_encode_topics(self, docs: List[str], chunk_ids: List[str]):
        """Extract topics from corpus documents using Clustered CorEx."""
        topic_results_path = os.path.join(self.working_dir, "topic_results.json")
        topic_model_path = os.path.join(self.working_dir, "topic_extractor.pkl")
        topic_sentences_path = os.path.join(self.working_dir, "topic_sentences.json")
        
        if not self.global_config.force_index_from_scratch and os.path.exists(topic_results_path):
            logger.info("Loading existing topic results")
            with open(topic_results_path, 'r') as f:
                topic_data = json.load(f)
            
            self.topic_labels = topic_data['topic_labels']
            self.topic_to_chunk_weights = topic_data['topic_to_chunk_weights']
            topic_texts = topic_data.get('topic_texts', self.topic_labels)
            
            # Load the topic extractor (needed for query topic extraction)
            if os.path.exists(topic_model_path):
                import pickle
                with open(topic_model_path, 'rb') as f:
                    self.topic_extractor = pickle.load(f)
                # Load topic sentences if available
                if os.path.exists(topic_sentences_path):
                    self.topic_extractor.load_topic_sentences(topic_sentences_path, embedding_model=self.embedding_model)
                logger.info("Loaded existing TopicExtractor model")
            else:
                # Need to refit for query processing
                logger.info("Refitting ClusteredTopicExtractor for query processing")
                self.topic_extractor = ClusteredTopicExtractor(
                    n_clusters=self.n_clusters,
                    n_topics_per_cluster=self.n_topics_per_cluster
                )
                self.topic_extractor.fit(
                    docs, 
                    topic_sentences_path=topic_sentences_path, 
                    llm_model=self.llm_model, 
                    embedding_model=self.embedding_model
                )
                import pickle
                with open(topic_model_path, 'wb') as f:
                    pickle.dump(self.topic_extractor, f)
        else:
            total_topics = self.n_clusters * self.n_topics_per_cluster
            logger.info(f"Extracting {total_topics} topics ({self.n_topics_per_cluster} per cluster) from {len(docs)} documents across {self.n_clusters} clusters")
            
            # Initialize and fit clustered topic extractor
            self.topic_extractor = ClusteredTopicExtractor(
                n_clusters=self.n_clusters,
                n_topics_per_cluster=self.n_topics_per_cluster
            )
            topic_doc_matrix, self.topic_labels = self.topic_extractor.fit(
                docs, 
                topic_sentences_path=topic_sentences_path,
                llm_model=self.llm_model,
                embedding_model=self.embedding_model,
            )
            
            # Save the topic extractor
            import pickle
            with open(topic_model_path, 'wb') as f:
                pickle.dump(self.topic_extractor, f)
            logger.info(f"Saved ClusteredTopicExtractor model to {topic_model_path}")
            
            # Build topic to chunk weights mapping
            self.topic_to_chunk_weights = {}
            topic_texts = []
            
            for topic_idx in range(len(self.topic_labels)):
                topic_key = compute_mdhash_id(self.topic_labels[topic_idx], prefix="topic-")
                topic_text = self.topic_extractor.get_topic_embedding_text(topic_idx)
                topic_texts.append(topic_text)
                
                self.topic_to_chunk_weights[topic_key] = {}
                
                weight_factor = 10
                for doc_idx, chunk_id in enumerate(chunk_ids):
                    weight = float(topic_doc_matrix[topic_idx, doc_idx])
                    if weight > self.topic_edge_threshold:
                        self.topic_to_chunk_weights[topic_key][chunk_id] = weight_factor*weight
            
            # Save topic results
            topic_data = {
                'topic_labels': self.topic_labels,
                'topic_texts': topic_texts,
                'topic_to_chunk_weights': self.topic_to_chunk_weights,
                'n_clusters': self.n_clusters,
                'n_topics_per_cluster': self.n_topics_per_cluster
            }
            with open(topic_results_path, 'w') as f:
                json.dump(topic_data, f)
            logger.info(f"Topic results saved to {topic_results_path}")
        
        # Encode topic texts (now using topic sentences)
        logger.info(f"Encoding {len(topic_texts)} topics")
        self.topic_embedding_store.insert_strings(topic_texts)
        
        logger.info(f"Extracted {len(topic_texts)} topics with {sum(len(v) for v in self.topic_to_chunk_weights.values())} topic-document edges")

    def _add_topic_edges(self, chunk_ids: List[str]):
        """Add edges between topic nodes and passage nodes."""
        logger.info("Adding topic-passage edges to graph")
        num_topic_edges = 0
        for topic_key, chunk_weights in self.topic_to_chunk_weights.items():
            for chunk_key, weight in chunk_weights.items():
                if chunk_key in chunk_ids:
                    self.node_to_node_stats[(topic_key, chunk_key)] = weight
                    self.node_to_node_stats[(chunk_key, topic_key)] = weight
                    num_topic_edges += 1
        logger.info(f"Added {num_topic_edges} topic-passage edges")

    def delete(self, docs_to_delete: List[str]):
        """Delete documents from all data structures."""
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]
        chunk_ids_to_delete = set([self.chunk_embedding_store.text_to_hash_id[chunk] for chunk in docs_to_delete])

        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        triples_to_delete = []
        all_openie_info_with_deletes = []

        for openie_doc in all_openie_info:
            if openie_doc['idx'] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc['extracted_triples'])
            else:
                all_openie_info_with_deletes.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)
        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(text_processing(list(triple)))
            doc_ids = self.proc_triples_to_docs[str(proc_triple)]
            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)
            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [[text_processing(list(triple)) for triple in true_triples_to_delete]]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(processed_true_triples_to_delete)
        triple_ids_to_delete = set([self.fact_embedding_store.text_to_hash_id[str(triple)] for triple in processed_true_triples_to_delete])

        ent_ids_to_delete = [self.entity_embedding_store.text_to_hash_id[ent] for ent in entities_to_delete]
        filtered_ent_ids_to_delete = []
        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node]
            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)
            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        # Handle topic node deletion
        topic_ids_to_delete = []
        for topic_key, chunk_weights in self.topic_to_chunk_weights.items():
            associated_chunks = set(chunk_weights.keys())
            remaining_chunks = associated_chunks.difference(chunk_ids_to_delete)
            if len(remaining_chunks) == 0:
                topic_ids_to_delete.append(topic_key)

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks, {len(triple_ids_to_delete)} Triples, {len(filtered_ent_ids_to_delete)} Entities, {len(topic_ids_to_delete)} Topics")

        self.save_openie_results(all_openie_info_with_deletes)
        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)
        self.topic_embedding_store.delete(topic_ids_to_delete)

        nodes_to_delete = list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete) + topic_ids_to_delete
        self.graph.delete_vertices(nodes_to_delete)
        self.save_igraph()
        self.ready_to_retrieve = False

    def retrieve(self, queries: List[str], num_to_retrieve: int = None, gold_docs: List[List[str]] = None):
        """Retrieve documents with topic-enhanced PPR."""
        retrieve_start_time = time.time()

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)
        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            rerank_start = time.time()
            query_fact_scores = self.get_fact_scores(query)
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
            rerank_end = time.time()
            self.rerank_time += rerank_end - rerank_start

            if len(top_k_facts) == 0:
                logger.info('No facts found after reranking, return DPR results')
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities_and_topics(
                    query=query,
                    link_top_k=self.global_config.linking_top_k,
                    query_fact_scores=query_fact_scores,
                    top_k_facts=top_k_facts,
                    top_k_fact_indices=top_k_fact_indices,
                    passage_node_weight=self.global_config.passage_node_weight)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]
            retrieval_results.append(QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()
        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s, Recognition Memory Time {self.rerank_time:.2f}s, PPR Time {self.ppr_time:.2f}s")

        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, _ = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[r.docs for r in retrieval_results], k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")
            return retrieval_results, overall_retrieval_result
        return retrieval_results

    def rag_qa(self, queries, gold_docs=None, gold_answers=None):
        """RAG QA with topic enhancement."""
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        overall_retrieval_result = None
        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve(queries=queries)

        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        if gold_answers is not None:
            overall_qa_em_result, _ = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[q.answer for q in queries_solutions], aggregation_fn=np.max)
            overall_qa_f1_result, _ = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[q.answer for q in queries_solutions], aggregation_fn=np.max)
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_em_result.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]
            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        return queries_solutions, all_response_message, all_metadata

    def retrieve_dpr(self, queries: List[str], num_to_retrieve: int = None, gold_docs: List[List[str]] = None):
        """DPR retrieval without graph enhancement."""
        retrieve_start_time = time.time()
        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k
        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()
        self.get_query_embeddings(queries)
        retrieval_results = []
        for query in tqdm(queries, desc="Retrieving"):
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]
            retrieval_results.append(QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))
        self.all_retrieval_time += time.time() - retrieve_start_time
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, _ = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[r.docs for r in retrieval_results], k_list=k_list)
            return retrieval_results, overall_retrieval_result
        return retrieval_results

    def rag_qa_dpr(self, queries, gold_docs=None, gold_answers=None):
        """RAG QA with DPR only."""
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)
        overall_retrieval_result = None
        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve_dpr(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve_dpr(queries=queries)
        queries_solutions, all_response_message, all_metadata = self.qa(queries)
        if gold_answers is not None:
            overall_qa_em_result, _ = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[q.answer for q in queries_solutions], aggregation_fn=np.max)
            overall_qa_f1_result, _ = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[q.answer for q in queries_solutions], aggregation_fn=np.max)
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_em_result.items()}
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]
            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List[QuerySolution]):
        """Execute QA inference."""
        all_qa_messages = []
        for query_solution in tqdm(queries, desc="Collecting QA prompts"):
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]
            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Wikipedia Title: {passage}\n\n'
            prompt_user += 'Question: ' + query_solution.question + '\nThought: '
            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                prompt_dataset_name = self.global_config.dataset
            else:
                prompt_dataset_name = 'musique'
            all_qa_messages.append(self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))

        all_qa_results = [self.llm_model.infer(qa_messages) for qa_messages in tqdm(all_qa_messages, desc="QA Reading")]
        all_response_message, all_metadata, _ = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        queries_solutions = []
        for idx, query_solution in enumerate(queries):
            response_content = all_response_message[idx]
            try:
                pred_ans = response_content.split('Answer:')[1].strip()
            except:
                pred_ans = response_content
            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)
        return queries_solutions, all_response_message, all_metadata

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """Add fact edges from triples to graph."""
        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")
        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()
            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)
                    node_key = compute_mdhash_id(content=triple[0], prefix="entity-")
                    node_2_key = compute_mdhash_id(content=triple[2], prefix="entity-")
                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get((node_key, node_2_key), 0.0) + 1
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get((node_2_key, node_key), 0.0) + 1
                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)
                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(set([chunk_key]))

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """Add edges connecting passage nodes to phrase nodes."""
        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()
        num_new_chunks = 0
        logger.info(f"Connecting passage nodes to phrase nodes.")
        for idx, chunk_key in tqdm(enumerate(chunk_ids)):
            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")
                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0
                num_new_chunks += 1
        return num_new_chunks

    def add_synonymy_edges(self):
        """Add synonymy edges between similar nodes."""
        logger.info(f"Expanding graph with synonymy edges")
        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())
        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")
        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)
        query_node_key2knn_node_keys = retrieve_knn(
            query_ids=entity_node_keys, key_ids=entity_node_keys, query_vecs=entity_embs, key_vecs=entity_embs,
            k=self.global_config.synonymy_edge_topk,
            query_batch_size=self.global_config.synonymy_edge_query_batch_size,
            key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            entity = self.entity_id_to_row[node_key]["content"]
            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]
                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break
                    nn_phrase = self.entity_id_to_row[nn]["content"]
                    if nn != node_key and nn_phrase != '':
                        self.node_to_node_stats[(node_key, nn)] = score
                        num_nns += 1

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """Load existing OpenIE results."""
        chunk_keys_to_save = set()
        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])
            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)
            all_openie_info = renamed_openie_info
            existing_openie_keys = set([info['idx'] for info in all_openie_info])
            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys
        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self, all_openie_info, chunks_to_save, ner_results_dict, triple_results_dict):
        """Merge OpenIE results."""
        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            try:
                chunk_openie_info = {
                    'idx': chunk_key, 'passage': passage,
                    'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                    'extracted_triples': triple_results_dict[chunk_key].triples}
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_key}: {e}")
                chunk_openie_info = {'idx': chunk_key, 'passage': passage, 'extracted_entities': [], 'extracted_triples': []}
            all_openie_info.append(chunk_openie_info)
        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """Save OpenIE results to JSON."""
        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])
        if len(all_openie_info) > 0:
            avg_ent_chars = round(sum_phrase_chars / num_phrases, 4) if num_phrases > 0 else 0
            avg_ent_words = round(sum_phrase_words / num_phrases, 4) if num_phrases > 0 else 0
            openie_dict = {'docs': all_openie_info, 'avg_ent_chars': avg_ent_chars, 'avg_ent_words': avg_ent_words}
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """Add new nodes and edges to graph."""
        self.add_new_nodes()
        self.add_new_edges()
        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """Add nodes from all embedding stores."""
        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}
        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()
        topic_to_row = self.topic_embedding_store.get_all_id_to_rows()
        node_to_rows = {**entity_to_row, **passage_to_row, **topic_to_row}
        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)
        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """Add edges from node_to_node_stats to graph."""
        edge_source_node_keys, edge_target_node_keys, edge_metadata = [], [], []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]:
                continue
            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({"weight": weight})
        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for src, tgt, meta in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if src in current_node_ids and tgt in current_node_ids:
                valid_edges.append((src, tgt))
                valid_weights["weight"].append(meta.get("weight", 1.0))
        self.graph.add_edges(valid_edges, attributes=valid_weights)

    def save_igraph(self):
        logger.info(f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges")
        self.graph.write_pickle(self._graph_pickle_filename)

    def get_graph_info(self) -> Dict:
        """Get graph statistics including topic nodes."""
        graph_info = {}
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))
        topic_nodes_keys = self.topic_embedding_store.get_all_ids()
        graph_info["num_topic_nodes"] = len(set(topic_nodes_keys))
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"] + graph_info["num_topic_nodes"]
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())
        graph_info["num_total_triples"] = len(self.node_to_node_stats)
        return graph_info

    def prepare_retrieval_objects(self):
        """Prepare objects for fast retrieval."""
        logger.info("Preparing for fast retrieval.")
        self.query_to_embedding = {'triple': {}, 'passage': {}, 'topic': {}}
        self.entity_node_keys = list(self.entity_embedding_store.get_all_ids())
        self.passage_node_keys = list(self.chunk_embedding_store.get_all_ids())
        self.fact_node_keys = list(self.fact_embedding_store.get_all_ids())
        self.topic_node_keys = list(self.topic_embedding_store.get_all_ids())

        expected_node_count = len(self.entity_node_keys) + len(self.passage_node_keys) + len(self.topic_node_keys)
        actual_node_count = self.graph.vcount()
        if expected_node_count != actual_node_count:
            logger.warning(f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}")
            if actual_node_count == 0 and expected_node_count > 0:
                self.add_new_nodes()
                self.save_igraph()

        try:
            igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
            self.node_name_to_vertex_idx = igraph_name_to_idx
            missing_nodes = [k for k in self.entity_node_keys + self.passage_node_keys + self.topic_node_keys if k not in igraph_name_to_idx]
            if missing_nodes:
                logger.warning(f"Missing {len(missing_nodes)} nodes in graph")
                self.add_new_nodes()
                self.save_igraph()
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx
            self.entity_node_idxs = [igraph_name_to_idx[k] for k in self.entity_node_keys]
            self.passage_node_idxs = [igraph_name_to_idx[k] for k in self.passage_node_keys]
            self.topic_node_idxs = [igraph_name_to_idx[k] for k in self.topic_node_keys]
        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs, self.passage_node_idxs, self.topic_node_idxs = [], [], []

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))
        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))
        self.topic_embeddings = np.array(self.topic_embedding_store.get_embeddings(self.topic_node_keys))

        # Load topic model
        topic_results_path = os.path.join(self.working_dir, "topic_results.json")
        topic_model_path = os.path.join(self.working_dir, "topic_extractor.pkl")
        topic_sentences_path = os.path.join(self.working_dir, "topic_sentences.json")
        if os.path.exists(topic_results_path):
            with open(topic_results_path, 'r') as f:
                topic_data = json.load(f)
            self.topic_to_chunk_weights = topic_data['topic_to_chunk_weights']
            self.topic_labels = topic_data['topic_labels']
        if os.path.exists(topic_model_path):
            import pickle
            with open(topic_model_path, 'rb') as f:
                self.topic_extractor = pickle.load(f)
            # Load topic sentences if available
            if os.path.exists(topic_sentences_path):
                self.topic_extractor.load_topic_sentences(topic_sentences_path, embedding_model=self.embedding_model)
            logger.info("Loaded TopicExtractor for query processing")

        all_openie_info, _ = self.load_existing_openie([])
        self.proc_triples_to_docs = {}
        for doc in all_openie_info:
            triples = flatten_facts([doc['extracted_triples']])
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple), set()).union({doc['idx']})

        if self.ent_node_to_chunk_ids is None:
            ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)
            for chunk_id in self.passage_node_keys:
                if chunk_id not in ner_results_dict:
                    ner_results_dict[chunk_id] = NerRawOutput(chunk_id=chunk_id, response=None, metadata={}, unique_entities=[])
                if chunk_id not in triple_results_dict:
                    triple_results_dict[chunk_id] = TripleRawOutput(chunk_id=chunk_id, response=None, metadata={}, triples=[])
            chunk_triples = [[text_processing(t) for t in triple_results_dict[cid].triples] for cid in self.passage_node_keys]
            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            self.add_fact_edges(self.passage_node_keys, chunk_triples)

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries):
        """Get embeddings for queries."""
        all_query_strings = []
        for query in queries:
            q_str = query.question if isinstance(query, QuerySolution) else query
            if q_str not in self.query_to_embedding['triple'] or q_str not in self.query_to_embedding['passage']:
                all_query_strings.append(q_str)
        if len(all_query_strings) > 0:
            logger.info(f"Encoding {len(all_query_strings)} queries")
            for instr_type in ['triple', 'passage', 'topic']:
                instr = get_query_instruction('query_to_fact' if instr_type == 'triple' else 'query_to_passage')
                embeddings = self.embedding_model.batch_encode(all_query_strings, instruction=instr, norm=True)
                for query, emb in zip(all_query_strings, embeddings):
                    self.query_to_embedding[instr_type][query] = emb

    def get_fact_scores(self, query: str) -> np.ndarray:
        """Get fact scores for query."""
        query_embedding = self.query_to_embedding['triple'].get(query)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query, instruction=get_query_instruction('query_to_fact'), norm=True)
        if len(self.fact_embeddings) == 0:
            return np.array([])
        try:
            scores = np.dot(self.fact_embeddings, query_embedding.T)
            scores = np.squeeze(scores) if scores.ndim == 2 else scores
            return min_max_normalize(scores)
        except:
            return np.array([])

    def get_topic_scores_from_embedding(self, query: str) -> np.ndarray:
        """Get topic scores using embedding similarity."""
        query_embedding = self.query_to_embedding['topic'].get(query)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query, instruction=get_query_instruction('query_to_passage'), norm=True)
        if len(self.topic_embeddings) == 0:
            return np.array([])
        try:
            scores = np.dot(self.topic_embeddings, query_embedding.T)
            scores = np.squeeze(scores) if scores.ndim == 2 else scores
            return min_max_normalize(scores)
        except:
            return np.array([])

    def get_topic_scores_from_corex(self, query: str) -> np.ndarray:
        """Extract topic scores directly from query using CorEx model or cosine similarity with topic sentences."""
        if self.topic_extractor is None:
            return self.get_topic_scores_from_embedding(query)
        try:
            scores = self.topic_extractor.get_query_topic_scores(query, embedding_model=self.embedding_model)
            return min_max_normalize(scores)
        except Exception as e:
            logger.error(f"Error extracting topics from query using CorEx: {str(e)}")
            return self.get_topic_scores_from_embedding(query)

    def get_combined_topic_scores(self, query: str, corex_weight: float = 0.6) -> np.ndarray:
        """Combine CorEx and embedding-based topic scores."""
        corex_scores = self.get_topic_scores_from_corex(query)
        embedding_scores = self.get_topic_scores_from_embedding(query)
        
        if len(corex_scores) == 0:
            return embedding_scores
        if len(embedding_scores) == 0:
            return corex_scores
        
        # Handle shape mismatch - use the smaller size (embedding-based)
        # This happens when some topics have duplicate/empty labels
        n_topics = min(len(corex_scores), len(embedding_scores))
        if len(corex_scores) != len(embedding_scores):
            logger.debug(f"Topic score shape mismatch: corex={len(corex_scores)}, embedding={len(embedding_scores)}. Using {n_topics} topics.")
            corex_scores = corex_scores[:n_topics]
            embedding_scores = embedding_scores[:n_topics]
        
        combined = corex_weight * corex_scores + (1 - corex_weight) * embedding_scores
        return min_max_normalize(combined)

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """Dense passage retrieval."""
        query_embedding = self.query_to_embedding['passage'].get(query)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query, instruction=get_query_instruction('query_to_passage'), norm=True)
        scores = np.dot(self.passage_embeddings, query_embedding.T)
        scores = np.squeeze(scores) if scores.ndim == 2 else scores
        scores = min_max_normalize(scores)
        sorted_ids = np.argsort(scores)[::-1]
        return sorted_ids, scores[sorted_ids]

    def get_top_k_weights(self, link_top_k, all_phrase_weights, linking_score_map):
        """Filter to top-k phrase weights."""
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])
        top_k_phrases_keys = set([compute_mdhash_id(p, prefix="entity-") for p in linking_score_map.keys()])
        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities_and_topics(self, query, link_top_k, query_fact_scores, top_k_facts, top_k_fact_indices, passage_node_weight=0.05):
        """Enhanced graph search with entity, passage, and topic nodes."""
        linking_score_map = {}
        phrase_scores = {}
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        topic_weights = np.zeros(len(self.graph.vs['name']))
        number_of_occurs = np.zeros(len(self.graph.vs['name']))
        phrases_and_ids = set()

        # Entity weights from facts
        for rank, f in enumerate(top_k_facts):
            fact_score = query_fact_scores[top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores
            for phrase in [f[0].lower(), f[2].lower()]:
                phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key)
                if phrase_id is not None:
                    weighted_score = fact_score
                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        weighted_score /= len(self.ent_node_to_chunk_ids[phrase_key])
                    phrase_weights[phrase_id] += weighted_score
                    number_of_occurs[phrase_id] += 1
                phrases_and_ids.add((phrase, phrase_id))

        number_of_occurs[number_of_occurs == 0] = 1
        phrase_weights /= number_of_occurs

        for phrase, phrase_id in phrases_and_ids:
            if phrase not in phrase_scores:
                phrase_scores[phrase] = []
            if phrase_id is not None:
                phrase_scores[phrase].append(phrase_weights[phrase_id])

        for phrase, scores in phrase_scores.items():
            if scores:
                linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k, phrase_weights, linking_score_map)

        # Topic weights using combined CorEx + embedding scores
        query_topic_scores = self.get_combined_topic_scores(query)
        if len(query_topic_scores) > 0 and len(self.topic_node_keys) > 0:
            num_topics = min(link_top_k, len(query_topic_scores))
            top_topic_indices = np.argsort(query_topic_scores)[-num_topics:][::-1]
            for topic_idx in top_topic_indices:
                if topic_idx < len(self.topic_node_keys):
                    topic_key = self.topic_node_keys[topic_idx]
                    topic_score = query_topic_scores[topic_idx]
                    topic_node_id = self.node_name_to_vertex_idx.get(topic_key)
                    if topic_node_id is not None and topic_score > 0.1:
                        topic_weights[topic_node_id] = topic_score # * self.topic_node_weight

        # Passage weights from DPR
        dpr_sorted_ids, dpr_sorted_scores = self.dense_passage_retrieval(query)
        normalized_dpr_scores = min_max_normalize(dpr_sorted_scores)
        for i, doc_id in enumerate(dpr_sorted_ids.tolist()):
            passage_key = self.passage_node_keys[doc_id]
            passage_node_id = self.node_name_to_vertex_idx[passage_key]
            passage_weights[passage_node_id] = normalized_dpr_scores[i] * passage_node_weight

        # Combine all weights for PPR
        node_weights = phrase_weights + passage_weights + topic_weights
        assert sum(node_weights) > 0, f'No weights found for query'

        ppr_start = time.time()
        ppr_sorted_ids, ppr_sorted_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        self.ppr_time += time.time() - ppr_start

        return ppr_sorted_ids, ppr_sorted_scores

    def graph_search_with_fact_entities(self, query, link_top_k, query_fact_scores, top_k_facts, top_k_fact_indices, passage_node_weight=0.05):
        """Backward compatible method - delegates to topic-enhanced version."""
        return self.graph_search_with_fact_entities_and_topics(query, link_top_k, query_fact_scores, top_k_facts, top_k_fact_indices, passage_node_weight)

    def rerank_facts(self, query: str, query_fact_scores: np.ndarray):
        """Rerank facts based on query relevance."""
        link_top_k = self.global_config.linking_top_k
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
        try:
            if len(query_fact_scores) <= link_top_k:
                candidate_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                candidate_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
            real_ids = [self.fact_node_keys[i] for i in candidate_indices]
            fact_rows = self.fact_embedding_store.get_rows(real_ids)
            candidate_facts = [eval(fact_rows[id]['content']) for id in real_ids]
            top_k_indices, top_k_facts, _ = self.rerank_filter(query, candidate_facts, candidate_indices, len_after_rerank=link_top_k)
            return top_k_indices, top_k_facts, {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}

    def run_ppr(self, reset_prob: np.ndarray, damping: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Run Personalized PageRank."""
        if damping is None:
            damping = 0.5
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping, directed=False, weights='weight',
            reset=reset_prob, implementation='prpack')
        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_ids = np.argsort(doc_scores)[::-1]
        return sorted_ids, doc_scores[sorted_ids]
