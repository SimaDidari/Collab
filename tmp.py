# G-Memory Insight Generation System: Complete Function Analysis
## Architecture Summary
The system uses a three-tier hierarchical structure:
1. **Interaction Graph**: Individual agent messages and state transitions
2. **Query Graph**: Task-level trajectories and relationships
3. **Insight Graph**: High-level rules extracted from successful/failed patterns

---
## Core Classes and Initialization
### `GMemory.__post_init__()`
**Purpose**: Initializes the G-Memory system with all necessary components.

**Key Components**:
- **Main Memory**: Chroma vector database for storing task trajectories
- **Task Layer**: Manages task relationships and graph structure
- **Insights Layer**: Handles insight generation, storage, and management
- **Configuration Parameters**:
  - `_hop`: Number of hops for graph traversal (default: 1)
  - `_start_insights_threshold`: Minimum tasks before generating insights (default: 5)
  - `_rounds_per_insights`: How often to generate insights (default: every 5 tasks)
  - `_insights_point_num`: Number of insight points to generate (default: 5)

**Code Flow**:
```python
def __post_init__(self):
    # Initialize vector database for memory storage
    self.main_memory = Chroma(embedding_function=self.embedding_func, persist_directory=self.persist_dir)
    
    # Set hyperparameters from global config
    self._hop = self.global_config.get('hop', 1)
    self._start_insights_threshold = self.global_config.get('start_insights_threshold', 5)
    # ... other parameters
    
    # Initialize task and insights management layers
    self.task_layer = TaskLayer(...)
    self.insights_layer = InsightsManager(...)
```

---

## Memory Addition and Trigger System

### `GMemory.add_memory(mas_message: MASMessage)`
**Purpose**: Adds completed task trajectories to memory and triggers insight generation when conditions are met.

**Process Flow**:
1. **Sparsification**: Removes incorrect steps from trajectory
2. **Memory Storage**: Adds trajectory to both task layer and vector database
3. **Insight Triggering**: Determines when to generate or merge insights

**Trigger Conditions**:
```python
# Start generating insights after threshold reached
if self.memory_size >= self._start_insights_threshold and self.memory_size % self._rounds_per_insights == 0:
    self.insights_layer.finetune_insights(self._insights_point_num)

# Merge similar insights every 20 tasks
if self.memory_size % 20 == 0: 
    self.insights_layer.merge_insights()
```

**Key Features**:
- **Label Validation**: Ensures all messages have success/failure labels
- **Automatic Triggering**: No manual intervention needed for insight generation
- **Scalable**: Handles growing memory sizes efficiently

---

## Memory Retrieval System

### `GMemory._retrieve_memory_raw()`
**Purpose**: Retrieves relevant successful and failed trajectories for insight generation.

**Parameters**:
- `query_task`: Current task for finding similar experiences
- `successful_topk`: Number of successful examples to retrieve
- `failed_topk`: Number of failed examples to retrieve
- `threshold`: Minimum similarity score for relevance

**Retrieval Strategy**:
1. **Graph-based Retrieval**: Uses task layer to find related tasks via graph traversal
2. **Similarity Augmentation**: Fills gaps using vector similarity search
3. **Score Filtering**: Only includes trajectories above similarity threshold
4. **Balanced Sampling**: Ensures mix of successful and failed examples

**Code Logic**:
```python
def sort_and_filter_by_similarity(docs: list[Document], threshold: float = 0.3):
    # Calculate cosine similarity with query task
    # Filter by threshold and sort by relevance
    # Return top-k most relevant trajectories
```

---

## Insight Generation Core Functions

### `InsightsManager.finetune_insights(insight_point_num: int)`
**Purpose**: Main function that orchestrates the insight generation process through comparative analysis and success pattern extraction.

**Two-Phase Process**:

#### **Phase 1: Comparative Analysis**
Compares successful vs failed trajectories to understand what leads to different outcomes.

**Process**:
1. **Pair Creation**: Creates pairs of successful and failed trajectories
2. **Prompt Building**: Uses comparative analysis prompts
3. **LLM Analysis**: Gets insights on why one succeeded and the other failed
4. **Rule Extraction**: Parses LLM response for actionable rules

#### **Phase 2: Success Pattern Analysis**
Analyzes multiple successful trajectories together to find common success patterns.

**Process**:
1. **Chunk Creation**: Groups successful trajectories into manageable chunks
2. **Pattern Analysis**: Looks for common elements across successes
3. **Rule Generation**: Creates general principles from success patterns

**Key Implementation Details**:
```python
def finetune_insights(self, insight_point_num: int):
    # Retrieve relevant trajectories
    successful_trajs, failed_trajs = self._retrieve_trajectories()
    
    # Phase 1: Comparative analysis
    compare_pairs = create_compare_pairs(successful_trajs, failed_trajs)
    for pair in compare_pairs:
        # Generate comparative prompts
        # Get LLM analysis
        # Parse and update rules
    
    # Phase 2: Success pattern analysis  
    for chunk in successful_chunks:
        # Generate success pattern prompts
        # Get LLM analysis
        # Parse and update rules
```

---

## Prompt Building Functions

### `InsightsManager._build_comparative_prompts()`
**Purpose**: Creates prompts for comparing successful and failed trajectories.

**Input Processing**:
- **Task Descriptions**: What the agent was trying to accomplish
- **Trajectories**: Step-by-step execution records
- **Failure Reasons**: Why the failed trajectory didn't succeed
- **Existing Rules**: Current insight rules for context

**Prompt Structure**:
```python
def _build_comparative_prompts(self, true_traj: MASMessage, false_traj: MASMessage, insights: list[dict]):
    # Extract existing rules for context
    existing_rules = [insight['rule'] for insight in insights]
    rule_text = '\n'.join([f'{i}. {r}' for i, r in enumerate(existing_rules, 1)])
    
    # Build comparative prompt with task details, trajectories, and failure reasons
    prompt = GMemoryPrompts.critique_compare_rules_user_prompt.format(...)
    
    # Return system + user message pair
    return [Message(role='system', content=system_prompt), Message(role='user', content=prompt)]
```

**Key Features**:
- **Contextual**: Includes existing rules to avoid duplication
- **Detailed**: Provides full trajectory information for analysis
- **Structured**: Uses consistent format for reliable LLM parsing

### `InsightsManager._build_success_prompts()`
**Purpose**: Creates prompts for analyzing patterns in successful trajectories.

**Focus Areas**:
- **Common Strategies**: What successful trajectories do similarly
- **Key Decision Points**: Critical moments that led to success
- **Efficient Approaches**: Methods that work consistently well

**Implementation**:
```python
def _build_success_prompts(self, success_trajectories: Iterable[MASMessage], insights: list[dict]):
    # Format multiple successful trajectories
    history = [f'task{i}:\n' + task.task_description + task.get_extra_field('key_steps') 
              for i, task in enumerate(success_trajectories)]
    
    # Include existing rules for context
    rule_text = '\n'.join([f'{i}. {r}' for i, r in enumerate(existing_rules, 1)])
    
    # Build success analysis prompt
    prompt = GMemoryPrompts.critique_success_rules_user_prompt.format(
        success_history='\n'.join(history),
        existing_rules=rule_text
    )
```

---

## Rule Processing System

### `InsightsManager._parse_rules(llm_text: str)`
**Purpose**: Extracts structured rule operations from LLM responses.

**Supported Operations**:
- **ADD**: Create new insight rule
- **EDIT**: Modify existing rule
- **REMOVE**: Delete ineffective rule  
- **AGREE**: Reinforce existing rule

**Parsing Logic**:
```python
def _parse_rules(self, llm_text):
    # Regex pattern to extract operations and rule text
    pattern = r'((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)'
    matches = re.findall(pattern, llm_text)
    
    # Validation rules
    banned_words = ['ADD', 'AGREE', 'EDIT']  # Prevent operation words in rule text
    for operation, text in matches:
        text = text.strip()
        # Must end with period and not contain banned words
        if text != '' and not any([w in text for w in banned_words]) and text.endswith('.'):
            res.append((operation.strip(), text))
```

**Quality Controls**:
- **Format Validation**: Ensures rules follow expected format
- **Content Filtering**: Removes malformed or incomplete rules
- **Duplicate Prevention**: Avoids processing invalid operations

### `InsightsManager._update_rules()`
**Purpose**: Applies parsed rule operations to the insight memory with sophisticated conflict resolution.

**Processing Order**: `REMOVE → AGREE → EDIT → ADD`
This order ensures destructive operations happen first, then constructive ones.

**Operation Handling**:

#### **ADD Operation**:
```python
elif operation_type == 'ADD':
    # Check for duplicates before adding
    if self._is_existing_rule(operation_rule_text): 
        delete_indices.append(i)  # Skip duplicate
    else:
        # Create new rule with initial score
        meta_data = {
            'rule': operation_rule_text,
            'score': 2,  # Starting score
            'positive_correlation_tasks': list(relative_tasks),
            'negative_correlation_tasks': list()
        }
        self.insights_memory.append(meta_data)
```

#### **EDIT Operation**:
```python
elif operation_type == 'EDIT':
    if self._is_existing_rule(operation_rule_text):
        # Convert to AGREE if rule already exists
        rule_num = self._retrieve_rule_index(operation_rule_text)
        operations[i] = (f'AGREE {rule_num + 1}', operation_rule_text)
    else:
        # Update rule text and boost score
        rule_data['rule'] = operation_rule_text
        rule_data['score'] += 1
```

#### **REMOVE Operation**:
```python
elif operation_type == 'REMOVE':
    rule_index = int(operation.split(' ')[1]) - 1
    rule_data = self.insights_memory[rule_index]
    # Stronger penalty when memory is full
    remove_strength = 3 if list_full else 1
    rule_data['score'] -= remove_strength
    # Track which tasks opposed this rule
    rule_data['negative_correlation_tasks'] = list(set(
        rule_data['negative_correlation_tasks'] + relative_tasks
    ))
```

#### **AGREE Operation**:
```python
elif operation_type == 'AGREE':
    rule_index = self._retrieve_rule_index(operation_rule_text)
    rule_data = self.insights_memory[rule_index]
    rule_data['score'] += 1  # Boost score
    # Track which tasks support this rule
    rule_data['positive_correlation_tasks'] = list(set(
        rule_data['positive_correlation_tasks'] + relative_tasks
    ))
```

**Advanced Features**:
- **Adaptive Scoring**: Rules that consistently help get higher scores
- **Task Correlation Tracking**: Maintains links between rules and supporting/opposing tasks
- **Memory Management**: Handles memory limits gracefully
- **Conflict Resolution**: Smart handling of duplicate and invalid operations

---

## Duplication Prevention System

### `InsightsManager._is_existing_rule(operation_rule_text: str)`
**Purpose**: Prevents duplicate rules from being added to memory.

**Algorithm**:
```python
def _is_existing_rule(self, operation_rule_text: str) -> bool:
    for insight in self.insights_memory:
        if insight['rule'] in operation_rule_text:
            return True
    return False
```

**Smart Handling**:
- **Substring Matching**: Checks if proposed rule is contained in existing rules
- **Flexible Comparison**: Allows for minor variations while preventing true duplicates
- **Memory Efficiency**: Keeps insight memory clean and focused

### `InsightsManager._retrieve_rule_index(operation_rule_text: str)`
**Purpose**: Finds the index of an existing rule for EDIT/AGREE operations.

**Usage**:
- **EDIT Conversion**: When trying to edit to an existing rule, converts to AGREE
- **AGREE Operations**: Locates rule to reinforce
- **Error Handling**: Returns -1 if rule not found

---

## Rule Merging and Consolidation

### `InsightsManager.merge_insights()`
**Purpose**: Consolidates similar and redundant rules to maintain memory quality.

**When Triggered**: Every 20 tasks automatically

**Process**:
1. **Score-based Filtering**: Takes highest-scoring rules first
2. **LLM-based Merging**: Uses language model to identify and merge similar rules
3. **Memory Replacement**: Replaces old rules with consolidated versions

**Implementation**:
```python
def merge_insights(self):
    # Sort rules by score (best first)
    sorted_insights = sorted(self.insights_memory, key=lambda x: x['score'], reverse=True)
    
    # Take top rules up to limit
    top_insights = sorted_insights[:limited_number]
    
    # Format for LLM merging
    current_rules = [f"{i+1}. {insight['rule']}" for i, insight in enumerate(top_insights)]
    
    # Use LLM to merge similar rules
    merge_prompts = [
        Message(role='system', content=GMemoryPrompts.merge_rules_system_prompt),
        Message(role='user', content=merge_prompt)
    ]
    
    # Replace memory with merged rules
    merged_response = self.llm_model(merge_prompts)
    # Parse and update memory...
```

**Benefits**:
- **Quality Maintenance**: Removes redundant or low-value rules
- **Memory Efficiency**: Prevents insight memory from growing indefinitely
- **Rule Refinement**: Improves rule quality through consolidation

---

## Utility and Helper Functions

### `InsightsManager.clear_insights()`
**Purpose**: Removes low-scoring rules to maintain memory quality.

**Logic**:
```python
def clear_insights(self):
    # Remove rules with negative or zero scores
    self.insights_memory = [insight for insight in self.insights_memory if insight['score'] > 0]
```

**Benefits**:
- **Automatic Cleanup**: Removes rules that consistently prove unhelpful
- **Memory Optimization**: Keeps only valuable insights
- **Performance**: Reduces processing time for rule retrieval

### `InsightsManager._index_done()`
**Purpose**: Persists insight memory to disk after updates.

**Implementation**:
```python
def _index_done(self):
    write_json(self.insights_memory, self.persist_file)
```

**Reliability Features**:
- **Persistence**: Ensures insights survive system restarts
- **Backup**: Maintains historical versions of insight memory
- **Recovery**: Allows system to restore previous states if needed

---

## Advanced Features and Optimizations

### Graph-based Task Retrieval
The system uses graph traversal to find related tasks, not just similarity search:

```python
# In _retrieve_memory_raw()
task_mains = self.task_layer.retrieve_related_task(
    query_task=query_task, 
    node_num=related_point_num, 
    hop=self._hop
)
```

This enables:
- **Structural Relationships**: Finds tasks connected through agent interactions
- **Multi-hop Reasoning**: Can traverse multiple relationship links
- **Contextual Relevance**: Better than pure semantic similarity

### Adaptive Scoring System
Rules accumulate evidence over time:

- **Positive Evidence**: Tasks that succeed when following rule (+1 score)
- **Negative Evidence**: Tasks that fail when rule applied (-1 to -3 score)
- **Threshold-based Removal**: Rules below 0 score get removed
- **Memory Pressure**: Stronger penalties when memory is full

### Batch Processing Efficiency
The system processes insights in batches:

- **Compare Pairs**: Successful vs failed trajectory analysis
- **Success Chunks**: Multiple successful trajectories analyzed together
- **Balanced Load**: Prevents overwhelming LLM with too many examples

---

## Error Handling and Edge Cases

### Invalid Operation Handling
```python
# In _update_rules()
if (rule_num is None) or (rule_num > len(self.insights_memory)) or (rule_num <= 0):
    delete_indices.append(i)  # Skip invalid operations
```

### Memory Limits
```python
list_full = len(self.insights_memory) >= max_rules_num
# Adjust penalties based on memory pressure
remove_strength = 3 if list_full else 1
```

### Malformed Rules
```python
# In _parse_rules()
if text != '' and not any([w in text for w in banned_words]) and text.endswith('.'):
    # Only process well-formed rules
```

---

## Performance Considerations

### Caching Strategy
- **Insights Cache**: Frequently accessed insights kept in memory
- **Embedding Cache**: Reuses computed embeddings when possible
- **Graph Cache**: Task relationships cached for faster retrieval

### Scalability Features
- **Threshold Management**: Configurable parameters for different system sizes
- **Batch Size Limits**: Prevents memory overflow with large task sets
- **Periodic Cleanup**: Automatic removal of low-value insights

### Memory Efficiency
- **Score-based Filtering**: Only keeps valuable insights
- **Periodic Merging**: Consolidates similar rules
- **Lazy Loading**: Loads insight data only when needed

---

## Integration with Multi-Agent Systems

### Message Format Compatibility
The system works with `MASMessage` objects that contain:
- **Task Description**: What the agent was trying to accomplish
- **Task Trajectory**: Step-by-step execution record
- **Success Label**: Whether the task succeeded or failed
- **Extra Fields**: Additional metadata like failure reasons

### Agent-Specific Insights
The system can generate role-specific insights:
```python
# From prompts - project_insights_system_prompt
# Adapts general insights to specific agent roles
# Considers agent background and responsibilities
```

### Real-time Learning
- **Continuous Updates**: Learns from every completed task
- **Immediate Application**: New insights available for next task
- **Feedback Loops**: Agent performance influences future insight generation

---

## Configuration and Customization

### Key Parameters
```python
# Configurable thresholds
_start_insights_threshold = 5      # When to start generating insights
_rounds_per_insights = 5          # How often to generate insights
_insights_point_num = 5           # Number of insight points per generation
_hop = 1                          # Graph traversal depth
MAX_RULE_THRESHOLD = 10           # Maximum rules in memory
```

### Prompt Customization
The system supports custom prompts for:
- **Comparative Analysis**: How to compare success vs failure
- **Success Pattern Analysis**: How to extract success patterns
- **Rule Merging**: How to consolidate similar rules
- **Rule Formatting**: Expected format for rule operations

---

## Conclusion

The G-Memory insight generation system represents a sophisticated approach to learning from multi-agent task execution. 
Through its hierarchical memory structure, adaptive scoring system, and intelligent rule management, 
it provides a robust framework for continuous improvement in multi-agent systems.

Key strengths include:
- **Automated Learning**: No manual intervention required
- **Quality Control**: Multiple validation and filtering layers
- **Scalability**: Handles growing memory efficiently
- **Flexibility**: Configurable for different use cases
- **Reliability**: Robust error handling and persistence

The system's ability to learn from both successes and failures, while maintaining rule quality through merging and scoring, 
makes it a valuable component for any multi-agent system requiring continuous improvement and adaptation.
