import os
import torch
import warnings  # ← ADD THIS
from typing import (
    Protocol, 
    Literal,  
    Optional, 
    List,
)
from openai import OpenAI
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .utils import load_config


# model configs
CONFIG: dict = load_config("configs/configs.yaml")
LLM_CONFIG: dict = CONFIG.get("llm_config", {})
MAX_TOKEN = LLM_CONFIG.get("max_token", 512)  
TEMPERATURE = LLM_CONFIG.get("temperature", 0.1)
NUM_COMPS = LLM_CONFIG.get("num_comps", 1)

URL = os.environ.get("OPENAI_API_BASE", "")
KEY = os.environ.get("OPENAI_API_KEY", "")
print('# api url: ', URL)
print('# api key: ', KEY)


completion_tokens, prompt_tokens = 0, 0

@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

class LLMCallable(Protocol):

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        pass

class LLM(ABC):
    
    def __init__(self, model_name: str):
        self.model_name: str = model_name

    @abstractmethod
    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        pass

class GPTChat(LLM):
    """
    Unified LLM class that supports both API-based models (GPT, etc.) 
    and local Qwen models. Auto-detects based on model_name.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        
        # Check if this is a Qwen local model
        if "qwen" in model_name.lower() or model_name.startswith("local-"):
            self._init_qwen_local()
        else:
            self._init_api_client()
    
    def _init_qwen_local(self):
        """Initialize local Qwen model"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_path = "/data1/davoud/models/Qwen2.5-14B-Instruct"
        print(f"Loading Qwen model from: {model_path}")
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.is_local = True
        print(f"✅ Qwen model loaded on device: {self.model.device}")
    
    def _init_api_client(self):
        """Initialize API client for OpenAI-compatible models"""
        self.client = OpenAI(
            base_url=URL,
            api_key=KEY
        )
        self.is_local = False

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        if self.is_local:
            return self._call_local(messages, temperature, max_tokens, stop_strs, num_comps)
        else:
            return self._call_api(messages, temperature, max_tokens, stop_strs, num_comps)
    
    def _call_local(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
        stop_strs: Optional[List[str]],
        num_comps: int
    ) -> str:
        """Call local Qwen model"""
        global prompt_tokens, completion_tokens
        
        # Convert to chat format
        chat_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        text = self.tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        prompt_tokens += model_inputs.input_ids.shape[1]
        
        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": MAX_TOKEN,
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,  # ← FIX
        }
        
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95
        else:
            gen_kwargs["do_sample"] = False
        
        # Generate (suppress warnings)  ← FIX
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*generation flags.*")
            generated_ids = self.model.generate(**model_inputs, **gen_kwargs)
        
        # Decode only the new tokens
        new_tokens = generated_ids[0][len(model_inputs.input_ids[0]):]  # ← Better variable name
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        

        completion_tokens += len(new_tokens)
       
        # Handle stop strings
        if stop_strs:
            for stop_str in stop_strs:
                if stop_str in answer:
                    answer = answer.split(stop_str)[0]

        return answer.strip() if answer else ""  # ← ADD .strip()
    
    def _call_api(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
        stop_strs: Optional[List[str]],
        num_comps: int
    ) -> str:
        """Call API-based model (original implementation)"""
        import time
        global prompt_tokens, completion_tokens
        
        messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        max_retries = 5  
        wait_time = 1 

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,  
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=num_comps,
                    stop=stop_strs
                )
                
                # ← REMOVED pdb.set_trace()
                answer = response.choices[0].message.content
                prompt_tokens += response.usage.prompt_tokens
                completion_tokens += response.usage.completion_tokens
                
                if answer is None:
                    print("Error: LLM returned None")
                    continue
                return answer  

            except Exception as e:
                error_message = str(e)
                if "rate limit" in error_message.lower() or "429" in error_message:
                    time.sleep(wait_time)
                else:
                    print(f"Error during API call: {error_message}")
                    break 

        return "" 


def get_price():
    global completion_tokens, prompt_tokens
    return completion_tokens, prompt_tokens, completion_tokens*60/1000000+prompt_tokens*30/1000000