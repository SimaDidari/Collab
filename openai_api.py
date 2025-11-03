# maas/provider/openai_api.py
# Add these imports at the top
import os
import torch
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer

# In the OpenAILLM class, modify the __init__ method:
class OpenAILLM(BaseLLM):
    """Check https://platform.openai.com/examples for examples"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.is_local = False  # Add this flag
        
        # Check if using local model based on config
        if config.api_type == LLMType.LOCAL or "local" in config.model.lower():
            self._init_local_model()
        else:
            self._init_client()
        
        self.auto_max_tokens = False
        self.cost_manager: Optional[CostManager] = None

    def _init_local_model(self):
        """Initialize local Qwen model"""
        model_path = self.config.base_url or os.environ.get("QWEN_MODEL_PATH", "/data1/davoud/models/Qwen2.5-14B-Instruct")
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
        self.pricing_plan = "local"  # No costs for local model
        print(f"✅ Qwen model loaded successfully")

    # Modify the acompletion method to handle local models:
    async def acompletion(
        self,
        messages: list[dict],
        stream: bool = False,
        timeout: int = USE_CONFIG_TIMEOUT,
        **kwargs,
    ) -> ChatCompletion:
        if self.is_local:
            return await self._local_completion(messages, stream, timeout, **kwargs)
        return await self._achat_completion(messages, stream, timeout, **kwargs)

    async def _local_completion(
        self,
        messages: list[dict],
        stream: bool = False,
        timeout: int = USE_CONFIG_TIMEOUT,
        **kwargs,
    ) -> ChatCompletion:
        """Handle local model completion"""
        # Convert messages to the format Qwen expects
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", 512),
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        
        temperature = kwargs.get("temperature", 0.7)
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95
        else:
            gen_kwargs["do_sample"] = False
        
        # Generate
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*generation flags.*")
            generated_ids = self.model.generate(**model_inputs, **gen_kwargs)
        
        # Decode only new tokens
        input_length = len(model_inputs.input_ids[0])
        new_tokens = generated_ids[0][input_length:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Create a ChatCompletion-like response
        from openai.types.chat import ChatCompletion, ChatCompletionMessage, Choice
        from openai.types import CompletionUsage
        
        return ChatCompletion(
            id="local-" + str(id(response_text)),
            model=self.config.model,
            object="chat.completion",
            created=int(time.time()),
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=response_text.strip()
                    ),
                    finish_reason="stop"
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=input_length,
                completion_tokens=len(new_tokens),
                total_tokens=input_length + len(new_tokens)
            )
        )
