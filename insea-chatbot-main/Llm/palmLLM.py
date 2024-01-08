from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import google.generativeai as palm 
from dotenv import load_dotenv
import os

load_dotenv()

PALM_API = os.getenv("PALM_API")

palm.configure(api_key=PALM_API)

model_name = 'models/text-bison-001'


class PalmLLM(LLM):
    
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        
        
        completion = palm.generate_text(
            model=model_name,
            prompt=prompt,
            temperature=0,
            max_output_tokens=800,
        )
        
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        return completion.result
        
