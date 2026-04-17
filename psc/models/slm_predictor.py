import requests
import json
import numpy as np
from typing import Tuple, List

class SLMPredictor:
    """
    A predictor that uses a Small Language Model (SLM) to forecast storage accesses.
    This implementation assumes an OpenAI-compatible API (like vLLM or Ollama) 
    running on an H100.
    """
    def __init__(self, api_url="http://localhost:8000/v1/completions", model="phi3"):
        self.api_url = api_url
        self.model = model
        self.history_buffer = []
        self.max_history = 20  # Sliding window size
        
    def add_to_history(self, obj_id: int, size: int, timestamp: int):
        self.history_buffer.append((obj_id, size, timestamp))
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)

    def _construct_prompt(self) -> str:
        # Convert history to a compact string format
        # Example: "Obj:12,Size:4k,T:0; Obj:15,Size:8k,T:10; ..."
        history_str = "; ".join([f"O:{h[0]},S:{h[1]},T:{h[2]}" for h in self.history_buffer])
        prompt = f"Storage Trace: {history_str}\nPredict next ObjectID and TimeDelta:"
        return prompt

    def predict(self, obj_id: int, size: int, recency: int, frequency: int) -> Tuple[float, float]:
        """
        Uses the SLM to predict reuse probability (p_i) and time-to-next-access (t_i).
        Note: We adapt the model's sequence prediction into the engine's required metrics.
        """
        if len(self.history_buffer) < 5:
            return (0.5, 1000.0)  # Cold start fallback
            
        prompt = self._construct_prompt()
        
        # In real life, call the vLLM server on your H100
        try:
            # payload = {
            #     "model": self.model,
            #     "prompt": prompt,
            #     "max_tokens": 10,
            #     "temperature": 0
            # }
            # response = requests.post(self.api_url, json=payload, timeout=0.1) # low timeout for storage
            # result = response.json()['choices'][0]['text']
            
            # --- Mocking SLM Output for Demonstration ---
            # In a real H100 setup, the SLM would output something like "O:12, DT:50"
            # which we would parse to calculate p_i and t_i.
            p_i = 0.85 # High confidence from SLM pattern matching
            t_i = 45.0 # Predicted delta
            
        except Exception:
            return (0.5, 1000.0)
            
        return p_i, t_i

    def train(self, df):
        """
        For SLMs, 'training' is usually Offline Fine-Tuning.
        You would use your H100 to fine-tune on a massive dump of the RocksDB traces.
        """
        print("Note: SLM 'training' typically involves supervised fine-tuning (SFT) on trace datasets.")
        pass
