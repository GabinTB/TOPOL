import os
from dotenv import load_dotenv; load_dotenv()
import time
import numpy as np
from openai import OpenAI
from typing import List, Optional
import concurrent.futures
from tqdm import tqdm
from src.helpers.requests import rate_limit

OPENAI_KEY = os.getenv("OPEN_AI_API")
assert OPENAI_KEY is not None, "Please set the OpenAI API key in the environment variables."

OPENAI_CLIENT = OpenAI(api_key=OPENAI_KEY)


def make_chat_completion(client: OpenAI, rpm: int=500, default_model: str="gpt-4o-mini-2024-07-18") -> callable:
    @rate_limit(calls_per_minute=rpm)
    def chat_completion(prompt: str, model: str = default_model, temperature: float = 0.7,
                        max_tokens: int = 100, verbose: bool = True) -> str:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            if verbose:
                print(f"Response: {response.choices[0].message.content}")
                print(f"Tokens used: {response.usage.total_tokens}")
                cost = response.usage.total_tokens * 0.000002 / 1000
                print(f"Cost of the request: ${cost:.6f}")
            return response.choices[0].message.content
        except Exception as e:
            raise e
    return chat_completion

chat_completion: callable = make_chat_completion(OPENAI_CLIENT, rpm=500, default_model="gpt-4o-mini-2024-07-18")


class OpenAIEmbedding:
    BATCH_LIMIT = 2048
    RPM = 3000
    MAX_LIMITS = 8191

    def __init__(self, model='text-embedding-3-small'):
        self.client = OPENAI_CLIENT
        self.model = model
        self._create_embedding = rate_limit(lambda: self.RPM)(self._create_embedding)
        self.get_embeddings = rate_limit(lambda: self.RPM)(self.get_embeddings)

    def print_models(self):
        """
        Print available models for embedding.
        """
        models = self.client.models.list()
        print("Available models:")
        for model in models:
            print(model.id)
        
    # @rate_limit(calls_per_minute=lambda self: self.RPM)
    def _create_embedding(self, text: str) -> Optional[np.ndarray]:
        assert isinstance(text, str), "Input text must be a string."
        try:
            response = self.client.embeddings.create(input=text, model=self.model)
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error: {e}")
            return None

    
    def get_embedding(self, text: str, pooling: str = "mean") -> Optional[np.ndarray]:
        if len(text) > self.MAX_LIMITS:
            batch_size = self.MAX_LIMITS - 1
            embeddings = []
            batches = [text[i : min(len(text), i + batch_size)] for i in range(0, len(text), batch_size)]
            for batch in batches:
                embeddings.append(self._create_embedding(batch))
            if pooling == "mean":
                return np.mean(np.array(embeddings), axis=0)
            elif pooling == "max":
                return np.max(np.array(embeddings), axis=0)
        else:
            return self._create_embedding(text)
    
    # @rate_limit(calls_per_minute=lambda self: self.RPM)
    def get_embeddings(self, texts: List[str],  dtype=np.double, fast=True, **kwargs) -> Optional[np.ndarray]:
        total_text_size_limit = self.MAX_LIMITS - 1
        batches = []
        current_batch = []
        current_sum = 0
        for text in texts:
            text = text[:min(len(text), total_text_size_limit+1)].replace("\n", " ")
            N = len(text)
            if current_sum + len(text) > total_text_size_limit:
                if len(current_batch):
                    batches.append(current_batch)
                current_batch = [text]
                current_sum = N
            else:
                current_batch.append(text)
                current_sum += N
        if len(current_batch):
            batches.append(current_batch)
        
        embeddings = []
        for batch in tqdm(batches, desc="Encoding batches (get_embeddings)", total=len(batches)):
        # for batch in batches:
            try:
                response = self.client.embeddings.create(input=batch, model=self.model, **kwargs)
                batch_embeddings = [d.embedding for d in response.data]
                assert len(batch_embeddings) == len(batch), f"Batch embeddings and batch text sizes mismatch: {len(batch_embeddings)} != {len(batch)}"
                embeddings.extend(batch_embeddings)
            except Exception as e:
                raise e
        assert len(embeddings) == len(texts), f"Embeddings and text sizes mismatch: {len(embeddings)} != {len(texts)}"
        
        return np.array(embeddings).astype(dtype)
        
    
    def encode_list(self, texts: list[str], dtype=np.double, **kwargs):

        assert isinstance(texts, list), "Input texts must be a list of strings."
        
        if len(texts) > self.BATCH_LIMIT:
            batches = [texts[i : min(len(texts), i + self.BATCH_LIMIT)] for i in range(0, len(texts), self.BATCH_LIMIT)]
            embeddings = []
            for batch in tqdm(batches, desc="Encoding batches", total=len(batches)):
                for text in tqdm(batch, desc="Encoding texts", total=len(batch)):
                    embeddings.append(self.get_embedding(text, **kwargs))
                # embeddings.extend([self.get_embedding(text, **kwargs) for text in tqdm(batch, desc="Encoding texts")])
        else:
            embeddings = [self.get_embedding(text, **kwargs) for text in texts]

        return np.array(embeddings).astype(dtype)
    
    def encode(self, texts, dtype=np.double, fast=True, **kwargs):
        if isinstance(texts, str):
            return self.get_embedding(texts, **kwargs)
        
        assert isinstance(texts, list), "Input texts must be a sring or a list of strings."
        
        if len(texts) > self.BATCH_LIMIT:
            batches = [texts[i : min(len(texts), i + self.BATCH_LIMIT)] for i in range(0, len(texts), self.BATCH_LIMIT)]
            print(f"Encoding {len(batches)} batches of {self.BATCH_LIMIT} texts (max) each.")

            embeddings = []
            if fast:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = executor.map(self.get_embeddings, batches, dtype, [kwargs] * len(batches))
                    for result in results:
                        embeddings.extend(result)
            else:
                for batch in tqdm(batches, desc="Embedding batches (encode)", total=len(batches)):
                # for batch in batches:
                    embeddings.extend(self.get_embeddings(batch, **kwargs))
        else:
            embeddings = self.get_embeddings(texts, dtype=dtype, **kwargs)

        return np.array(embeddings).astype(dtype)