import os
import numpy as np
import google.generativeai as genai; genai.configure(api_key=os.getenv("GOOGLE_AI_API_KEY"))
from google.api_core import retry
from openai import OpenAI
from tqdm import tqdm
from src.helpers.requests import rate_limit
from src.openai_api import OpenAIEmbedding, make_chat_completion

GOOGLE_GENAI_KEY = os.getenv("GOOGLE_AI_API_KEY")
assert GOOGLE_GENAI_KEY is not None, "Please set the Google Generative AI API key in the environment variables."

GGENAI_CLIENT = OpenAI(
    api_key=GOOGLE_GENAI_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


chat_completion: callable = make_chat_completion(GGENAI_CLIENT, rpm=15, default_model="gemini-2.0-flash")


class GoogleAIEmbedding(OpenAIEmbedding):
    """
    Class to get embeddings from Google Generative AI API.
    """
    def __init__(self, model: str = "text-embedding-004", **kwargs):
        super().__init__(model=model, **kwargs)
        self.BATCH_LIMIT = 2048
        self.RPM = 15
        self.MAX_LIMITS = 8191
        self.client = GGENAI_CLIENT
    

class AdvancedGoogleAIEmbedding:
    EMBED_MODELS = [m.name.replace("models/", "") for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]
    TASKS = ['retrieval_query', 'retrieval_document', 'semantic_similarity', 'classification', 'clustering']

    def __init__(self, task, model='text-embedding-004'):
        assert model in self.EMBED_MODELS, f"Model {model} not supported. Supported models: {self.EMBED_MODELS}"
        assert task in self.TASKS, f"Task {task} not supported. Supported tasks: {self.TASKS}"
        self.model = model
        self.task = task
        pass

    def show_available_models(self):
        """
        Show available models for embedding.
        """
        print("Available models:")
        for model in self.EMBED_MODELS:
            print(f"- {model}")
    
    def show_available_tasks(self):
        """
        Show available tasks for embedding.
        """
        print("Available tasks:")
        for task in self.TASKS:
            print(f"- {task}")

    @retry.Retry(timeout=300.0)
    def _create_embedding(self, text: str) -> list[float]:
        assert isinstance(text, str), "Input text must be a string."
        try:
            response = genai.embed_content(model=self.model, content=text, task_type=self.task)
            return response['embedding']
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def get_embedding(self, text: str, pooling: str = "mean") -> list[float]:
        """
        Get the embedding for a given text.
        Args:
            text (str): The input text to embed.
            pooling (str): The pooling method to use. Options are "mean", "max", "min", "sum".
        Returns:
            list[float]: The embedding vector.
        """
        assert isinstance(text, str), "Input text must be a string."
        embedding = self._create_embedding(text)
        if embedding is None:
            return None
        if pooling == "mean":
            return np.mean(embedding, axis=0)
        elif pooling == "max":
            return np.max(embedding, axis=0)
        elif pooling == "min":
            return np.min(embedding, axis=0)
        elif pooling == "sum":
            return np.sum(embedding, axis=0)
        else:
            raise ValueError("Invalid pooling method. Choose from 'mean', 'max', 'min', 'sum'.")
        
    def get_embeddings(self, texts: list[str], dtype=np.double, fast=True, **kwargs) -> np.ndarray:
        """
        Get embeddings for a list of texts.
        Args:
            texts (list[str]): The input texts to embed.
            dtype (np.dtype): The data type of the output array.
            fast (bool): Whether to use fast mode or not.
            **kwargs: Additional arguments for the embedding function.
        Returns:
            np.ndarray: The embeddings array.
        """        
        embeddings = []
        for text in tqdm(texts, desc="Encoding batches", total=len(texts)):
            embedding = self._create_embedding(text)
            if embedding is not None:
                embeddings.append(embedding)
        
        return np.array(embeddings, dtype=dtype)
    
    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        Args:
            texts (list[str]): The input texts to encode.
            **kwargs: Additional arguments for the embedding function.
        Returns:
            np.ndarray: The encoded embeddings.
        """
        if isinstance(texts, str):
            return self.get_embedding(texts, **kwargs)
        
        assert isinstance(texts, list), "Input texts must be a sring or a list of strings."
        assert all(isinstance(text, str) for text in texts), "All elements in the list must be strings."
        assert len(texts), "Input texts must be a non-empty list."
        return self.get_embeddings(texts, **kwargs)