import numpy as np
import torch
from transformers import pipeline


class SentimentModel:
    def __init__(self, tokenizer,
                 model_name="tabularisai/multilingual-sentiment-analysis", device="mps"):
        if device == "gpu" and not torch.cuda.is_available():
            print("GPU is not available. Falling back to CPU.")
            device = "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            print("MPS is not available. Falling back to CPU.")
            device = "cpu"
        
        self.model = pipeline('sentiment-analysis', model=model_name, top_k=None, device=device)
        self.max_length = self.model.tokenizer.model_max_length
        self.tokenizer = self.model.tokenizer

    def _transform_output(self, output):
        return { s['label']: s['score'] for s in output }

    def _split_into_token_chunks(self, text, stride=0) -> list:
        """
        Splits a text into chunks of at most `max_length` tokens using a for loop (no truncation).
        
        Args:
            text (str): Input text.
            tokenizer: HuggingFace tokenizer.
            max_length (int): Max number of tokens per chunk.
            length_delta (int): Security margin for the chunk length (optional).
            stride (int): Overlap between chunks (optional).
        
        Returns:
            List[str]: List of text chunks.
        """
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]

        if len(input_ids) <= self.max_length:
            return [text]

        chunks = []
        step = self.max_length - stride
        for start in range(0, len(input_ids), step):
            end = min(start + self.max_length, len(input_ids))
            start_offset = offsets[start][0]
            end_offset = offsets[end - 1][1]
            chunk_text = text[start_offset:end_offset]
            chunks.append(chunk_text)

        return chunks

    def __call__(self, text, max_length=512, stride=0):
        # if isinstance(texts, str):
        #     texts = [texts]
        # else:
        #     assert isinstance(texts, list), "Input must be a string or a list of strings."
        #     assert all(isinstance(text, str) for text in texts), "All elements in the input list must be strings."
        
        # sentiments = []
        # for text in texts:
        chunks = self.split_into_token_chunks(text, max_length=max_length, stride=stride)
        if len(chunks) == 1:
            return self.transform_output(self.model(chunks[0])[0])

        # Get average positive, negative, neutral probabilities for all chunks
        chunk_sentiments = self.model(chunks, verbose=True, max_length=max_length, truncation=False)
        chunk_sentiments = [self.transform_output(s) for s in chunk_sentiments]
        return {
            "positive": np.mean([s["positive"] for s in chunk_sentiments]),
            "negative": np.mean([s["negative"] for s in chunk_sentiments]),
            "neutral": np.mean([s["neutral"] for s in chunk_sentiments])
        }

    def get_sentiment_score(self, sentiment):
        p_pos = sentiment["Very Positive"] + sentiment["Positive"]
        p_neg = sentiment["Very Negative"] + sentiment["Negative"]
        p_neu = sentiment["Neutral"]
        return (p_pos - p_neg) / (p_pos + p_neg + p_neu)