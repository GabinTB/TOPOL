import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline



class SentimentModel:
    def __init__(self, model_name="tabularisai/multilingual-sentiment-analysis", device=-1, max_lenghth=None, n_special_tokens=None):
        # model = AutoModelForSequenceClassification.from_pretrained(model_name) # Always use CPU for pipeline creation
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = model.to(device) # Move model to selected device manually if needed
        # self.model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
        self.model = pipeline('sentiment-analysis', model=model_name, top_k=None, device=device)
        self.max_length = self.model.tokenizer.model_max_length if max_lenghth is None else max_lenghth
        self.set_n_special_tokens(n_special_tokens)
        self.max_length = self.model.tokenizer.model_max_length
        self.set_n_special_tokens(n_special_tokens)

    def set_n_special_tokens(self, n_special_tokens=None):
        if n_special_tokens is None:
            dummy = self.model.tokenizer("test", return_offsets_mapping=False)
            self.n_special_tokens = len(dummy["input_ids"]) - len(self.model.tokenizer("test", add_special_tokens=False)["input_ids"])
        else:
            self.n_special_tokens = n_special_tokens

    def _transform_output(self, output):
        # return { s['label']: s['score'] for s in output }
        temp_output = { s['label'].lower(): s['score'] for s in output }
        if "very positive" in temp_output:
            temp_output["positive"] += temp_output["very positive"]
            del temp_output["very positive"]
        if "very negative" in temp_output:
            temp_output["negative"] += temp_output["very negative"]
            del temp_output["very negative"]
        return temp_output

    def _split_into_token_chunks(self, text, stride=0) -> list:
        """
        Splits a text into chunks of at most `max_length` tokens using a for loop (no truncation).
        
        Args:
            text (str): Input text.
            max_length (int): Max number of tokens per chunk.
            stride (int): Overlap between chunks (optional).
        
        Returns:
            List[str]: List of text chunks.
        """

        encoding = self.model.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]

        if len(input_ids) <= self.max_length - self.n_special_tokens:
            return [text]

        chunks = []
        step = (self.max_length - self.n_special_tokens) - stride
        for start in range(0, len(input_ids), step):
            end = min(start + (self.max_length - self.n_special_tokens), len(input_ids))
            start_offset = offsets[start][0]
            end_offset = offsets[end - 1][1]
            chunk_text = text[start_offset:end_offset]
            chunks.append(chunk_text)

        return chunks

    def __call__(self, text, truncation=False, stride=0):
        if not truncation:
            chunks = self._split_into_token_chunks(text, stride=stride)
            if len(chunks) == 1:
                return self._transform_output(self.model(chunks[0])[0])

            # Get average positive, negative, neutral probabilities for all chunks
            chunk_sentiments = self.model(chunks, verbose=True, max_length=self.max_length, truncation=False)
            chunk_sentiments = [self._transform_output(s) for s in chunk_sentiments]
            return {
                "positive": np.mean([s["positive"] for s in chunk_sentiments]),
                "negative": np.mean([s["negative"] for s in chunk_sentiments]),
                "neutral": np.mean([s["neutral"] for s in chunk_sentiments])
            }
        else:
            sentiment = self._transform_output(self.model(text, verbose=True, max_length=self.max_length, truncation=True)[0])
            return sentiment

    def get_sentiment_score(self, sentiment):
        p_pos = sentiment["positive"]# if "very positive" not in sentiment else sentiment["very positive"] + sentiment["positive"]
        p_neg = sentiment["negative"]# if "very negative" not in sentiment else sentiment["very negative"] + sentiment["negative"]
        p_neu = sentiment["neutral"]
        return (p_pos - p_neg) / (p_pos + p_neg + p_neu)