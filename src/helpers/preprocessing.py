import re
import unicodedata
import pandas as pd
import spacy; spacy.prefer_gpu()
from tqdm import tqdm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.helpers import spacy as spacy_helpers


try:
    entity_detection = spacy.load("en_core_web_sm")
except:
    spacy_helpers.install_spacy_model(lang='en', pipeline_for="sm")
    entity_detection = spacy.load("en_core_web_sm")

def clean_text(text: str, entities_to_mask: list) -> str:

    # 0. Remove newlines and special characters
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\\', ' ')

    # 1. Normalize Unicode
    text = unicodedata.normalize("NFKC", text)
    
    # 2. Remove non-informative patterns
    text = re.sub(r'\s+', ' ', text)        # normalize whitespace
    text = re.sub(r'[•◦▪●]', '', text)      # remove bullet symbols
    text = re.sub(r'[_*~^]+', '', text)     # remove markdown / symbol clutter

    # 3. Mask entities (dates, times, locations, names and organizations)
    entities = entity_detection(text).ents
    for ent in entities:
        if ent.label_ in entities_to_mask:
            text = text.replace(ent.text, ent.label_)

    # 4. Mask percentages and numbers
    text = re.sub(r'\b\d+(\.\d+)?%?\b', 'NUM', text)

    # 5. Strip leading/trailing whitespace
    text = text.strip()

    return text


def get_clean_text(row, entities_to_mask=["DATE", "TIME", "PERSON", "ORGANIZATION", "NUM", "LOCATION"], max_lenght=20000) -> str:
    text = row['text']
    text = clean_text(text, entities_to_mask=entities_to_mask)
    words = text.split()
    if len(words) > max_lenght:
        text = ' '.join(words[:max_lenght])
    return text

def get_clean_texts(df: pd.DataFrame, entities_to_mask=["DATE", "TIME", "PERSON", "ORGANIZATION", "NUM", "LOCATION"], max_lenght=20000, verbose=False) -> list:
    documents = []
    if verbose:
        for i, row in tqdm(df.iterrows(), total=len(df)):
            document = get_clean_text(row, entities_to_mask=entities_to_mask, max_lenght=max_lenght)
            documents.append(document)
    else:
        return df.apply(get_clean_text, axis=1, max_lenght=max_lenght).values.tolist()
    
    return documents