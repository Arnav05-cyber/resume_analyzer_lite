# download_models.py
import spacy
from sentence_transformers import SentenceTransformer

print("Downloading spaCy model...")
spacy.cli.download("en_core_web_sm")
print("spaCy model downloaded.")

print("Downloading Sentence Transformer model...")
# This will download the model to the default cache directory
SentenceTransformer('paraphrase-MiniLM-L3-v2')
print("Sentence Transformer model downloaded.")