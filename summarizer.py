import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

nltk.download('punkt')

WORD_VECTOR_FILE = 'data/fasttext/en.vec'
MIN_WORDS = 5

print("🔁 Loading word vectors...")
model = KeyedVectors.load_word2vec_format(WORD_VECTOR_FILE)
print("✅ Word vectors loaded.")

def clean_text(text):
    text = re.sub(r'[^\w\s.,!?\'\"]+', '', text)
    text = re.sub(r'\b\d+\.', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_valid_sentence(s):
    s = s.strip()
    words = s.split()
    return len(words) >= MIN_WORDS and s[0].isalpha() and s[-1] in ".!?"

def capitalize_sentence(s):
    return s[0].upper() + s[1:] if s else s

def sentence_vector(sentence):
    words = word_tokenize(sentence.lower())
    word_vecs = [model[word] for word in words if word in model]
    return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(model.vector_size)

def summarize_text(text, top_n=3):
    cleaned_text = clean_text(text)
    sentences = sent_tokenize(cleaned_text)
    sentences = [capitalize_sentence(s) for s in sentences if is_valid_sentence(s)]
    if not sentences:
        return text.strip()

    sentence_vectors = [sentence_vector(sent) for sent in sentences]
    sim_matrix = cosine_similarity(sentence_vectors)
    sentence_scores = sim_matrix.sum(axis=1)

    top_indices = np.argsort(sentence_scores)[-top_n:]
    top_indices.sort()

    top_sentences = [sentences[idx] for idx in top_indices]
    return ' '.join(top_sentences)
