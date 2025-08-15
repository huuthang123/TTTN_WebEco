import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

nltk.download('punkt')

# ===== Config =====
WORD_VECTOR_FILE = 'data/fasttext/en.vec'  # Đường dẫn đến mô hình vector
TEXT_FILE = 'data/sample.txt'              # File chứa review
TOP_N = 3                                   # Số câu tóm tắt

# ===== Cleaning =====
def clean_text(text):
    # Xóa HTML tags, markdown bullets, emoji, ký tự đặc biệt, thừa khoảng trắng
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[-–—•*]+', '', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Loại bỏ emoji & ký tự đặc biệt
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def capitalize_sentences(text):
    # Viết hoa đầu mỗi câu
    return " ".join(s.capitalize() for s in sent_tokenize(text))

# ===== Load word vectors =====
print("🔁 Loading word vectors...")
model = KeyedVectors.load_word2vec_format(WORD_VECTOR_FILE)
print("✅ Word vectors loaded.")

# ===== Load and clean text =====
with open(TEXT_FILE, encoding='utf-8') as f:
    raw_text = f.read()

cleaned_text = capitalize_sentences(clean_text(raw_text))
sentences = [s.strip() for s in sent_tokenize(cleaned_text) if len(s.split()) >= 3]

# ===== Sentence Embedding =====
def sentence_vector(sentence):
    words = word_tokenize(sentence.lower())
    vecs = [model[word] for word in words if word in model]
    if not vecs:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

sentence_vectors = [sentence_vector(s) for s in sentences]

# ===== Check all vectors =====
if all((vec == 0).all() for vec in sentence_vectors):
    print("⚠️ All sentence vectors are empty. Check your input or vector model.")
    exit()

# ===== Similarity & Ranking =====
sim_matrix = cosine_similarity(sentence_vectors)
scores = sim_matrix.sum(axis=1)
top_idx = np.argsort(scores)[-TOP_N:]
top_idx.sort()

# ===== Output Summary =====
print("\n📄 Summary:\n")
for idx in top_idx:
    print(f"- {sentences[idx]}")
