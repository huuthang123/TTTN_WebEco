import fasttext
from pymongo import MongoClient
from datetime import datetime
from collections import Counter
from bson import ObjectId
import re

# ======================
# CONFIG
# ======================
MONGO_URI = "mongodb+srv://ngthang0311:Huuthang123@cluster0.o4wdz.mongodb.net/WebEcommerceTTTN?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "WebEcommerceTTTN"
FASTTEXT_MODEL_PATH = "data/fasttext/en.bin"

# ======================
# KẾT NỐI MONGODB
# ======================
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
reviews_col = db["reviews"]
summaries_col = db["productsummaries"]  # đúng tên collection của schema

# ======================
# LOAD FASTTEXT MODEL
# ======================
print("🔄 Loading FastText model...")
model = fasttext.load_model(FASTTEXT_MODEL_PATH)
print("✅ FastText model loaded.")

# ======================
# HÀM LỌC TỪ
# ======================
def preprocess_text(text):
    """Tách từ và loại ký tự đặc biệt, giữ nguyên lowercase"""
    return re.findall(r'\b[a-z]+\b', text.lower())

# ======================
# HÀM TÓM TẮT REVIEW (loại câu trùng)
# ======================
def summarize_reviews(reviews):
    if not reviews:
        return ""

    # Ghép tất cả review
    text = " ".join(reviews)
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return ""

    # Tính vector cho từng câu
    sentence_vectors = [model.get_sentence_vector(s) for s in sentences]
    avg_vector = sum(sentence_vectors) / len(sentence_vectors)

    def cosine_similarity(v1, v2):
        return sum(a*b for a, b in zip(v1, v2)) / (
            (sum(a*a for a in v1) ** 0.5) * (sum(b*b for b in v2) ** 0.5)
        )

    # Chấm điểm và sắp xếp
    scored_sentences = [
        (s, cosine_similarity(model.get_sentence_vector(s), avg_vector))
        for s in sentences
    ]
    scored_sentences.sort(key=lambda x: x[1], reverse=True)

    # Loại bỏ câu trùng lặp (case-insensitive)
    seen = set()
    summary_sentences = []
    for s, _ in scored_sentences:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            summary_sentences.append(s)
        if len(summary_sentences) >= 3:
            break

    return ". ".join(summary_sentences).strip()

# ======================
# XỬ LÝ TOÀN BỘ SẢN PHẨM
# ======================
product_ids = reviews_col.distinct("productId")
print(f"🔍 Found {len(product_ids)} products to summarize.")

for pid in product_ids:
    try:
        # Lấy tất cả review (comment)
        reviews = [r["comment"] for r in reviews_col.find({"productId": pid}) if r.get("comment")]
        total_reviews = len(reviews)
        if total_reviews == 0:
            continue

        # Sinh tóm tắt
        summary = summarize_reviews(reviews)

        # Tính tần suất từ
        all_words = []
        for review in reviews:
            all_words.extend(preprocess_text(review))
        word_freq = dict(Counter(all_words))

        # Lưu vào DB theo schema ProductSummary
        summaries_col.update_one(
            {"productId": ObjectId(pid)},
            {
                "$set": {
                    "summary": summary,
                    "wordFreq": word_freq,
                    "lastUpdated": datetime.utcnow(),
                    "totalReviews": total_reviews
                }
            },
            upsert=True
        )

        print(f"✅ Summarized product {pid} ({total_reviews} reviews)")

    except Exception as e:
        print(f"❌ Error with product {pid}: {e}")

print("🎯 All summaries created successfully.")
