import redis
import json
import pymongo
import fasttext
import numpy as np
import re
from stopwordsiso import stopwords
from datetime import datetime

# ===== CONFIG =====
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
MONGO_URI = "mongodb+srv://ngthang0311:Huuthang123@cluster0.o4wdz.mongodb.net/WebEcommerceTTTN?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "WebEcommerceTTTN"
FASTTEXT_MODEL_PATH = "data/fasttext/en.bin"

# ===== CONNECT =====
client_mongo = pymongo.MongoClient(MONGO_URI)
db = client_mongo[DB_NAME]
reviews_col = db["reviews"]
summaries_col = db["productsummaries"]

print("Loading FastText model...")
model = fasttext.load_model(FASTTEXT_MODEL_PATH)
print("FastText model loaded.")

# ===== Preprocess =====
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()
    sw = stopwords("en")
    return [w for w in words if w not in sw]

# ===== Generate summary from sentences =====
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def generate_summary(sentences, model, top_n=3):
    sentence_vectors = []
    for sent in sentences:
        tokens = preprocess(sent)
        if not tokens:
            continue
        vecs = [model.get_word_vector(w) for w in tokens]
        sent_vec = np.mean(vecs, axis=0)
        sentence_vectors.append((sent, sent_vec))

    if not sentence_vectors:
        return ""

    doc_vector = np.mean([vec for _, vec in sentence_vectors], axis=0)

    ranked = sorted(sentence_vectors, key=lambda x: cosine_sim(doc_vector, x[1]), reverse=True)
    top_sentences = [sent for sent, _ in ranked[:top_n]]
    return " ".join(top_sentences)

# ===== Update summary with new review =====
def update_summary_with_new_review(product_id, new_review_text):
    summary_doc = summaries_col.find_one({"productId": product_id})

    old_summary = summary_doc.get("summary", "") if summary_doc else ""
    old_word_freq = summary_doc.get("wordFreq", {}) if summary_doc else {}
    total_reviews = summary_doc.get("totalReviews", 0) if summary_doc else 0

    # Update wordFreq dần dần
    new_words = preprocess(new_review_text)
    for w in new_words:
        old_word_freq[w] = old_word_freq.get(w, 0) + 1

    # Combine old summary + new review to generate new summary
    combined_text = old_summary + " " + new_review_text
    combined_sentences = re.split(r"(?<=[.!?]) +", combined_text)

    new_summary = generate_summary(combined_sentences, model)

    # Update DB
    summaries_col.update_one(
        {"productId": product_id},
        {
            "$set": {
                "summary": new_summary,
                "wordFreq": old_word_freq,
                "totalReviews": total_reviews + 1,
                "lastUpdated": datetime.utcnow()
            }
        },
        upsert=True
    )
    print(f"Updated summary for product {product_id}")

# ===== Redis subscriber =====
r = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
pubsub = r.pubsub()
pubsub.subscribe("reviews")

print("Listening for new reviews on Redis channel 'reviews'...")

for message in pubsub.listen():
    if message["type"] == "message":
        try:
            review_data = json.loads(message["data"])
            product_id = review_data["productId"]
            comment = review_data.get("comment", "")

            if product_id and comment:
                update_summary_with_new_review(product_id, comment)
        except Exception as e:
            print("Error processing review:", e)
