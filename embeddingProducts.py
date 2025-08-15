from pymongo import MongoClient
import fasttext
import numpy as np

# Kết nối DB
client = MongoClient("mongodb+srv://ngthang0311:Huuthang123@cluster0.o4wdz.mongodb.net/WebEcommerceTTTN?retryWrites=true&w=majority&appName=Cluster0")
db = client['WebEcommerceTTTN']
products = db['products']

# Load model
model = fasttext.load_model("data/fasttext/en.bin")

def get_embedding(text):
    tokens = text.lower().split()
    vecs = [model.get_word_vector(t) for t in tokens if t.isalpha()]
    if not vecs:
        return []
    return np.mean(vecs, axis=0).tolist()

# Cập nhật embedding cho từng sản phẩm
for p in products.find():
    name = p.get("name", "")
    if not name:
        continue
    embedding = get_embedding(name)
    products.update_one({"_id": p["_id"]}, {"$set": {"embedding": embedding}})
    print(f"Updated embedding for product {p['_id']}")