from flask import Flask, request, jsonify
import fasttext
import numpy as np

app = Flask(__name__)

model = fasttext.load_model("data/fasttext/en.bin")

def get_embedding(text):
    tokens = text.lower().split()
    vecs = [model.get_word_vector(t) for t in tokens if t.isalpha()]
    if not vecs:
        return []
    return np.mean(vecs, axis=0).tolist()

@app.route('/embedding', methods=['POST'])
def embedding():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    vector = get_embedding(text)
    return jsonify({"embedding": vector})

if __name__ == '__main__':
    app.run(port=8000)
