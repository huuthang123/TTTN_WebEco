import redis
import json
import requests
from summarizer import summarize_text  # file summarizer.py bạn vừa tạo

# Cấu hình Redis và Node.js API
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
CHANNEL_NAME = 'reviews'
NODE_ENDPOINT = "http://localhost:3000/api/summaries"

# Kết nối Redis
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
pubsub = r.pubsub()
pubsub.subscribe(CHANNEL_NAME)

print(f"🔔 Đang lắng nghe review mới trên channel: {CHANNEL_NAME}...")

for message in pubsub.listen():
    if message['type'] == 'message':
        try:
            # Parse dữ liệu review từ Redis
            review = json.loads(message['data'])
            review_id = review.get("_id")
            product_id = review.get("productId")
            comment = review.get("comment", "")

            print(f"\n📝 Nhận review mới: {review_id}")
            print(f"Nội dung: {comment}")

            # Tóm tắt bằng fastText
            summary = summarize_text(comment, top_n=3)
            print(f"📄 Tóm tắt: {summary}")

            # Gửi tóm tắt về Node.js
            payload = {
                "reviewId": review_id,
                "productId": product_id,
                "summary": summary
            }
            res = requests.post(NODE_ENDPOINT, json=payload)
            if res.status_code == 200:
                print(f"✅ Đã gửi tóm tắt cho review {review_id} về Node.js")
            else:
                print(f"⚠️ Lỗi gửi summary ({res.status_code}): {res.text}")

        except Exception as e:
            print(f"❌ Lỗi xử lý review: {e}")
