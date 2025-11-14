import os
import json
import time
import base64
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Import face_recognition
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠️ face_recognition not installed. Install with: pip install face_recognition")

app = Flask(__name__)
CORS(app)

# 🔒 RATE LIMITING CONFIGURATION
def get_identifier():
    """Custom identifier for rate limiting - dùng eventId nếu có"""
    event_id = request.get_json().get('eventId') if request.is_json else None
    if event_id:
        return f"event_{event_id}"
    return get_remote_address()

# Khởi tạo Limiter
limiter = Limiter(
    app=app,
    key_func=get_identifier,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",  # Sử dụng in-memory storage (có thể thay bằng Redis sau)
)

# Firebase config
FIREBASE_DB_URL = os.getenv('FIREBASE_DB_URL', 'https://serious-hold-468214-u2-default-rtdb.asia-southeast1.firebasedatabase.app')
FIREBASE_API_KEY = os.getenv('FIREBASE_API_KEY', '')

# Cache để lưu embeddings
persons_cache = {}
persons_embeddings_cache = {}
cache_timestamp = 0
CACHE_DURATION = 300

# 🔒 RATE LIMIT FOR SPECIFIC ENDPOINTS
@app.route('/api/recognize', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")  # 10 requests mỗi phút cho mỗi event
def recognize():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        event_id = data.get('eventId')
        image_base64 = data.get('image')
        
        if not event_id:
            return jsonify({
                "status": "error",
                "message": "Missing eventId"
            }), 400
        
        if not image_base64:
            return jsonify({
                "status": "error",
                "message": "Missing image data"
            }), 400
        
        # ... (phần còn lại của code recognize)
        
    except Exception as error:
        print(f"❌ Error in recognize: {error}")
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500

@app.route('/api/batch/recognize', methods=['POST'])
@limiter.limit("5 per minute")  # Giới hạn thấp hơn cho batch
def batch_recognize():
    # ... implementation
    pass

@app.route('/api/persons/train', methods=['POST'])
@limiter.limit("2 per hour")  # Train không nên chạy thường xuyên
def train_persons():
    # ... implementation
    pass

# 🔍 ENDPOINT ĐỂ KIỂM TRA RATE LIMITING
@app.route('/api/rate-limit-info', methods=['GET'])
def rate_limit_info():
    """Trả về thông tin về rate limiting hiện tại"""
    return jsonify({
        "status": "success",
        "rate_limits": {
            "/api/recognize": "10 requests per minute per event",
            "/api/batch/recognize": "5 requests per minute",
            "/api/persons/train": "2 requests per hour",
            "default": "200 per day, 50 per hour"
        }
    })

# Xử lý lỗi Rate Limit
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "status": "error",
        "message": f"Rate limit exceeded: {e.description}",
        "retry_after": f"Please wait {e.retry_after} seconds"
    }), 429

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"🚀 Starting server with Rate Limiting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
