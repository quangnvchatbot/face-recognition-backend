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
    print("✅ face_recognition imported successfully")
except ImportError as e:
    FACE_RECOGNITION_AVAILABLE = False
    print(f"❌ face_recognition import failed: {e}")

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
    storage_uri="memory://",
)

# Firebase config - SỬ DỤNG URL CỦA BẠN
FIREBASE_DB_URL = os.getenv('FIREBASE_DB_URL', 'https://serious-hold-468214-u2-default-rtdb.asia-southeast1.firebasedatabase.app')
FIREBASE_API_KEY = os.getenv('FIREBASE_API_KEY', '')

# Cache để lưu embeddings
persons_cache = {}
persons_embeddings_cache = {}
cache_timestamp = 0
CACHE_DURATION = 300  # 5 minutes

# ============ CORE FUNCTIONS ============

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        return image.convert('RGB')
    except Exception as error:
        print(f"❌ Error converting base64 to image: {error}")
        return None

def get_face_embedding(image):
    """Extract face embedding from image"""
    if not FACE_RECOGNITION_AVAILABLE:
        print("❌ face_recognition not available")
        return None
    
    try:
        # Convert PIL Image to numpy array
        image_array = np.array(image)
        
        # Detect faces - sử dụng hog để tiết kiệm bộ nhớ
        face_locations = face_recognition.face_locations(image_array, model="hog")
        
        if len(face_locations) == 0:
            print("⚠️ No face detected in image")
            return None
        
        print(f"✅ Found {len(face_locations)} face(s)")
        
        # Get encoding of first face
        face_encodings = face_recognition.face_encodings(image_array, face_locations)
        
        if len(face_encodings) == 0:
            print("⚠️ Could not encode face")
            return None
        
        return face_encodings[0]
    
    except Exception as error:
        print(f"❌ Error getting face embedding: {error}")
        return None

def load_persons_with_embeddings():
    """Load customers từ Firebase và decode embeddings - ĐÃ SỬA ĐỂ ĐỌC ARRAY"""
    global persons_cache, persons_embeddings_cache, cache_timestamp
    
    current_time = time.time()
    # Nếu cache còn hợp lệ, dùng cache
    if cache_timestamp and (current_time - cache_timestamp) < CACHE_DURATION:
        return persons_cache, persons_embeddings_cache
    
    try:
        # ✅ SỬA PATH: Đọc từ 'customers' 
        customers_url = f"{FIREBASE_DB_URL}/customers.json"
        response = requests.get(customers_url, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Failed to fetch customers: {response.status_code}")
            return {}, {}
        
        customers_data = response.json()
        
        # ✅ XỬ LÝ ARRAY: customers là array, không phải object
        persons_cache = {}
        embeddings = {}
        
        if isinstance(customers_data, list):
            for idx, customer in enumerate(customers_data):
                if customer and isinstance(customer, dict):
                    customer_id = customer.get('id', f'customer_{idx}')
                    persons_cache[customer_id] = customer
                    
                    # Decode embedding nếu có
                    if 'faceEmbedding' in customer:
                        try:
                            embedding_data = customer['faceEmbedding']
                            if isinstance(embedding_data, list):
                                embedding = np.array(embedding_data)
                                embeddings[customer_id] = embedding
                                print(f"✅ Loaded embedding for {customer.get('name', customer_id)}")
                        except Exception as e:
                            print(f"⚠️ Failed to decode embedding for {customer_id}: {e}")
        
        cache_timestamp = current_time
        print(f"✅ Loaded {len(persons_cache)} customers, {len(embeddings)} with embeddings")
        return persons_cache, embeddings
    
    except Exception as error:
        print(f"❌ Error loading customers: {error}")
        return {}, {}

def find_matching_person(face_embedding):
    """Find matching person from database"""
    if face_embedding is None:
        return None
    
    persons, embeddings = load_persons_with_embeddings()
    
    if not embeddings:
        print("⚠️ No embeddings in database")
        return None
    
    best_match_id = None
    best_match_distance = 0.6  # Threshold for face_recognition
    
    # Compare with all known faces
    for person_id, known_embedding in embeddings.items():
        try:
            # Đảm bảo known_embedding là numpy array
            if isinstance(known_embedding, list):
                known_embedding = np.array(known_embedding)
                
            distance = face_recognition.face_distance([known_embedding], face_embedding)[0]
            
            # Lower distance = better match
            if distance < best_match_distance:
                best_match_distance = distance
                best_match_id = person_id
                print(f"🎯 Better match: {person_id} with distance {distance:.4f}")
        except Exception as error:
            print(f"⚠️ Error comparing with {person_id}: {error}")
    
    if best_match_id:
        person_data = persons.get(best_match_id, {})
        confidence = 1 - best_match_distance  # Convert distance to confidence
        return {
            'person_id': best_match_id,
            'person_name': person_data.get('name', 'Unknown'),
            'position': person_data.get('position', 'N/A'),
            'department': person_data.get('roomId', 'N/A'),  # Sử dụng roomId từ customer
            'confidence': round(confidence * 100, 2)
        }
    
    print(f"🔍 No match found (best distance: {best_match_distance:.4f})")
    return None

# ============ API ENDPOINTS ============

@app.route('/')
def home():
    """Root endpoint for health check"""
    return jsonify({
        "status": "running",
        "service": "Face Recognition API",
        "version": "1.0",
        "face_recognition_available": FACE_RECOGNITION_AVAILABLE,
        "endpoints": {
            "health": "/health",
            "recognize": "/api/recognize",
            "train": "/api/train-customer",
            "rate_limit_info": "/api/rate-limit-info"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "face_recognition": FACE_RECOGNITION_AVAILABLE,
        "timestamp": time.time(),
        "environment": os.getenv('RAILWAY_ENVIRONMENT', 'development')
    }), 200

@app.route('/api/recognize', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")
def recognize():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON data received"
            }), 400
            
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
        
        if not FACE_RECOGNITION_AVAILABLE:
            return jsonify({
                "status": "error",
                "message": "Face recognition not available on server"
            }), 500
        
        print(f"🔍 Processing recognition for event {event_id}")
        
        # Convert base64 to image
        image = base64_to_image(image_base64)
        if image is None:
            return jsonify({
                "status": "error",
                "message": "Invalid image format"
            }), 400
        
        # Get face embedding
        face_embedding = get_face_embedding(image)
        if face_embedding is None:
            return jsonify({
                "status": "no_face",
                "message": "No face detected in image"
            }), 200
        
        # Find matching person
        matched_person = find_matching_person(face_embedding)
        if matched_person is None:
            return jsonify({
                "status": "unknown",
                "message": "Face not recognized"
            }), 200
        
        person_id = matched_person['person_id']
        
        # ✅ CẬP NHẬT: Lưu checkin vào customers array
        try:
            # Lấy toàn bộ customers để tìm index
            customers_url = f"{FIREBASE_DB_URL}/customers.json"
            customers_response = requests.get(customers_url, timeout=5)
            customers = customers_response.json() if customers_response.status_code == 200 else []
            
            customer_index = -1
            if isinstance(customers, list):
                for idx, customer in enumerate(customers):
                    if customer and customer.get('id') == person_id:
                        customer_index = idx
                        break
            
            if customer_index >= 0:
                # Update customer checkedIn status
                checkin_data = {
                    "checkedIn": True,
                    "checkedInTime": int(time.time() * 1000),
                    "checkinEvent": event_id
                }
                
                update_url = f"{FIREBASE_DB_URL}/customers/{customer_index}.json"
                update_response = requests.patch(update_url, json=checkin_data)
                
                if update_response.status_code == 200:
                    print(f"✅ Updated checkin status for {person_id}")
                else:
                    print(f"⚠️ Failed to update checkin status for {person_id}: {update_response.status_code}")
        
        except Exception as update_error:
            print(f"⚠️ Error updating checkin status: {update_error}")
            # Không throw error vì recognition vẫn thành công
        
        return jsonify({
            "status": "success",
            "person_id": person_id,
            "person_name": matched_person['person_name'],
            "position": matched_person['position'],
            "department": matched_person['department'],
            "confidence": matched_person['confidence'],
            "message": "Checkin successful"
        }), 200
        
    except Exception as error:
        print(f"❌ Error in recognize: {error}")
        return jsonify({
            "status": "error",
            "message": f"Internal server error: {str(error)}"
        }), 500

# ✅ THÊM ENDPOINT MỚI: Train face embedding cho customer
@app.route('/api/train-customer', methods=['POST'])
@limiter.limit("5 per minute")
def train_customer():
    try:
        data = request.get_json()
        customer_id = data.get('customerId')
        image_url = data.get('imageUrl')
        
        if not customer_id or not image_url:
            return jsonify({"error": "Missing customerId or imageUrl"}), 400
        
        print(f"🔧 Training face embedding for customer: {customer_id}")
        
        # Download image từ Firebase Storage
        response = requests.get(image_url, timeout=15)
        if response.status_code != 200:
            return jsonify({"error": f"Failed to download image: {response.status_code}"}), 400
        
        # Convert to image và extract embedding
        image_data = response.content
        image = Image.open(BytesIO(image_data))
        embedding = get_face_embedding(image)
        
        if embedding is None:
            return jsonify({"error": "No face detected in customer image"}), 400
        
        # ✅ Lưu embedding lên Firebase
        embedding_list = embedding.tolist()
        
        # Lấy toàn bộ customers để tìm index
        customers_url = f"{FIREBASE_DB_URL}/customers.json"
        customers_response = requests.get(customers_url, timeout=5)
        customers = customers_response.json() if customers_response.status_code == 200 else []
        
        if isinstance(customers, list):
            # Tìm customer index
            customer_index = -1
            for idx, customer in enumerate(customers):
                if customer and customer.get('id') == customer_id:
                    customer_index = idx
                    break
            
            if customer_index >= 0:
                # Update customer với embedding
                update_url = f"{FIREBASE_DB_URL}/customers/{customer_index}.json"
                update_response = requests.patch(update_url, json={'faceEmbedding': embedding_list})
                
                if update_response.status_code == 200:
                    # Clear cache để load lại embeddings mới
                    global cache_timestamp
                    cache_timestamp = 0
                    
                    return jsonify({
                        "status": "success", 
                        "customerId": customer_id,
                        "embedding_length": len(embedding_list),
                        "message": "Face embedding trained and saved successfully"
                    })
                else:
                    return jsonify({"error": f"Failed to save embedding to Firebase: {update_response.status_code}"}), 500
        
        return jsonify({"error": "Customer not found in database"}), 404
        
    except Exception as error:
        print(f"❌ Error in train-customer: {error}")
        return jsonify({"error": str(error)}), 500

@app.route('/api/batch/recognize', methods=['POST'])
@limiter.limit("5 per minute")
def batch_recognize():
    """Batch recognition endpoint - placeholder"""
    return jsonify({
        "status": "not_implemented", 
        "message": "Batch recognition will be available soon"
    }), 501

@app.route('/api/rate-limit-info', methods=['GET'])
def rate_limit_info():
    """Trả về thông tin về rate limiting hiện tại"""
    return jsonify({
        "status": "success",
        "rate_limits": {
            "/api/recognize": "10 requests per minute per event",
            "/api/train-customer": "5 requests per minute",
            "/api/batch/recognize": "5 requests per minute",
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
    port = int(os.getenv('PORT', 8080))  # ✅ SỬA PORT: 5000 → 8080
    print(f"🚀 Starting Face Recognition Server on port {port}")
    print(f"🔧 Face recognition available: {FACE_RECOGNITION_AVAILABLE}")
    print(f"📊 Firebase URL: {FIREBASE_DB_URL}")
    print(f"🌐 Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'development')}")
    app.run(host='0.0.0.0', port=port, debug=False)
