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

# Import face_recognition với error handling tốt hơn
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ face_recognition imported successfully")
except ImportError as e:
    FACE_RECOGNITION_AVAILABLE = False
    print(f"❌ face_recognition import failed: {e}")

app = Flask(__name__)
CORS(app)

# Firebase config từ environment
FIREBASE_DB_URL = os.getenv('FIREBASE_DB_URL', 'https://serious-hold-468214-u2-default-rtdb.asia-southeast1.firebasedatabase.app')
FIREBASE_API_KEY = os.getenv('FIREBASE_API_KEY', '')

# Cache để lưu embeddings của persons (tránh fetch mỗi lần)
persons_cache = {}
persons_embeddings_cache = {}
cache_timestamp = 0
CACHE_DURATION = 300  # Cache 5 minutes

def load_persons_with_embeddings():
    """Load persons từ Firebase và decode embeddings"""
    global persons_cache, persons_embeddings_cache, cache_timestamp
    
    current_time = time.time()
    # Nếu cache còn hợp lệ, dùng cache
    if cache_timestamp and (current_time - cache_timestamp) < CACHE_DURATION:
        return persons_cache, persons_embeddings_cache
    
    try:
        persons_url = f"{FIREBASE_DB_URL}/persons.json"
        response = requests.get(persons_url, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Failed to fetch persons: {response.status_code}")
            return {}, {}
        
        persons = response.json() or {}
        embeddings = {}
        
        # Process each person
        for person_id, person_data in persons.items():
            if isinstance(person_data, dict):
                persons_cache[person_id] = person_data
                
                # Decode embedding nếu có
                if 'face_embedding' in person_data:
                    try:
                        embedding_data = person_data['face_embedding']
                        # Nếu embedding là string, parse JSON
                        if isinstance(embedding_data, str):
                            embedding = np.array(json.loads(embedding_data))
                        else:  # Nếu đã là list
                            embedding = np.array(embedding_data)
                        persons_embeddings_cache[person_id] = embedding
                        print(f"✅ Loaded embedding for {person_id}")
                    except Exception as e:
                        print(f"⚠️ Failed to decode embedding for {person_id}: {e}")
        
        cache_timestamp = current_time
        print(f"✅ Loaded {len(persons_cache)} persons, {len(persons_embeddings_cache)} with embeddings")
        return persons_cache, persons_embeddings_cache
    
    except Exception as error:
        print(f"❌ Error loading persons: {error}")
        return {}, {}

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        return image.convert('RGB')  # Đảm bảo là RGB
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
        
        # Detect faces - sử dụng model nhỏ hơn để tiết kiệm bộ nhớ
        face_locations = face_recognition.face_locations(image_array, model="hog")  # hoặc "cnn"
        
        if len(face_locations) == 0:
            print("⚠️ No face detected in image")
            return None
        
        print(f"✅ Found {len(face_locations)} face(s)")
        
        # Get encoding of first face (usually the largest)
        face_encodings = face_recognition.face_encodings(image_array, face_locations)
        
        if len(face_encodings) == 0:
            print("⚠️ Could not encode face")
            return None
        
        return face_encodings[0]  # Return first face encoding
    
    except Exception as error:
        print(f"❌ Error getting face embedding: {error}")
        return None

def find_matching_person(face_embedding):
    """Find matching person from database"""
    if face_embedding is None:  # ✅ SỬA LỖI: điều kiện ngược
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
            'department': person_data.get('department', 'N/A'),
            'confidence': round(confidence * 100, 2)  # Convert to percentage
        }
    
    print(f"🔍 No match found (best distance: {best_match_distance:.4f})")
    return None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "face_recognition": FACE_RECOGNITION_AVAILABLE,
        "timestamp": time.time()
    }), 200

@app.route('/api/recognize', methods=['POST', 'OPTIONS'])
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
        image_base64 = data.get('image')  # Base64 image from frontend
        
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
            }), 200  # ✅ Thay đổi thành 200 để frontend xử lý
        
        # Find matching person
        matched_person = find_matching_person(face_embedding)
        if matched_person is None:
            return jsonify({
                "status": "unknown",
                "message": "Face not recognized"
            }), 200  # ✅ Thay đổi thành 200 để frontend xử lý
        
        person_id = matched_person['person_id']
        
        # Check if already checked in
        checkins_url = f"{FIREBASE_DB_URL}/events/{event_id}/checkins.json"
        checkins_response = requests.get(checkins_url, timeout=5)
        checkins = checkins_response.json() if checkins_response.status_code == 200 else {}
        
        already_checked_in = False
        if isinstance(checkins, dict):
            for checkin_id, checkin in checkins.items():
                if isinstance(checkin, dict) and checkin.get('person_id') == person_id:
                    already_checked_in = True
                    break
        
        if already_checked_in:
            return jsonify({
                "status": "duplicate",
                "person_id": person_id,
                "person_name": matched_person['person_name'],
                "position": matched_person['position'],
                "department": matched_person['department'],
                "confidence": matched_person['confidence'],
                "message": "Already checked in"
            }), 200
        
        # Create new checkin
        checkin_id = f"checkin_{int(time.time() * 1000)}"
        checkin_data = {
            "person_id": person_id,
            "person_name": matched_person['person_name'],
            "position": matched_person['position'],
            "department": matched_person['department'],
            "timestamp": int(time.time() * 1000),
            "confidence": matched_person['confidence']
        }
        
        # Save to Firebase
        post_url = f"{FIREBASE_DB_URL}/events/{event_id}/checkins/{checkin_id}.json"
        post_response = requests.put(post_url, json=checkin_data, timeout=5)
        
        if post_response.status_code not in [200, 201]:
            return jsonify({
                "status": "error",
                "message": "Failed to save checkin to database"
            }), 500
        
        print(f"✅ Checkin successful for {matched_person['person_name']}")
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

@app.route('/api/events/<event_id>/checkins', methods=['GET'])
def get_checkins(event_id):
    try:
        checkins_url = f"{FIREBASE_DB_URL}/events/{event_id}/checkins.json"
        response = requests.get(checkins_url, timeout=5)
        
        checkins_dict = response.json() if response.status_code == 200 else {}
        
        # Convert dict to list for frontend
        checkins_list = []
        if isinstance(checkins_dict, dict):
            for checkin_id, checkin_data in checkins_dict.items():
                if isinstance(checkin_data, dict):
                    checkin_item = {
                        "checkin_id": checkin_id,
                        **checkin_data
                    }
                    checkins_list.append(checkin_item)
        
        return jsonify({
            "status": "success",
            "event_id": event_id,
            "checkins": checkins_list,
            "count": len(checkins_list)
        }), 200
    except Exception as error:
        print(f"❌ Error in get_checkins: {error}")
        return jsonify({
            "status": "error",
            "message": str(error),
            "checkins": [],
            "count": 0
        }), 500

@app.route('/api/persons/train', methods=['POST', 'OPTIONS'])
def train_persons():
    """Train face embeddings for all persons in database"""
    if request.method == 'OPTIONS':
        return '', 204
    
    if not FACE_RECOGNITION_AVAILABLE:
        return jsonify({
            "status": "error",
            "message": "Face recognition not available"
        }), 500
    
    try:
        persons_url = f"{FIREBASE_DB_URL}/persons.json"
        response = requests.get(persons_url, timeout=10)
        
        if response.status_code != 200:
            return jsonify({
                "status": "error",
                "message": "Failed to fetch persons"
            }), 500
        
        persons = response.json() or {}
        trained_count = 0
        
        for person_id, person_data in persons.items():
            if isinstance(person_data, dict) and 'image_url' in person_data:
                try:
                    # Download image
                    img_response = requests.get(person_data['image_url'], timeout=10)
                    if img_response.status_code == 200:
                        image = Image.open(BytesIO(img_response.content)).convert('RGB')
                        
                        # Get embedding
                        embedding = get_face_embedding(image)
                        if embedding is not None:
                            # Save embedding to Firebase
                            embedding_list = embedding.tolist()
                            update_url = f"{FIREBASE_DB_URL}/persons/{person_id}/face_embedding.json"
                            update_response = requests.put(update_url, json=embedding_list, timeout=5)
                            
                            if update_response.status_code in [200, 201]:
                                trained_count += 1
                                print(f"✅ Trained {person_data.get('name', person_id)}")
                            else:
                                print(f"❌ Failed to save embedding for {person_id}")
                    else:
                        print(f"⚠️ Failed to download image for {person_id}")
                except Exception as e:
                    print(f"⚠️ Failed to train {person_id}: {e}")
        
        # Clear cache
        global cache_timestamp
        cache_timestamp = 0
        
        return jsonify({
            "status": "success",
            "trained_count": trained_count,
            "message": f"Trained {trained_count} persons"
        }), 200
    
    except Exception as error:
        print(f"❌ Error in train_persons: {error}")
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Clear embeddings cache"""
    global persons_cache, persons_embeddings_cache, cache_timestamp
    persons_cache = {}
    persons_embeddings_cache = {}
    cache_timestamp = 0
    return jsonify({"status": "success", "message": "Cache cleared"})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"🚀 Starting server on port {port}")
    print(f"🔧 Face recognition available: {FACE_RECOGNITION_AVAILABLE}")
    app.run(host='0.0.0.0', port=port, debug=False)
