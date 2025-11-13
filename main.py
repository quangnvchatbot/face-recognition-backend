import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import db, credentials
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Firebase
try:
    # Lấy credentials từ environment variable
    firebase_cred = json.loads(os.getenv('FIREBASE_CREDENTIALS', '{}'))
    cred = credentials.Certificate(firebase_cred)
    
    firebase_admin.initialize_app(cred, {
        'databaseURL': os.getenv('FIREBASE_DB_URL', 'https://serious-hold-468214-u2-default-rtdb.asia-southeast1.firebasedatabase.app')
    })
except Exception as e:
    print(f"Firebase init error: {e}")

database = db.reference()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"}), 200

@app.route('/api/recognize', methods=['POST', 'OPTIONS'])
def recognize():
    """
    Recognize face and create checkin
    
    Request JSON:
    {
        "eventId": "event_001"
    }
    
    Response:
    {
        "status": "success|duplicate|error",
        "person_id": "person_001",
        "person_name": "Nguyễn Văn Quảng",
        "position": "Chuyên viên",
        "department": "PHÒNG CÔNG NGHỆ SỐ",
        "confidence": 0.95,
        "message": "Checkin successful"
    }
    """
    
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        event_id = data.get('eventId')
        
        if not event_id:
            return jsonify({
                "status": "error",
                "message": "Missing eventId"
            }), 400
        
        # 1. Lấy danh sách persons từ Firebase
        persons_ref = database.child('persons')
        persons = persons_ref.get().val()
        
        if not persons:
            return jsonify({
                "status": "error",
                "message": "No persons in database"
            }), 404
        
        # 2. Chọn random person (tạm thời - sau thay bằng face recognition)
        import random
        person_ids = list(persons.keys())
        random_person_id = random.choice(person_ids)
        matched_person = persons[random_person_id]
        
        # 3. Check duplicate checkin
        checkins_ref = database.child('events').child(event_id).child('checkins')
        checkins = checkins_ref.get().val() or {}
        
        for checkin_id, checkin in checkins.items():
            if checkin.get('person_id') == random_person_id:
                return jsonify({
                    "status": "duplicate",
                    "person_id": random_person_id,
                    "person_name": matched_person.get('name'),
                    "message": "Already checked in"
                }), 409
        
        # 4. Create checkin record
        checkin_id = f"checkin_{int(__import__('time').time() * 1000)}"
        checkin_data = {
            "person_id": random_person_id,
            "person_name": matched_person.get('name'),
            "position": matched_person.get('position'),
            "department": matched_person.get('department'),
            "timestamp": int(__import__('time').time() * 1000),
            "image_url": matched_person.get('image_url')
        }
        
        checkins_ref.child(checkin_id).set(checkin_data)
        
        # 5. Return success
        return jsonify({
            "status": "success",
            "person_id": random_person_id,
            "person_name": matched_person.get('name'),
            "position": matched_person.get('position'),
            "department": matched_person.get('department'),
            "confidence": 0.95,
            "message": "Checkin successful"
        }), 200
        
    except Exception as error:
        print(f"Error: {error}")
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500

@app.route('/api/events/<event_id>/checkins', methods=['GET'])
def get_checkins(event_id):
    """Get all checkins for an event"""
    try:
        checkins_ref = database.child('events').child(event_id).child('checkins')
        checkins = checkins_ref.get().val() or {}
        
        return jsonify({
            "status": "success",
            "event_id": event_id,
            "checkins": checkins,
            "count": len(checkins)
        }), 200
    except Exception as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
