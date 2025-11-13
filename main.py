import os
import json
import time
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import db, credentials
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Firebase
firebase_cred_str = os.getenv('FIREBASE_CREDENTIALS')
firebase_db_url = os.getenv('FIREBASE_DB_URL', 'https://serious-hold-468214-u2-default-rtdb.asia-southeast1.firebasedatabase.app')

if not firebase_cred_str:
    print("ERROR: FIREBASE_CREDENTIALS not found")
else:
    try:
        firebase_cred = json.loads(firebase_cred_str)
        cred = credentials.Certificate(firebase_cred)
        firebase_admin.initialize_app(cred, {
            'databaseURL': firebase_db_url
        })
        print("Firebase initialized")
    except Exception as e:
        print(f"Firebase error: {e}")

database = db.reference()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/api/recognize', methods=['POST', 'OPTIONS'])
def recognize():
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
        
        # Get persons from Firebase
        persons_ref = database.child('persons')
        persons_snapshot = persons_ref.get()
        persons = persons_snapshot.val() if persons_snapshot.val() else {}
        
        if not persons:
            return jsonify({
                "status": "error",
                "message": "No persons in database"
            }), 404
        
        # Random person (temporary)
        person_ids = list(persons.keys())
        random_person_id = random.choice(person_ids)
        matched_person = persons[random_person_id]
        
        # Check duplicate
        checkins_ref = database.child('events').child(event_id).child('checkins')
        checkins_snapshot = checkins_ref.get()
        checkins = checkins_snapshot.val() if checkins_snapshot.val() else {}
        
        for checkin_id, checkin in checkins.items():
            if checkin.get('person_id') == random_person_id:
                return jsonify({
                    "status": "duplicate",
                    "person_id": random_person_id,
                    "person_name": matched_person.get('name'),
                    "message": "Already checked in"
                }), 409
        
        # Create checkin
        checkin_id = f"checkin_{int(time.time() * 1000)}"
        checkin_data = {
            "person_id": random_person_id,
            "person_name": matched_person.get('name'),
            "position": matched_person.get('position'),
            "department": matched_person.get('department'),
            "timestamp": int(time.time() * 1000),
            "image_url": matched_person.get('image_url')
        }
        
        checkins_ref.child(checkin_id).set(checkin_data)
        
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
    try:
        checkins_ref = database.child('events').child(event_id).child('checkins')
        checkins_snapshot = checkins_ref.get()
        checkins = checkins_snapshot.val() if checkins_snapshot.val() else {}
        
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
