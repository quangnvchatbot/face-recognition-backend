import os
import json
import time
import random
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Firebase config từ environment
FIREBASE_DB_URL = os.getenv('FIREBASE_DB_URL', 'https://serious-hold-468214-u2-default-rtdb.asia-southeast1.firebasedatabase.app')
FIREBASE_API_KEY = os.getenv('FIREBASE_API_KEY', '')

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
        
        # Get persons from Firebase (HTTP request)
        persons_url = f"{FIREBASE_DB_URL}/persons.json"
        persons_response = requests.get(persons_url, timeout=5)
        
        if persons_response.status_code != 200:
            return jsonify({
                "status": "error",
                "message": "Failed to fetch persons"
            }), 500
        
        persons = persons_response.json() or {}
        
        if not persons:
            return jsonify({
                "status": "error",
                "message": "No persons in database"
            }), 404
        
        # Random person (temporary - tạm thời)
        person_ids = list(persons.keys())
        random_person_id = random.choice(person_ids)
        matched_person = persons[random_person_id]
        
        # Check duplicate
        checkins_url = f"{FIREBASE_DB_URL}/events/{event_id}/checkins.json"
        checkins_response = requests.get(checkins_url, timeout=5)
        
        checkins = checkins_response.json() if checkins_response.status_code == 200 else {}
        
        # Check if person already checked in
        if isinstance(checkins, dict):
            for checkin_id, checkin in checkins.items():
                if isinstance(checkin, dict) and checkin.get('person_id') == random_person_id:
                    return jsonify({
                        "status": "duplicate",
                        "person_id": random_person_id,
                        "person_name": matched_person.get('name'),
                        "position": matched_person.get('position'),
                        "department": matched_person.get('department'),
                        "message": "Already checked in"
                    }), 200  # Changed from 409 to 200 - frontend will handle status field
        
        # Create checkin
        checkin_id = f"checkin_{int(time.time() * 1000)}"
        checkin_data = {
            "person_id": random_person_id,
            "person_name": matched_person.get('name'),
            "position": matched_person.get('position', 'N/A'),
            "department": matched_person.get('department', 'N/A'),
            "timestamp": int(time.time() * 1000),
            "image_url": matched_person.get('image_url')
        }
        
        # POST to Firebase
        post_url = f"{FIREBASE_DB_URL}/events/{event_id}/checkins/{checkin_id}.json"
        post_response = requests.put(post_url, json=checkin_data, timeout=5)
        
        if post_response.status_code not in [200, 201]:
            return jsonify({
                "status": "error",
                "message": "Failed to create checkin"
            }), 500
        
        return jsonify({
            "status": "success",
            "person_id": random_person_id,
            "person_name": matched_person.get('name'),
            "position": matched_person.get('position', 'N/A'),
            "department": matched_person.get('department', 'N/A'),
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
        checkins_url = f"{FIREBASE_DB_URL}/events/{event_id}/checkins.json"
        response = requests.get(checkins_url, timeout=5)
        
        checkins_dict = response.json() if response.status_code == 200 else {}
        
        # FIX: Convert dict to list for frontend
        checkins_list = []
        if isinstance(checkins_dict, dict):
            for checkin_id, checkin_data in checkins_dict.items():
                if isinstance(checkin_data, dict):
                    # Add checkin_id to the object for reference
                    checkin_item = {
                        "checkin_id": checkin_id,
                        **checkin_data
                    }
                    checkins_list.append(checkin_item)
        
        return jsonify({
            "status": "success",
            "event_id": event_id,
            "checkins": checkins_list,  # Now it's a list!
            "count": len(checkins_list)
        }), 200
    except Exception as error:
        print(f"Error in get_checkins: {error}")
        return jsonify({
            "status": "error",
            "message": str(error),
            "checkins": [],  # Return empty list on error
            "count": 0
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
