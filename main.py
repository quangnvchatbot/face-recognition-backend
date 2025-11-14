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

print("✅ Starting Flask application...")

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

print("✅ Flask app initialized")

# Firebase config
FIREBASE_DB_URL = os.getenv('FIREBASE_DB_URL', 'https://serious-hold-468214-u2-default-rtdb.asia-southeast1.firebasedatabase.app')

# ============ API ENDPOINTS ============

@app.route('/')
def home():
    print("📍 Root endpoint called")
    return jsonify({
        "status": "running",
        "service": "Face Recognition API",
        "version": "2.0",
        "message": "API is working!",
        "endpoints": {
            "health": "/health",
            "recognize": "/api/recognize (POST)",
            "test": "/api/test"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    print("📍 Health check called")
    return jsonify({
        "status": "ok",
        "timestamp": time.time(),
        "environment": "production"
    }), 200

@app.route('/api/test', methods=['GET'])
def test():
    print("📍 Test endpoint called")
    return jsonify({
        "status": "success",
        "message": "Test endpoint is working!",
        "firebase_url": FIREBASE_DB_URL
    }), 200

@app.route('/api/recognize', methods=['POST', 'OPTIONS'])
def recognize():
    if request.method == 'OPTIONS':
        return '', 204
    
    print("📍 Recognize endpoint called")
    try:
        return jsonify({
            "status": "success",
            "message": "Recognition endpoint is ready",
            "note": "Face recognition features will be added later"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "message": "Endpoint not found",
        "available_endpoints": [
            "/",
            "/health", 
            "/api/test",
            "/api/recognize"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    print(f"🚀 Starting Face Recognition Server on port {port}")
    print(f"📊 Firebase URL: {FIREBASE_DB_URL}")
    print("🔧 Starting Flask development server...")
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    print("🔧 App is being run by Gunicorn")
