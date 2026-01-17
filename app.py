"""
Flask backend for the Multilingual Fake News Detector.
Stubs real model calls with mocked responses if model not trained.
"""

from flask import Flask, request, jsonify, send_from_directory
import os
import random

app = Flask(__name__, static_folder='web', static_url_path='')

# Mock Model State
MOCK_MODE = True

@app.route('/')
def index():
    return send_from_directory('web', 'index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    if MOCK_MODE:
        # Simulate processing time
        import time
        time.sleep(1.5)
        
        # Simple heuristic for demo purposes
        is_fake = "miracle" in text.lower() or "shocking" in text.lower() or "guaranteed" in text.lower()
        confidence = 0.85 + (random.random() * 0.14)
        
        prediction = "Fake" if is_fake else "Real"
        score = confidence if is_fake else (1 - confidence) # normalize for UI
        
        # Generate dummy token attributions
        tokens = text.split()
        # Randomly assign high importance to some adjectives
        attributions = []
        for token in tokens:
            weight = 0.0
            if token.lower() in ["miracle", "shocking", "guaranteed", "secret"]:
                weight = 0.9
            else:
                weight = random.random() * 0.2
            attributions.append(weight)
            
        response = {
            'prediction': prediction,
            'confidence': confidence,
            'tokens': tokens,
            'attributions': attributions,
            'explanation_html': '<div>Visualizer not available in mock mode</div>'
        }
        return jsonify(response)
    else:
        # Import actual model
        # from src.models.classifier import FakeNewsClassifier
        # ... real inference ...
        pass

if __name__ == '__main__':
    print("Starting server on http://localhost:5000")
    app.run(debug=True, port=5000)
