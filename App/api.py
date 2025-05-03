from flask import Flask, request, jsonify
import joblib
import numpy as np
from datetime import datetime
import pandas as pd
from typing import Dict

app = Flask(__name__)

try:
    artifacts = joblib.load('C:\\Users\\HP\\OneDrive\\Desktop\\Trinity Mobility Task\\Tranied_Model.joblib')
    model = artifacts['model']
    scaler = artifacts['scaler']
    le = artifacts['label_encoder']
    feature_order = artifacts['feature_order']
    
    print(" HVAC Prediction Artifacts Loaded Successfully :")
    print(f"- Model: {type(model).__name__}")
    print(f"- Features: {len(feature_order)} sensors")
    print(f"- Classes: {list(le.classes_)}")
except Exception as e:
    raise RuntimeError(f" Failed to load HVAC model artifacts: {str(e)}")


# Request model
class PredictionRequest:
    def __init__(self, timestamp: str, sensor_data: Dict[str, float]):
        self.timestamp = timestamp
        self.sensor_data = sensor_data

    @staticmethod
    def from_json(json_data):
        return PredictionRequest(
            timestamp=json_data['timestamp'],
            sensor_data=json_data['sensor_data']
        )


# Response model
class PredictionResponse:
    def __init__(self, timestamp: str, prediction: str, confidence: float, status: str, model: str, processed_at: str, diagnostics: dict):
        self.timestamp = timestamp
        self.prediction = prediction
        self.confidence = confidence
        self.status = status
        self.model = model
        self.processed_at = processed_at
        self.diagnostics = diagnostics

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "status": self.status,
            "model": self.model,
            "processed_at": self.processed_at,
            "diagnostics": self.diagnostics
        }

@app.route('/')
def root():
    return jsonify({
        "message": "HVAC Maintenance Prediction API",
        "model": type(model).__name__,
        "sensor_count": len(feature_order),
        "expected_sensors": feature_order
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        request_data = request.get_json()
        prediction_request = PredictionRequest.from_json(request_data)
        
        # Validate and process input
        pd.to_datetime(prediction_request.timestamp)
        features = [prediction_request.sensor_data[col] for col in feature_order]
        scaled = scaler.transform([features])
        
        # Get prediction and probabilities
        prediction = model.predict(scaled)
        proba = model.predict_proba(scaled)
        confidence = np.max(proba)
        
        # Create response without abnormal sensors data
        response = PredictionResponse(
            timestamp=prediction_request.timestamp,
            prediction=le.inverse_transform(prediction)[0],
            confidence=float(confidence),
            status="success",
            model=type(model).__name__,
            processed_at=datetime.now().isoformat(),
            diagnostics={
                "class_probabilities": dict(zip(le.classes_, [float(p) for p in proba[0]])),
            }
        )
        
        return jsonify(response.to_dict())

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        "model_type": type(model).__name__,
        "trained_classes": list(le.classes_),
        "required_sensors": feature_order,
        "input_example": {
            "timestamp": "2024-01-01T12:00:00",
            "sensor_data": {f"sensor_{i:02d}": 0.0 for i in range(52)}
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
