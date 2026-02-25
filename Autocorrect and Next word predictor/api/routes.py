from flask import Blueprint, request, jsonify
from services.keyboard_service import KeyboardService

api = Blueprint("api", __name__)

@api.route("/predict", methods=["POST"])
def predict():

    text = request.json["text"]

    corrected, predictions = KeyboardService.process(text)

    return jsonify({
        "corrected": corrected,
        "predictions": predictions
    })