from pathlib import Path
import tempfile

from flask import Blueprint, jsonify, render_template, request

from application.extractor import extract_symptoms
from application.image_classifier import predict_image


def create_routes(hpo_terms):
	routes = Blueprint("routes", __name__)

	@routes.get("/")
	def index():
		return render_template("index.html")

	@routes.get("/api/health")
	def health():
		model_path = Path("models/injury_classifier.joblib")
		return jsonify({
			"status": "ok",
			"hpo_terms": len(hpo_terms),
			"image_model_available": model_path.exists(),
		})

	@routes.post("/api/nlp/extract")
	def extract():
		payload = request.get_json(silent=True) or {}
		text = payload.get("text")
		if not isinstance(text, str) or not text.strip():
			return jsonify({"error": "JSON field 'text' is required."}), 400

		results = extract_symptoms(text, hpo_terms)
		return jsonify({
			"text": text,
			"results": [
				{
					"id": term["id"],
					"name": term["name"],
					"score": round(score, 2),
				}
				for term, score in results
			],
		})

	@routes.post("/api/image/predict")
	def predict():
		uploaded_file = request.files.get("image")
		if uploaded_file is None or not uploaded_file.filename:
			return jsonify({"error": "Multipart file field 'image' is required."}), 400

		model_path = Path("models/injury_classifier.joblib")
		if not model_path.exists():
			return jsonify({
				"error": "Image model is not trained yet.",
				"next_step": "Run train_image_classifier.py with a labeled dataset.",
			}), 503

		suffix = Path(uploaded_file.filename).suffix.lower()
		temporary_path = None
		try:
			with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
				temporary_path = temporary_file.name
			uploaded_file.save(temporary_file.name)
			prediction = predict_image(temporary_path, model_path)
		except (OSError, ValueError) as error:
			return jsonify({"error": f"Unable to process image: {error}"}), 400
		finally:
			if temporary_path:
				Path(temporary_path).unlink(missing_ok=True)

		return jsonify(prediction)

	return routes
