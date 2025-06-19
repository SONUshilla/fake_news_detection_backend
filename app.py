from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Text is required"}), 400

    # Transform input text
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    label = "Real News" if prediction == 1 else "Fake News"

    # Get confidence (max probability)
    probabilities = model.predict_proba(vector)[0]
    confidence = round(max(probabilities) * 100, 2)  # as percentage

    return jsonify({
        "prediction": label,
        "confidence": f"{confidence}%"
    })
@app.route('/contact', methods=['POST'])
def contact():
    data = request.get_json()
    name = data.get("name", "")
    email = data.get("email", "")
    message = data.get("message", "")

    if not (name.strip() and email.strip() and message.strip()):
        return jsonify({"error": "All fields are required"}), 400

    # For now, just log it — later you can send email or store in DB
    print("📩 Contact Message Received:")
    print(f"Name: {name}")
    print(f"Email: {email}")
    print(f"Message: {message}")

    return jsonify({"success": "Message received successfully!"})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
