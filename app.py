from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

# Setup
app = Flask(__name__)
CORS(app)

# Load the saved model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Text is required"}), 400

    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    label = "Real News" if prediction == 1 else "Fake News"

    return jsonify({"prediction": label})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
