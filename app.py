from flask import Flask, render_template, request, jsonify
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.cosmos import CosmosClient
import uuid, datetime, os

app = Flask(__name__)

# --- Azure Language config (set these in your terminal OR App Service later) ---
LANGUAGE_KEY = os.environ.get("LANGUAGE_KEY")
LANGUAGE_ENDPOINT = os.environ.get("LANGUAGE_ENDPOINT")

# --- Cosmos DB config ---
COSMOS_CONNECTION = os.environ.get("COSMOS_CONNECTION")
cosmos_client = CosmosClient.from_connection_string(COSMOS_CONNECTION)
db = cosmos_client.get_database_client("SentimentDB")
container = db.get_container_client("Results")

def get_language_client():
    return TextAnalyticsClient(endpoint=LANGUAGE_ENDPOINT, credential=AzureKeyCredential(LANGUAGE_KEY))

@app.route("/")
def index():
    results = list(container.read_all_items())
    results = sorted(results, key=lambda x: x.get("timestamp", ""), reverse=True)
    return render_template("index.html", results=results)

@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    client = get_language_client()

    # Sentiment
    sentiment_response = client.analyze_sentiment([text])[0]
    sentiment = sentiment_response.sentiment
    scores = sentiment_response.confidence_scores

    # Key phrases
    kp_response = client.extract_key_phrases([text])[0]
    key_phrases = kp_response.key_phrases

    # Save to Cosmos DB
    item = {
        "id": str(uuid.uuid4()),
        "text": text,
        "sentiment": sentiment,
        "positive": round(scores.positive, 2),
        "neutral": round(scores.neutral, 2),
        "negative": round(scores.negative, 2),
        "key_phrases": list(key_phrases),
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    container.create_item(item)

    return jsonify(item)

if __name__ == "__main__":
    app.run(debug=True)