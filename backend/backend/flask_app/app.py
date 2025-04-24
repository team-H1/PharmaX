import os
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import requests
import base64
import google.generativeai as genai

# ✅ Configure Gemini AI API
GEMINI_API_KEY = 'AIzaSyAHOYLPC-uWszyjX21mb6wdXUl7btXzeRQ'
VISION_API_KEY = 'AIzaSyBePEU4qRZHkk7kb_UipzTlWsnW-no87_I'
genai.configure(api_key=GEMINI_API_KEY)

# ✅ Initialize Flask App with CORS settings
app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "http://localhost:5173"}})

# ✅ Helper Function to Encode Image
def encode_image(image):
    return base64.b64encode(image.read()).decode("utf-8")

# ✅ Extract Text from Image using Google Vision API
def extract_text_google(image_base64):
    url = f"https://vision.googleapis.com/v1/images:annotate?key={VISION_API_KEY}"
    payload = {
        "requests": [
            {"image": {"content": image_base64}, "features": [{"type": "TEXT_DETECTION"}]}
        ]
    }
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        return None
    result = response.json()
    try:
        return result['responses'][0]['textAnnotations'][0]['description']
    except (KeyError, IndexError):
        return None

# ✅ Identify Tablet Name using Gemini AI
def identify_tablet_gemini(extracted_text):
    if not extracted_text:
        return None
    prompt = f"Extract the *exact medicine/tablet name* from the following text:\n\n{extracted_text}\n\nOnly return the medicine name. No extra text."
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip() if response else None

# ✅ Fetch Medicine Details based on a question
def get_medicine_details_with_question(medicine_name, question):
    if not medicine_name:
        return None
    prompt = f"For the medicine '{medicine_name}', provide a direct answer to:\n\n{question}\n\nEnsure your response is *clear, structured, and accurate* without disclaimers."
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip() if response else None

# ✅ Fetch Concise Medicine Details (for identify-tablet)
def get_concise_medicine_details(medicine_name):
    if not medicine_name:
        return None
    prompt = (
        f"For the medicine '{medicine_name}', provide the following structured details:\n"
        "1. Uses: \n2. Side Effects: \n3. Common Manufacturers: \n4. Components: \n5. Preferred Age:\n\n"
        "Ensure each point is *concise, fact-based, and clear* without disclaimers or unnecessary text."
    )
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip() if response else None

# ✅ Home Route
@app.route('/')
def home():
    return "PharmaX Flask Backend Running!"

# ✅ Identify Tablet from Image Endpoint (Backslip Identification)
@app.route('/api/identify-tablet', methods=['POST', 'OPTIONS'])
def identify_tablet():
    if request.method == "OPTIONS":
        response = make_response("")
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response, 200

    if 'image' not in request.files:
        return jsonify({"response": "No image provided."}), 400

    image = request.files['image']
    image_base64 = encode_image(image)
    extracted_text = extract_text_google(image_base64)
    if not extracted_text:
        return jsonify({"response": "Text extraction failed."}), 500

    tablet_name = identify_tablet_gemini(extracted_text)
    if not tablet_name:
        return jsonify({"response": "Tablet identification failed."}), 500

    summary = get_concise_medicine_details(tablet_name)
    summary = summary if summary else "No additional details available."
    return jsonify({"response": f"Tablet Name: {tablet_name}\n{summary}"}), 200

# ✅ Medicine Info Endpoint
@app.route('/api/medicine-info', methods=['POST', 'OPTIONS'])
def medicine_info():
    if request.method == "OPTIONS":
        response = make_response("")
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response, 200

    data = request.get_json()
    if not data or "tablet_name" not in data or "question" not in data:
        return jsonify({"response": "Tablet name and question are required."}), 400

    tablet_name = data["tablet_name"]
    question = data["question"]
    details = get_medicine_details_with_question(tablet_name, question)
    return jsonify({"response": details if details else "No details available."}), 200

# ✅ AI Chat Endpoint
@app.route('/api/ai-chat', methods=['POST', 'OPTIONS'])
def ai_chat():
    if request.method == "OPTIONS":
        response = make_response("")
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response, 200

    data = request.get_json()
    if not data or "tablet_name" not in data or "question" not in data:
        return jsonify({"response": "Tablet name and question are required."}), 400

    tablet_name = data["tablet_name"]
    question = data["question"]
    details = get_medicine_details_with_question(tablet_name, question)
    return jsonify({"response": details if details else "No details available."}), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)