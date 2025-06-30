from flask import Flask, request, jsonify, render_template
import requests
import os

app = Flask(__name__)

# קבלת טוקנים מהסביבה
HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
TOGETHER_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # אפשר לשנות למודל אחר מ together.ai

# מידע מותאם לעסק שלך
business_info = (
    "אתה בוט של עסק בשם 'NextWave AI & Web'.\n"
    "ענה רק לפי המידע הבא:\n"
    "- שירותים: בניית אתרים מותאמים אישית, יצירת בוטים חכמים.\n"
    "- זמני תגובה מהירים ותמיכה מקצועית.\n"
    "- לפרטים והזמנות ניתן לפנות דרך אתר האינטרנט או Fiverr.\n"
    "ענה על שאלות רק לפי המידע הזה והיה אדיב ומקצועי."
)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_message = data.get("message", "")

    # Headers - כל טוקן בנפרד בהתאם ל-API
    huggingface_headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
        "Content-Type": "application/json"
    }

    together_headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    # Payload ל-Huggingface
    huggingface_payload = {
        "inputs": f"{business_info}\n\nשאלה: {user_message}",
        "parameters": {
            "max_new_tokens": 100,
            "return_full_text": False
        }
    }

    # Payload ל-Together
    together_payload = {
        "model": TOGETHER_MODEL,
        "messages": [
            {"role": "system", "content": business_info},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }

    try:
        # קודם ניסיון של Together API
        response = requests.post(TOGETHER_API_URL, headers=together_headers, json=together_payload)
        response.raise_for_status()
        output = response.json()

        # בדיקה ותשובה מ-Together API
        if "choices" in output and output["choices"]:
            answer = output["choices"][0]["message"]["content"].strip()
        else:
            answer = "לא התקבלה תשובה מהמודל."

    except Exception as e:
        # אם Together נכשל, ניסיון ב-Huggingface
        try:
            response = requests.post(HUGGINGFACE_API_URL, headers=huggingface_headers, json=huggingface_payload)
            response.raise_for_status()
            output = response.json()

            if isinstance(output, list) and "generated_text" in output[0]:
                answer = output[0]["generated_text"]
            elif "error" in output:
                answer = f"שגיאה מהמודל: {output['error']}"
            else:
                answer = "לא התקבלה תשובה מהמודל."
        except Exception as e2:
            return jsonify({"error": f"שגיאה בשני ה-APIs: {str(e)} ; {str(e2)}"}), 500

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
