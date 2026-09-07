from flask import Flask, render_template, request, jsonify
from chatbot import get_response

app = Flask(__name__)


# ---------------- HOME ---------------- #

@app.route("/")
def home():
    return render_template("index.html")


# ---------------- CHAT PAGE ---------------- #

@app.route("/chat")
def chat():
    return render_template("chat.html")


# ---------------- ABOUT PAGE ---------------- #

@app.route("/about")
def about():
    return render_template("about.html")


# ---------------- CHAT API ---------------- #

@app.route("/chat_api", methods=["POST"])
def chat_api():

    data = request.get_json()

    message = data.get("message", "").strip()

    answer, confidence = get_response(message)

    return jsonify({
        "answer": answer,
        "confidence": round(confidence * 100, 2)
    })


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(debug=True)