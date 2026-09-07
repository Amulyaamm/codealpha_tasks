from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


app = Flask(__name__)


# ============================================================
# TRANSLATION MODEL
# ============================================================

MODEL_NAME = "facebook/nllb-200-distilled-600M"

print("Loading translation model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = torch.device("cpu")

model.to(device)

print("Translation model loaded successfully on cpu")


# ============================================================
# LANGUAGE CODES
# ============================================================

LANGUAGES = {

    "en": "eng_Latn",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ml": "mal_Mlym",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn"

}


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# TRANSLATE
# ============================================================

@app.route("/translate", methods=["POST"])
def translate():

    try:

        data = request.get_json()

        print("\n========== TRANSLATION REQUEST ==========")
        print("Received:", data)


        if not data:

            return jsonify({
                "success": False,
                "message": "No data received."
            }), 400


        text = data.get("text", "").strip()

        source = data.get("source", "en")

        target = data.get("target", "hi")


        print("Text:", text)
        print("Source:", source)
        print("Target:", target)


        if not text:

            return jsonify({
                "success": False,
                "message": "Please enter text."
            }), 400


        if target not in LANGUAGES:

            return jsonify({
                "success": False,
                "message": "Unsupported target language."
            }), 400


        # ----------------------------------------------------
        # AUTO DETECTION
        # ----------------------------------------------------

        if source == "auto":

            source = "en"

            print(
                "Auto detection selected. "
                "Using English as default source."
            )


        if source not in LANGUAGES:

            return jsonify({
                "success": False,
                "message": "Unsupported source language."
            }), 400


        if source == target:

            return jsonify({
                "success": True,
                "translation": text
            })


        source_code = LANGUAGES[source]

        target_code = LANGUAGES[target]


        # ----------------------------------------------------
        # TOKENIZER
        # ----------------------------------------------------

        tokenizer.src_lang = source_code


        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )


        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }


        # ----------------------------------------------------
        # TRANSLATE
        # ----------------------------------------------------

        with torch.no_grad():

            generated_tokens = model.generate(

                **inputs,

                forced_bos_token_id=
                tokenizer.convert_tokens_to_ids(
                    target_code
                ),

                max_length=512

            )


        translation = tokenizer.batch_decode(

            generated_tokens,

            skip_special_tokens=True

        )[0]


        print("Translation:", translation)

        print("========================================\n")


        return jsonify({

            "success": True,

            "translation": translation

        })


    except Exception as e:

        print("\n========== TRANSLATION ERROR ==========")

        print(str(e))

        print("=======================================\n")


        return jsonify({

            "success": False,

            "message": str(e)

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "ok",

        "message": "LinguaTranslate server is running."

    })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=8000,

        debug=True

    )
