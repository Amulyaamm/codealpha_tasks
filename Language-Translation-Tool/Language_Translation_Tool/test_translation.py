
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

text = "Hello, how are you?"

tokenizer.src_lang = "eng_Latn"

inputs = tokenizer(
    text,
    return_tensors="pt"
)

translated_tokens = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("kan_Knda"),
    max_length=128
)

translation = tokenizer.batch_decode(
    translated_tokens,
    skip_special_tokens=True
)[0]

print("\nEnglish:")
print(text)

print("\nKannada:")
print(translation)

