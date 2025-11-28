from flask import Flask, request, jsonify
from flask_cors import CORS 
import fitz  
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import traceback

app = Flask(__name__)
CORS(app)  

MODELS = {
    "xlmr": "cahya/xlm-roberta-large-indonesian-NER",  
    "indobert": "indobenchmark/indobert-base-p1",  
    "roberta": "flax-community/indonesian-roberta-base",
}

loaded_models = {}

ENTITY_COLORS = {
    "PER": "bg-red-300 text-red-900 font-medium px-1 rounded", 
    "ORG": "bg-blue-300 text-blue-900 font-medium px-1 rounded",  
    "LOC": "bg-yellow-300 text-yellow-900 font-medium px-1 rounded",  
    "MISC": "bg-green-300 text-green-900 font-medium px-1 rounded",  
    "O": "",  
}

def get_model(model_key: str):
    if model_key not in loaded_models:
        try:
            print(f"[INFO] Loading {MODELS[model_key]} ... (pertama kali ~1-2 menit)")
            tokenizer = AutoTokenizer.from_pretrained(MODELS[model_key])
            model = AutoModelForTokenClassification.from_pretrained(MODELS[model_key])
            # Pastikan max_len=512
            tokenizer.model_max_length = 512
            model.config.max_position_embeddings = 512
            loaded_models[model_key] = (tokenizer, model)
            print(f"[SUCCESS] Model {model_key} loaded! Labels: {list(model.config.id2label.values())}")
        except Exception as e:
            print(f"[ERROR] Gagal load model: {e}")
            raise
    return loaded_models[model_key]

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error ekstrak PDF: {e}")

def chunk_text(text: str, tokenizer, max_chunk=500, overlap=50):  
    """FIXED: Strict chunking tanpa overflow (algoritma sliding window aman)"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_size = min(max_chunk, len(tokens) - i)
        chunk = tokens[i:i + chunk_size]
        if chunk: 
            chunks.append(chunk)
        i += (max_chunk - overlap)
        if i >= len(tokens):
            break
    print(f"[DEBUG] Text: {len(tokens)} tokens → {len(chunks)} chunks (max {max_chunk})")
    return chunks

def predict_chunks(chunks, tokenizer, model):
    """Prediksi per chunk dengan no_grad & error trap"""
    model.eval()
    all_tokens = []
    all_labels = []
    with torch.no_grad():
        for idx, chunk in enumerate(chunks):
            try:
                if not chunk:
                    continue
                input_ids = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
                attention_mask = [1] * len(input_ids)
                if len(input_ids) > 512:
                    print(f"[WARN] Chunk {idx} overflow: {len(input_ids)} → truncate")
                    input_ids = input_ids[:510] + [tokenizer.sep_token_id] 
                    attention_mask = [1] * len(input_ids)

                input_ids = torch.tensor([input_ids])
                attention_mask = torch.tensor([attention_mask])

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()

                chunk_tokens = tokenizer.convert_ids_to_tokens(chunk)
                chunk_labels = [model.config.id2label.get(p, 'O') for p in preds[1:-1]]

                all_tokens.extend(chunk_tokens)
                all_labels.extend(chunk_labels)
            except Exception as e:
                print(f"[ERROR] Chunk {idx} fail: {e}")
                all_tokens.extend(chunk)
                all_labels.extend(['O'] * len(chunk))
    return all_tokens, all_labels

def tokens_to_html(tokens, labels):
    """Render HTML dengan cleaning subwords"""
    html_parts = []
    for token, label in zip(tokens, labels):
        clean = token
        if token.startswith("▁"):
            clean = " " + token[1:]
        elif token.startswith("##"):
            clean = token[2:]
        clean = clean.replace("Ġ", " ")  

        entity = label.replace("B-", "").replace("I-", "")
        color = ENTITY_COLORS.get(entity, "")

        if color and entity != "O":
            html_parts.append(f'<span class="{color}">{clean}</span>')
        else:
            html_parts.append(clean)
    return "".join(html_parts)

@app.route("/")
def home():
    return jsonify({"status": "LER Server OK", "port": 5000, "models": list(MODELS.keys())})

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

@app.post("/predict-pdf")
def predict_pdf():
    try:
        if "pdf" not in request.files:
            return jsonify({"error": "Tidak ada file PDF"}), 400

        file = request.files["pdf"]
        model_key = request.form.get("model", "xlmr")

        if file.filename == "":
            return jsonify({"error": "File kosong"}), 400

        pdf_bytes = file.read()
        text = extract_text_from_pdf(pdf_bytes)
        if not text:
            return jsonify({"error": "PDF kosong atau corrupt"}), 400

        tokenizer, model = get_model(model_key)

        chunks = chunk_text(text, tokenizer, max_chunk=500, overlap=50)
        tokens, labels = predict_chunks(chunks, tokenizer, model)

        if not tokens:
            return jsonify({"error": "Tidak ada teks yang diproses"}), 400

        result_html = tokens_to_html(tokens, labels)

        full_html = f"""
        <div class="mt-8 p-6 bg-gray-50 rounded-lg border-l-4 border-indigo-500">
            <h3 class="text-lg font-bold mb-4 text-indigo-700">Hasil NER (Model: {model_key.upper()})</h3>
            <div class="text-sm leading-relaxed whitespace-pre-wrap font-mono overflow-auto max-h-96">
                {result_html}
            </div>
            <p class="text-xs text-gray-500 mt-4">
                Stats: {len(tokens):,} tokens | {len(chunks)} chunks | {len(text):,} chars | Entitas unik: {len(set(l for l in labels if l != 'O'))}
            </p>
        </div>
        """
        return jsonify({"html": full_html, "success": True})

    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        print(f"[ERROR] {error_msg}\n{traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500

if __name__ == "__main__":
    print("=== LER SERVER STARTING (Port 5000) ===")
    print("Test: curl http://localhost:5000/health")
    app.run(host="0.0.0.0", port=5000, debug=True)  