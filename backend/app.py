from flask import Flask, request, jsonify
from flask_cors import CORS 
import fitz  
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import traceback
import PyPDF2
import re

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
checkpoint_path = "/models" # <- sesuaikan path disini

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)

print("Model & tokenizer loaded from", checkpoint_path)



id2label = model.config.id2label
# CLEANER TEXT
def clean_text(text: str) -> str:

    # Hapus karakter tidak berguna
    text = text.replace('\uf0d8', '').replace('\uf0b7', '')

    # Gabungkan newline berturut-turut
    text = re.sub(r'\n+', ' ', text)

    # 🔹 Hapus blok footer/header MA (semua varian umum)
    text = re.sub(
        r'(mahkamah\s+agung\s+republik\s+indonesia\s+){2,}.*?(putusan\.mahkamahagung\.go\.id)?',
        '',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # 🔹 Hapus baris yang mengandung nomor putusan + telp (pola footer kuat)
    text = re.sub(
        r'putusan\s+no(mor)?\s+[^\n]{0,200}?(telp|putusan\.mahkamahagung\.go\.id)',
        '',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # 🔹 Hapus pola telp footer tersendiri
    text = re.sub(
        r'telp\s*:\s*\d{2,3}[-\s]\d{3,4}[-\s]\d{3,4}(\s*\(ext\.?\s*\d+\))?',
        '',
        text,
        flags=re.IGNORECASE
    )

    # 🔹 Hapus pola "hal. putusan no ..."
    text = re.sub(
        r'hal\.?\s*(putusan|catatan)\s+no(mor)?\s+[^\n]{0,200}',
        '',
        text,
        flags=re.IGNORECASE
    )


    # Hapus disclaimer/peringatan yang sering muncul di dokumen
    text = re.sub(
        r'(dalam\s+hal-hal\s+tertentu.*?kepaniteraan\s+mahkamah\s+agung\s+ri\s+melalui:?\s*email:?\s*kepaniteraan@mahkamahagung\.go\.id)',
        '',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Hapus semua blok direktori/putusan MA yang muncul di tengah kalimat
    text = re.sub(
        r'direktori\s+putusan\s+mahkamah\s+agung\s+republik\s+indonesia\s+putusan\.mahkamahagung\.go\.id',
        '',
        text,
        flags=re.IGNORECASE
    )

    # Hapus pola "putusan nomor ... direktori putusan ... putusan.mahkamahagung.go.id" di tengah kalimat
    text = re.sub(
        r'putusan\s+no(mor)?\s+.*?direktori\s+putusan\s+mahkamah\s+agung\s+republik\s+indonesia\s+putusan\.mahkamahagung\.go\.id',
        '',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )



    text = re.sub(
        r'Disclaimer.*?(kepaniteraan@mahkamahagung\.go\.id)',
        '', text, flags=re.IGNORECASE | re.DOTALL
    )


    # Hapus info halaman
    text = re.sub(r'hal(?:\.|aman)\s*\d+(\s*dari\s*\d+)?', '', text, flags=re.IGNORECASE)

    text = re.sub(r'(Hal\.\s*\S+(?:\s*\S+)?\.?\s*)|(Halaman \d+(?:\.\d+)?\s*)|Putusan Nomor \S+\s*', '', text)


    # Normalisasi teks hasil OCR
    replacements = {
        r'p\s*u\s*t\s*u\s*s\s*a\s*n': 'PUTUSAN',
        r'p\s*e\s*n\s*e\s*t\s*a\s*p\s*a\s*n': 'PENETAPAN',
        r't\s*e\s*r\s*d\s*a\s*k\s*w\s*a': 'Terdakwa',
        r't\s*e\s*m\s*p\s*a\s*t': 'Tempat',
        r't\s*a\s*h\s*u\s*n': 'Tahun',
        r'j\s*u\s*m\s*l\s*a\s*h': 'Jumlah',
        r'm\s*e\s*n\s*g\s*a\s*d\s*i\s*l\s*i': 'MENGADILI',
        r'p\s*e\s*n\s*e\s*t\s*a\s*p\s*a\s*n':'PENETAPAN'
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Rapikan teks umum
    text = re.sub(r'[\u2026]+|\.{3,}', '', text)
    text = re.sub(r'-{2,}', '-', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s([,.;:!?])', r'\1', text)

    # Format hasil akhir
    parts = [t.strip() for t in text.strip().split(';') if t.strip()]
    return ';\n'.join(parts)



def clean_text2(text):
    text = text.replace('P U T U S A N', 'PUTUSAN').replace('T erdakwa', 'Terdakwa').replace('T empat', 'Tempat').replace('T ahun', 'Tahun')
    text = text.replace('P  E  N  E  T  A  P  A  N', 'PENETAPAN').replace('J u m l a h', 'Jumlah').replace('M E N G A D I L I', 'MENGADILI')
    text = re.sub(r'(Hal\.\s*\S+(?:\s*\S+)?\.?\s*)|(Halaman \d+(?:\.\d+)?\s*)|Putusan Nomor \S+\s*', '', text)
    # text = re.sub(r'\b0+(\d+)', r'\1', text)
    text = text.replace('\uf0d8', '').replace('\uf0b7', '').replace('\n', ' ')
    text = re.sub(r'([“”"])', r' \1 ', text) # -m spasi antara grup 1 tak hapus
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'halaman\s*\d+\s*dari\s*\d+\s*', '', text)
    text = re.sub(r'^\s*dari\s+\d+\s+skt\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+/PN', '/PN', text)
    text = re.sub(r'(?i)(nomor)(\d+)', r'\1 \2', text)
    text = re.sub(r'\(\s*(.?)\s\)', r'(\1)', text)
    text = re.sub(r':\s*(\S)', r': \1', text)
    text = re.sub(r',\s*(sh)', r' ,\1', text, flags=re.IGNORECASE)
    text = re.sub(r',\s*(s)', r' ,\1', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d)\s*\((.*?)\)', r'\1 (\2)', text)
    text = re.sub(r'(\d+)\s*/pid', r'\1/pid', text)
    text = re.sub(r'(\d+)\s*/\s*(pid\.\w+)\s*/\s*(\d{4})\s*/\s*(pn)', r'\1/\2/\3/\4', text, flags=re.IGNORECASE)

    return text.lower().strip()


def multiple_replace(text, replacements):
    regex = re.compile('|'.join(re.escape(key) for key in replacements.keys()))

    def replace(match):
        return replacements[match.group(0)]

    return regex.sub(replace, text)

replacements = {
    '1 0 (sepuluh)': '10 (sepuluh)', 'Nopember': 'November','Pebruari': 'Februari', 'Halaman1dari13Putusan' : 'Putusan', ':ABDUL' : 'ABDUL', 'A STUTI' :'ASTUTI', "SA'AT" : 'SA’AT', "Rifa'i" : 'Rifa’i',
    'N omor': 'Nomor', 'K itab': 'Kitab', 'Hakmelakukan': 'Hak melakukan', '20 20': '2020', 'ke manfatan': 'kemanfatan', '2 7' : '27', 'A gustus' : 'Agustus', 'September2023,' : 'September 2023,',
    '( 2)': '(2)', '.,': ' .,', ',S': ' ,S', 'PN.Pmk': 'PN Pmk', 'Pencuriandalam': 'Pencurian dalam', 'B in' : 'bin', 'Aapril' : 'April', 'Ja nuari' : 'Januari', 'Ted dy' : 'Teddy', 'Ro omius,' : 'Roomius,',
    'Pembunuhanberencana': 'Pembunuhan berencana', 'h ukum': 'hukum', 'PNPmk': 'PN Pmk', '//PN': '/PN','R .B.': 'R. B.', '2022oleh' : '2022', 'olehFajrini' : 'oleh Fajrini', 'Alimuddi n,' : 'Alimuddin,',
    '202 3/PN': '2023/PN', 'Perkos aan': 'Perkosaan', 'memberatka n': 'memberatkan', 'p idana': 'pidana', 'sebag aimana': 'sebagaimana', '1 0' : '10','bulan;' : 'bulan', '2 23' : '2023',
    'Ram dhani': 'Ramdhani', 'H ADY': 'HADY', '2 023': '2023','2 022': '2022', 'f isik': 'fisik', 'be rsalah': 'bersalah', '(sepuluh )': '(sepuluh)', 'M U A R I' : 'MUARI', 'Bulan;' : 'Bulan', 'Ap ril' : 'April',
    '(enam )': '(enam)', 'No mor': 'Nomor', '( 3)': '(3)', 'Nomo r': 'Nomor', 'an ak': 'anak', 'tuj uh': 'tujuh', 'hu bungan': 'hubungan', 'KADIR;' : 'KADIR', 'selam a1 (satu)' : 'selama 1 (satu)',
    '(empat )' : '(empat)', "PA'I" : 'PA’I', 'S eptember' : 'September',
    'Keadaa n' : 'Keadaan', '27Januari' : '27 Januari', 'Nomor265/Pid.B/2021/PN' : 'Nomor 265/Pid.B/2021/PN', 'Bulan;3.Menetapkan' : 'Bulan 3.Menetapkan','bulan;3.Menetapkan' : 'bulan 3.Menetapkan', 'Januari,2023' : 'Januari 2023',
    '202 2' : '2022','202 4' : '2024', '202 3' : '2023','2023//' : '2023', '20 22' : '2022','20 21' : '2021','202 1' : '2021', '27januari' : '27 januari', 'januari,2023' : 'januari 2023', 'sepember' : 'september',
    'Bk l' : 'Bkl', 'bk l' : 'bkl','pm k' : 'pmk', 'bki' : 'bkl',  'pid.b /' : 'pid.b/', 'pn.' : 'pn ', 'pid. b' : 'pid.b', 'pnpmk' : 'pn pmk', '2023pn' : '2023 pn', 'smp.' : 'smp',
    '(al m)' : '(alm)', 'Al m' : 'Alm', "'" : '’', 'b in' : 'bin', 'bu lan' : 'bulan', 'kuhpdan' : 'kuhp dan',
    'pencuriandalam' : 'pencurian dalam', 'keadaa n' : 'keadaan', 'pembunuhanberencana' : 'pembunuhan berencana', 'bulan3.menetapkan' : 'bulan 3.menetapkan', 'bulan3.menyatakan' :  'bulan 3.menyatakan',
    't aufik' : 'taufik', 'olehhaidir' : 'oleh haidir','olehfajrini' : 'oleh fajrini', 'janurai' : 'januari', 'nopember': 'november','pebruari': 'februari','a gustus' : 'agustus', 'september2023,' : 'september 2023,','aapril' : 'april', 'ja nuari' : 'januari',
    'no mor': 'nomor', ',serta' : ', serta', 'agussyamsul' : 'agus syamsul', 'a stuti' : 'astuti', 'widiati':'widiyati'
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

def ner_inference(text):
    # Tokenize
    encoded = tokenizer(text, return_tensors="pt", truncation=True)

    # Forward pass
    with torch.no_grad():
        outputs = model(**encoded)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"].squeeze())

    results = []
    for token, pred_id in zip(tokens, predictions):
        label = id2label[pred_id]

        # Skip special tokens
        if token.startswith("▁"):   # RoBERTa tokenizer, special whitespace symbol
            token = token[1:]

        if token not in ["<s>", "</s>", "<pad>"]:
            results.append((token, label))

    return results

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

        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n" # fungsi extract_text dari pypdf2.pdfreader()

        text=text.replace('\n',' ').split(';') # split dengan ';' supaya baris setiap konten lebih konsisten. karena ';' kandidat untuk akhir baris terbaik
        text=[t.strip()+';' for t in text] # jangan lupa tambahkan ';' lagi untuk menyesuaikan fungsi generateEntity
        text='\n'.join(text) # jadikan string lagi untuk disimpan
        # pdf_bytes = file.read()
        # text = extract_text_from_pdf(pdf_bytes)
        if not text:
            return jsonify({"error": "PDF kosong atau corrupt"}), 400

        text=clean_text(text)
        text=clean_text2(text)
        text=multiple_replace(text, replacements)
        sentences=[sentence+';' for sentence in text.split(';')]
        




        # mapping (token, label) per sentence
        listEntitiesFinal=[]
        for s in sentences:
            entities=ner_inference(s)
            
            # replace G with \s
            entitiesFinal=[]
            for i in range(len(entities)):
                entitiesFinal.append((entities[i][0].replace('Ġ', ' '), entities[i][1]))
                
            listEntitiesFinal.append(entitiesFinal)
        out=''
        for e in listEntitiesFinal:
            out+=' '.join(e)+"-------------------------------------<br>"
        full_html = f"""
        <div class="mt-8 p-6 bg-gray-50 rounded-lg border-l-4 border-indigo-500">
            <h3 class="text-lg font-bold mb-4 text-indigo-700">Hasil NER (Model: {model_key.upper()})</h3>
            <div class="text-sm leading-relaxed whitespace-pre-wrap font-mono overflow-auto max-h-96">
                {out}
            </div>
        </div>
        """
        return jsonify({"html": full_html, "success": True})

    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        print(f"[ERROR] {error_msg}\n{traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500

if __name__ == "__main__":
    print("=== LER SERVER STARTING (Port 5000) ===")
    app.run(host="0.0.0.0", port=5000, debug=True)  
