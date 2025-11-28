# Legal Entity Recognition (LER) Indonesia  
### Ekstraksi Otomatis Entitas Hukum dari PDF Putusan Pengadilan

Aplikasi berbasis AI yang mampu mengenali dan mewarnai secara otomatis:  
- **Nama Orang** (Terdakwa, Hakim, Jaksa, Advokat) → merah  
- **Organisasi / Institusi** (Pengadilan Negeri, Kejaksaan Negeri, dll) → biru  
- **Lokasi** (Jakarta Pusat, Surabaya, Ruang Sidang Cakra) → kuning  
- **Lain-lain** (nomor perkara, UU, tanggal) → hijau  

Menggunakan model **XLM-RoBERTa Large** fine-tune bahasa Indonesia (akurasi ~88% pada teks hukum).

## Fitur
- Upload PDF putusan (PN, PT, MA, MK)
- Highlight otomatis semua entitas hukum
- Legenda warna + tombol Reset
- Tidak butuh internet setelah model pertama kali di-download
- 100% offline & gratis

## Prasyarat
- Python 3.10 – 3.12
- RAM minimal 8 GB (disarankan 16 GB untuk model large)
- Windows / macOS / Linux

## Cara Menjalankan (5 menit saja!)

### 1. Clone / Download proyek ini
```bash
git clone https://github.com/nama-kamu/legal-entity-recognition-indonesia.git
cd legal-entity-recognition-indonesia
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
pip install -r requirements.txt
python backend/app.py
```
[INFO] Loading cahya/xlm-roberta-large-indonesian-NER ...
[SUCCESS] Model xlmr loaded!
 * Running on http://0.0.0.0:5000

Buka aplikasi di browser
Cukup dobel-klik file ini →
frontend/index.html
atau buka manual: http://127.0.0.1:5000 (kalau pakai Live Server VS Code juga bisa)
6. Gunakan!

Pilih file PDF putusan pengadilan
Pilih model (rekomendasi: XLM-RoBERTa Large)
Klik PROSES PDF
Tunggu 15–50 detik → hasil muncul dengan warna-warni + legend
