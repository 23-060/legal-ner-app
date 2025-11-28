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
- Legend warna + tombol Reset

## Prasyarat
- Python 3.10 – 3.12
- RAM minimal 8 GB (disarankan 16 GB untuk model large)
- Windows / macOS / Linux

## Cara Menjalankan (5 menit saja!)

### Clone / Download proyek ini
```bash
git clone https://github.com/nama-kamu/legal-entity-recognition-indonesia.git
cd legal-entity-recognition-indonesia
python -m venv venv

```

### Buat virtual environment (wajib!)
```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Run Backend nya
```bash
pip install -r requirements.txt
python backend/app.py
```

### Buka aplikasi di browser untuk Frontend nya
Cukup dobel-klik file ini : frontend/index.html
atau buka manual: http://127.0.0.1:5000 (kalau pakai Live Server VS Code juga bisa)


#### Cara menggunakan Aplikasi

Pilih file PDF putusan pengadilan
Pilih model (rekomendasi: XLM-RoBERTa Large)
Klik PROSES PDF
Tunggu 15–50 detik → hasil muncul dengan warna-warni + legend
