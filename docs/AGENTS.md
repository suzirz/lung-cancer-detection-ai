# AGENTS.md — Panduan untuk AI Coding Agent

Dokumen ini buat AI agent (Claude Code, Cursor, dll) yang bantu ngoding di repo `lung-cancer-detection-ai`. Baca `PRD.md`, `ARCHITECTURE.md`, dan `TODO.md` dulu sebelum mulai kerja.

## 1. Konteks Project
- Project portofolio + kompetisi (bukan produk medis real). Fokus: model klasifikasi hybrid CNN + ML klasik untuk CT scan paru
- Owner: pelajar SMK (RPL/PPLG), jadi kode harus **readable & well-commented**, bukan cuma "jalan"

## 2. Aturan Umum
- **Jangan** mengubah struktur folder di `ARCHITECTURE.md` tanpa alasan kuat — kalau perlu, update dokumentasinya juga
- **Selalu** pisahkan logic preprocessing, training, dan inference ke modul berbeda (lihat `src/`)
- **Jangan** hardcode path absolut — pakai config/`.env` atau argumen CLI
- Setiap fungsi training harus punya seed yang bisa di-set (reproducibility)
- Simpan model hasil training ke `models/`, jangan commit file besar (>50MB) ke git — pakai `.gitignore`

## 3. Konvensi Kode
- Python, ikuti PEP8
- Nama file: `snake_case.py`
- Docstring tiap fungsi utama (boleh singkat, format Google-style)
- Notebook (`notebooks/`) cuma buat eksplorasi — kode final harus dipindah ke `src/`

## 4. Testing & Validasi
- Sebelum training penuh, selalu test pipeline preprocessing di sample kecil (5-10 gambar) dulu
- Evaluasi wajib pakai: accuracy, precision, recall, F1, AUC-ROC, confusion matrix
- **Recall/sensitivity adalah prioritas utama** (lihat PRD) — jangan cuma optimasi accuracy

## 5. Kalau Agent Ragu
- Kalau ada ambiguitas soal dataset/arsitektur, cek `PRD.md` & `ARCHITECTURE.md` dulu sebelum nanya
- Kalau task belum ada di `TODO.md`, tambahin dulu sebagai item baru sebelum dikerjain
- Jangan install dependency baru yang berat (misal PyTorch Lightning, DVC) tanpa dicatat di `requirements.txt` + alasan singkat

## 6. Command Umum (isi setelah setup repo)
```bash
# setup
pip install -r requirements.txt

# preprocessing
python src/preprocessing/run.py

# training
python src/training/train.py --config configs/default.yaml

# demo
streamlit run app/main.py
```

## 7. Update Dokumen
Kalau agent bikin perubahan arsitektur/scope signifikan, **update `ARCHITECTURE.md` / `PRD.md` di PR yang sama** — jangan biarkan dokumen basi.
