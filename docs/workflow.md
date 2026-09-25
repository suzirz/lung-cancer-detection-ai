# WORKFLOW — Development Process

## 1. Git Workflow
* Branch utama: `main` (selalu stabil, jangan push langsung ke sini)
* Branch kerja: `feature/nama-fitur` (misal `feature/cnn-baseline`, `feature/gradcam`)
* Commit message singkat & jelas: `[preprocessing] add HU windowing function`
* PR ke `main` setelah fitur selesai & ditest minimal

## 2. Alur Eksperimen (ML/DL)
1. Eksplorasi ide di `notebooks/` dulu (cepat iterasi)
2. Kalau udah stabil, pindahkan kode ke `src/` dalam bentuk fungsi/modul reusable
3. Catat hasil eksperimen (accuracy/recall/config yang dipakai) — bisa manual di file `experiments.md` atau pakai tool tracking
4. Jangan overwrite model lama — simpan versi (`model_v1.h5`, `model_v2.h5`, dst) sampai ada yang fix jadi final

## 3. Alur Kerja Harian (Solo Dev)
1. Cek `TODO.md` — ambil task minggu berjalan
2. Kerjain di branch terpisah
3. Test kecil dulu sebelum run full training (biar ga buang waktu kalau ada bug)
4. Commit + update `TODO.md` (centang task selesai)
5. Kalau ada perubahan arsitektur/scope → update `ARCHITECTURE.md`/`PRD.md`

## 4. Review Checklist Sebelum "Selesai"
- [ ] Kode di `src/` udah rapi, bukan sisa notebook mentah
- [ ] Semua path pakai config, bukan hardcode
- [ ] README update (cara install & run)
- [ ] Model & hasil evaluasi tersimpan
- [ ] Demo bisa dijalanin orang lain tanpa setup ribet

## 5. Kalau Stuck
* Cek lagi `SKILL.md` — mungkin ada gap knowledge yang perlu dipelajari dulu
* Cek `PRD.md` — kadang solusinya simplify scope, bukan push terus
* Diskusi/tanya AI agent dengan konteks dari `AGENTS.md`
