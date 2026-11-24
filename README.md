# Eksperimen Perbandingan Klasifikasi Sentimen: Scikit-learn vs Gemini LLM

Dokumentasi proyek eksperimen awal untuk membandingkan dua pendekatan klasifikasi sentimen otomatis (`positif` / `negatif`) pada ulasan pelanggan e-commerce: **model machine learning klasik (Scikit-learn)** dan **LLM API (Gemini)**. Seluruh eksperimen, metrik, waktu eksekusi, dan rekomendasi di bawah ini berasal langsung dari implementasi dan hasil aktual pada [notebook/experiment_notebook.ipynb](notebook/experiment_notebook.ipynb).

---

## Daftar Isi

- [1. Problem Statement dan Dataset](#1-problem-statement-dan-dataset)
- [2. Ringkasan Eksperimen Kedua Pendekatan](#2-ringkasan-eksperimen-kedua-pendekatan)
- [3. Hasil Evaluasi dan Perbandingan](#3-hasil-evaluasi-dan-perbandingan)
- [4. Analisis Trade-off dan Limitation](#4-analisis-trade-off-dan-limitation)
- [5. Rekomendasi Technical Approach](#5-rekomendasi-technical-approach)
- [6. Struktur Repository](#6-struktur-repository)
- [7. Panduan Menjalankan Proyek](#7-panduan-menjalankan-proyek)

---

## 1. Problem Statement dan Dataset

### Latar Belakang dan Tujuan

Sebuah tim produk e-commerce ingin membangun fitur otomatis untuk mengklasifikasikan sentimen ulasan pelanggan (`positif` atau `negatif`) pada halaman produk. Sebelum menentukan arsitektur produksi, dilakukan eksperimen perbandingan yang adil di atas **test set yang sama persis** antara:

1. **Pendekatan 1 — Model Klasik (Scikit-learn)**: Melatih model klasifikasi berbasis TF-IDF di atas dataset ulasan yang tersedia.
2. **Pendekatan 2 — LLM API (Gemini)**: Memanfaatkan Gemini API secara *zero-shot* tanpa proses training konvensional.

### Deskripsi dan Karakteristik Dataset

Dataset utama berada di [data/customer_reviews_sentiment.csv](data/customer_reviews_sentiment.csv) dengan karakteristik aktual sebagai berikut:

- **Ukuran dataset**: `200` baris dan `4` kolom (`review_id`, `product_name`, `review_text`, `sentiment`).
- **Fitur dan target**: Kolom teks `review_text` sebagai fitur input dan `sentiment` sebagai label target biner (`positif` / `negatif`).
- **Kualitas data**: `0` missing value dan `0` baris duplikat penuh; mencakup `15` `product_name` unik dengan panjang ulasan `6`–`11` kata (median `8.0` kata).
- **Distribusi label keseluruhan**: `positif` sebanyak `110` baris (`55%`) dan `negatif` sebanyak `90` baris (`45%`).
- **Temuan kritis (duplikasi teks)**: Dari `200` baris, hanya terdapat **`40` teks ulasan (`review_text`) unik** yang masing-masing berulang `1`–`12` kali di berbagai produk tanpa label yang bertentangan (`0` konflik label).

### Pembagian Data (Train/Test Split)

Untuk mencegah *exact-text data leakage* akibat teks identik yang berulang, dataset dibagi menggunakan `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` dengan `groups=review_text`:

- **Training set**: `159` baris (`32` teks unik; `83` positif, `76` negatif).
- **Test set**: `41` baris (`8` teks unik; `27` positif / `65.9%`, `14` negatif / `34.1%`).
- **Overlap teks train vs test**: `0` teks identik.

---

## 2. Ringkasan Eksperimen Kedua Pendekatan

### A. Model Klasik (Scikit-learn)

- **Preprocessing teks (`clean_text`)**: Mengubah teks ke huruf kecil (*lowercase*), mengganti karakter non-alfanumerik menjadi spasi, dan menormalkan spasi ganda. Kata negasi Bahasa Indonesia (`tidak`, `bukan`, `kurang`, `jangan`) sengaja dipertahankan agar polaritas sentimen tidak hilang.
- **Ekstraksi fitur**: `TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=1, sublinear_tf=True)`, menghasilkan `338` fitur dari training set.
- **Seleksi model & algoritma**: Tiga kandidat dibandingkan di training set menggunakan 5-fold `GroupKFold` (`groups=X_train`):

| Kandidat Model | Parameter Utama | CV Accuracy (Mean) | CV F1-Weighted (Mean) | CV F1-Macro (Mean) |
|---|---|---:|---:|---:|
| **`LinearSVC`** *(Terpilih)* | `C=1.0`, `random_state=42` | **0.6536** | **0.6479** | **0.6518** |
| `MultinomialNB` | `alpha=1.0` | 0.5974 | 0.5940 | 0.5870 |
| `LogisticRegression` | `C=1.0`, `max_iter=1000`, `random_state=42` | 0.5466 | 0.5026 | 0.5351 |

- **Training**: Model terpilih (`LinearSVC` dalam `Pipeline` bersama `TfidfVectorizer`) dilatih ulang pada seluruh `159` baris training set.

### B. LLM API (Gemini)

- **Model**: `gemini-3.5-flash-lite` menggunakan SDK `google-genai`.
- **Strategi prompting**: **Zero-shot** tanpa contoh dari training maupun test set. Prompt meminta model mengklasifikasikan ulasan sebagai `"positif"` atau `"negatif"` dan menjawab hanya satu kata huruf kecil tanpa tanda baca atau penjelasan.
- **Parameter API**:
  - `temperature=0.0` (meminimalkan variasi/ketidakpastian output).
  - `max_output_tokens=10` (membatasi output hanya untuk satu kata label).
- **Mekanisme prediksi & error handling**:
  - Setiap baris pada `X_test` (`41` sampel) dipanggil secara sekuensial ke Gemini API.
  - Dilengkapi *exponential backoff retry* hingga `3` percobaan (jeda `1.5s`, `3.0s`) untuk menangani error sementara atau rate limit.
  - Respons mentah dibersihkan dengan `normalize_llm_label()` (menghapus karakter non-huruf, mengekstrak token `positif`/`negatif`), dan mengembalikan `'invalid'` jika format tidak valid atau seluruh retry gagal.

---

## 3. Hasil Evaluasi dan Perbandingan

Kedua pendekatan dievaluasi pada `41` sampel test set dan ground-truth label yang identik. Karena distribusi kelas di test set tidak seimbang (`27` positif vs `14` negatif), ringkasan utama menggunakan **`weighted` average**, dilengkapi **`macro` average** untuk memantau performa kelas minoritas (`negatif`).

### Tabel Perbandingan Utama

| Pendekatan | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | F1-Macro |
|---|---:|---:|---:|---:|---:|
| Scikit-learn (`LinearSVC`) | 0.6341 | 0.8234 | 0.6341 | 0.6276 | 0.6333 |
| Gemini LLM (`gemini-3.5-flash-lite`) | 0.8293 | 1.0000 | 0.8293 | 0.9050 | 0.9190 |

### Rincian Metrik per Kelas & Confusion Matrix

**1. Scikit-learn (`LinearSVC`)** — `26 / 41` prediksi benar:
- Kelas `negatif` (support `14`): Precision `0.4828`, Recall `1.0000`, F1-Score `0.6512`
- Kelas `positif` (support `27`): Precision `1.0000`, Recall `0.4444`, F1-Score `0.6154`
- Confusion Matrix (`baris = true [negatif, positif]`, `kolom = pred [negatif, positif]`):

```text
                Pred: negatif    Pred: positif
True: negatif        14               0
True: positif        15              12
```

**2. Gemini LLM (`gemini-3.5-flash-lite`)** — `34 / 41` prediksi benar (`7` prediksi berstatus `invalid` akibat API error):
- Kelas `negatif` (support `14`): Precision `1.0000`, Recall `0.9286`, F1-Score `0.9630`
- Kelas `positif` (support `27`): Precision `1.0000`, Recall `0.7778`, F1-Score `0.8750`
- Confusion Matrix untuk respons valid (`34` sampel sukses; `7` sampel `api_error` tidak masuk kolom prediksi valid):

```text
                Pred: negatif    Pred: positif
True: negatif        13               0
True: positif         0              21
```

Visualisasi heatmap Confusion Matrix berdampingan dan tabel inspeksi kesalahan tersedia langsung di Section 6.3 pada [notebook/experiment_notebook.ipynb](notebook/experiment_notebook.ipynb), sedangkan log prediksi lengkap per baris tersimpan di [results/predictions.csv](results/predictions.csv).

---

## 4. Analisis Trade-off dan Limitation

### Perbandingan Performa dan Pola Kesalahan

- **Keunggulan performa**: Gemini LLM unggul signifikan dibanding `LinearSVC` (`+19.52` poin Accuracy, `+27.74` poin F1-Score weighted) meskipun `7` dari `41` panggilan API gagal. Pada `34` panggilan yang berhasil (`status == 'ok'`), Gemini mencapai akurasi **100% (`34/34`)** tanpa salah klasifikasi antara `positif` dan `negatif`.
- **Pola kesalahan Scikit-learn (`LinearSVC`)**: Seluruh `15` kesalahan merupakan *false negative* pada kelas `positif` (Recall `positif` hanya `0.4444`):
  - `9` baris: *"Produknya rapi dan bersih, tidak ada cacat sama sekali."* diprediksi `negatif` karena mengandung token `tidak` dan `cacat` yang berasosiasi kuat dengan ulasan negatif di training set. Model TF-IDF gagal memahami negasi ganda (`tidak ada cacat` = positif).
  - `6` baris: *"Kualitas produk sangat memuaskan, akan order lagi."* diprediksi `negatif` karena frasa `kualitas produk` bertumpang tindih dengan ulasan negatif di training set dan kosakata positifnya kurang terwakili pada `32` teks unik training.
- **Pola kesalahan Gemini LLM**: `0` kesalahan semantik pada `34` panggilan sukses; seluruh `7` kegagalan (`6` ground-truth `positif`, `1` ground-truth `negatif` pada baris indeks `34`–`40`) disebabkan oleh `ClientError: 429 RESOURCE_EXHAUSTED` setelah 34 panggilan sekuensial.

### Effort Implementasi dan Kebutuhan Training

| Dimensi | Scikit-learn (`LinearSVC`) | Gemini LLM (`gemini-3.5-flash-lite`) |
|---|---|---|
| Preprocessing & Feature | Perlu merancang `clean_text`, tokenisasi, n-gram TF-IDF, dan menjaga kata negasi. | Minimal; fokus pada perancangan prompt dan normalisasi/validasi output. |
| Training & Seleksi | Wajib split bebas leakage, `GroupKFold` CV, dan `fit()` ulang pada data berlabel. | Tidak memerlukan training konvensional (*zero-shot*). |
| Konfigurasi & Error Handling | Deterministik, tanpa ketergantungan jaringan/API eksternal. | Membutuhkan pengelolaan API key, retry/backoff, penanganan rate limit, dan parser output. |
| Kebutuhan Dataset Berlabel | Tinggi: `32` teks unik training tidak cukup untuk generalisasi TF-IDF pada variasi kalimat baru. | Tidak perlu training set; cukup test set berlabel untuk evaluasi. |

### Kecepatan Eksekusi dan Inference

| Metrik Waktu (Terukur) | Scikit-learn (`LinearSVC`) | Gemini LLM (`gemini-3.5-flash-lite`) |
|---|---|---|
| Waktu training | `0.012088` detik (`159` baris, `338` fitur) | Tidak ada (`0` detik) |
| Total waktu inference (`41` sampel) | `0.004494` detik | `70.5785` detik (sekuensial) |
| Rata-rata latency per ulasan | `0.1096` ms | `1721.31` ms (keseluruhan) / `983.55` ms (`34` call sukses) / `5304.73` ms (`7` call error dengan retry) |

Inference lokal `LinearSVC` sekitar **15.700x lebih cepat** dibanding pemanggilan API Gemini. Pada volume ribuan ulasan per hari, panggilan sekuensial sinkron ke Gemini akan menjadi bottleneck latency dan memicu rate limit.

### Potensi Biaya Penggunaan Gemini API

- **Konsumsi token aktual**: Dari `34` panggilan sukses di test set, tercatat **`2.340` prompt tokens** (~`68.8` token/ulasan) dan **`68` output tokens** (`2.0` token/ulasan).
- **Catatan pengukuran biaya**: Biaya moneter aktual (dalam USD/IDR) **belum dihitung secara langsung** dalam eksperimen ini karena tarif bergantung pada paket/tier penagihan API yang aktif. Namun secara struktural, biaya Gemini bersifat variabel linear per ulasan (termasuk overhead token bila terjadi retry), sedangkan `LinearSVC` berjalan lokal dengan biaya API `0` setelah training selesai.

### Keterbatasan (Limitation)

- **Model Klasik (`LinearSVC`)**: Sangat bergantung pada variasi teks dataset (`32` teks unik training belum memadai), lemah dalam generalisasi frasa baru, dan gagal menangkap konteks semantik seperti negasi ganda.
- **LLM API (`gemini-3.5-flash-lite`)**: Terbukti rentan terhadap rate limit (`7/41` atau `17.1%` gagal dengan `429 RESOURCE_EXHAUSTED` meski sudah ada backoff 3 percobaan), latency tinggi (~`0.98` detik/call sukses), ketergantungan pada layanan eksternal, biaya token berulang, serta potensi variasi format output mentah (mis. kapitalisasi/tanda baca sehingga wajib dinormalisasi).

---

## 5. Rekomendasi Technical Approach

Berdasarkan hasil eksperimen aktual, pendekatan yang direkomendasikan adalah **arsitektur bertahap / hybrid** (bukan langsung memilih salah satu secara tunggal di kondisi saat ini):

1. **Fase 1 — Jangka Pendek (MVP Asinkron dengan Gemini API)**:
   - Gunakan `gemini-3.5-flash-lite` sebagai *background batch job* (bukan pemanggilan sinkron saat halaman produk dibuka) yang dilengkapi *rate limiter*, caching berbasis hash `review_text`, dan *retry queue*.
   - **Alasan**: Mengamankan akurasi semantik tinggi (`100%` pada panggilan sukses) tanpa membuat pengguna halaman produk terkena latency ~1 detik maupun kegagalan `429 RESOURCE_EXHAUSTED`.
2. **Fase 2 — Jangka Menengah/Panjang (Produksi Real-Time dengan Model Lokal)**:
   - Kumpulkan ulasan nyata bervariasi yang telah dilabeli di Fase 1 (serta diaudit sampelnya), lalu latih model klasifikasi lokal (Scikit-learn pada dataset yang lebih kaya atau model teks lokal ringan) untuk menggantikan inferensi API pada trafik tinggi.
   - **Alasan**: Menghilangkan bottleneck skalabilitas dan ketergantungan API eksternal sekaligus mempertahankan kecepatan inferensi sub-milidetik (`~0.11 ms`/ulasan pada `LinearSVC`) dan biaya operasional marginal mendekati nol.

**Mengapa tidak langsung memakai salah satu secara penuh?**
- Memakai **`LinearSVC` langsung saat ini** berisiko menurunkan kualitas fitur produk karena Accuracy baru `0.6341` dan Recall kelas `positif` hanya `0.4444` akibat dataset latih baru punya `32` teks unik.
- Memakai **Gemini API secara sinkron di produksi** tidak layak dari sisi latency (`~0.98–1.72 s`/ulasan), risiko kegagalan kuota (`17.1%` error `429` pada burst `41` request), dan biaya operasional linear seiring pertumbuhan skala ulasan.

---

## 6. Struktur Repository

```text
.
├── .env                                 # Menyimpan GEMINI_API_KEY (diabaikan git)
├── .gitignore                           # Mengabaikan .venv dan .env
├── README.md                            # Dokumentasi eksperimen ini
├── requirements.txt                     # Daftar dependencies Python proyek
├── data/
│   └── customer_reviews_sentiment.csv   # Dataset ulasan pelanggan (200 baris, 4 kolom)
├── documentation/                       # Direktori dokumentasi tambahan (saat ini kosong)
├── notebook/
│   ├── experiment_notebook.ipynb        # Notebook utama berisi kode, output, plot, & analisis
│   └── starter_notebook.ipynb           # Kerangka awal notebook eksperimen
└── results/
    └── predictions.csv                  # Hasil prediksi test set (41 baris) kedua pendekatan
```

---

## 7. Panduan Menjalankan Proyek

### 1. Prasyarat

- **Python**: `3.13` (eksperimen dijalankan menggunakan `Python 3.13.2`).
- **Gemini API Key**: Diperoleh dari Google AI Studio untuk menjalankan eksperimen Pendekatan 2.

### 2. Clone Repository

```bash
git clone https://github.com/Atongers/assignment-model-experiment-fathan-ridho-wahyudi.git
cd assignment-model-experiment-fathan-ridho-wahyudi
```

### 3. Buat dan Aktifkan Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS (Bash):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Instal Dependencies

Dependensi proyek tercantum di [requirements.txt](requirements.txt) (`pandas`, `scikit-learn`, `python-dotenv`, `google-genai`, `matplotlib`, `seaborn`):

```bash
pip install -r requirements.txt
```

### 5. Konfigurasi Gemini API Key

Buat file `.env` di direktori utama repository (diabaikan oleh [.gitignore](.gitignore)), lalu isi variabel `GEMINI_API_KEY`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Notebook akan memuat variabel ini secara otomatis menggunakan `python-dotenv`.

### 6. Menjalankan Notebook Eksperimen

Eksperimen diimplementasikan seluruhnya pada [notebook/experiment_notebook.ipynb](notebook/experiment_notebook.ipynb) (tidak ada script CLI terpisah). Buka dan jalankan seluruh cell secara berurutan menggunakan Jupyter/VS Code Notebook dengan kernel `.venv`:

- **Dataset input**: [data/customer_reviews_sentiment.csv](data/customer_reviews_sentiment.csv)
- **Notebook eksperimen & visualisasi confusion matrix**: [notebook/experiment_notebook.ipynb](notebook/experiment_notebook.ipynb)
- **Output prediksi & log latency/token/error**: [results/predictions.csv](results/predictions.csv)
