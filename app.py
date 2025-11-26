import os
import time
import pickle
import re
import html
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- 1. KONFIGURASI DAN INISIALISASI ---
app = Flask(__name__)
CACHE_FILE = "stbi_cache_data.pkl"
DATASET_FILE = "Dataset_Abstrak_Final_Renumbered.csv"

# Inisialisasi Sastrawi
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()
factory_sw = StopWordRemoverFactory()
stop_words_sastrawi = set(factory_sw.get_stop_words())

# Global Variables (akan diisi saat warmup)
DF = None
PREPROCESSED_DATA = {}  # Menyimpan corpus hasil preprocessing untuk setiap variasi
TFIDF_MODELS = {}       # Menyimpan model Vectorizer dan Matrix untuk setiap variasi


# --- 2. FUNGSI UTILITY TEXT PROCESSING ---

def clean_text(text):
    """Membersihkan teks dasar (hapus karakter aneh, lowercase)."""
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text) # Hapus simbol
    text = re.sub(r'\s+', ' ', text).strip() # Hapus spasi ganda
    return text

def preprocess_document(text, use_stopword, use_stemming):
    """
    Melakukan preprocessing lengkap: Cleaning -> Tokenizing -> SW Removal -> Stemming
    """
    # 1. Cleaning
    text = clean_text(text)
    tokens = text.split()

    # 2. Stopword Removal
    if use_stopword:
        tokens = [t for t in tokens if t not in stop_words_sastrawi]

    # 3. Stemming (Proses terberat)
    if use_stemming:
        # Menggunakan map untuk mempercepat stemming per kata
        tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)

def highlight_terms(text, query_terms):
    """
    Memberikan highlight <mark> pada kata-kata query di dalam teks asli.
    Case-insensitive tapi mempertahankan format asli.
    """
    if not text: return ""
    
    # Escape HTML dulu untuk keamanan
    text = html.escape(text)
    
    for term in query_terms:
        if len(term) > 2:  # Hindari highlight kata terlalu pendek
            # Regex replace dengan ignore case
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            text = pattern.sub(lambda m: f"<mark class='highlight'>{m.group(0)}</mark>", text)
    return text

def get_smart_snippet(text, query_terms, length=200):
    """
    Mengambil potongan teks yang mengandung kata kunci, bukan cuma awal kalimat.
    """
    text_lower = text.lower()
    best_pos = 0
    
    # Cari posisi kata kunci pertama yang muncul
    for term in query_terms:
        pos = text_lower.find(term.lower())
        if pos != -1:
            best_pos = pos
            break
            
    start = max(0, best_pos - 50)
    end = min(len(text), start + length)
    
    snippet = text[start:end]
    if start > 0: snippet = "..." + snippet
    if end < len(text): snippet = snippet + "..."
    
    return snippet


# --- 3. CACHING & WARMUP SYSTEM (PERSISTENCE) ---

def get_variation_key(use_sw, use_stem):
    return f"sw_{use_sw}_stem_{use_stem}"

def load_or_build_cache():
    """
    Logika Utama: Cek apakah file cache ada di disk.
    Jika ADA: Load langsung (Cepat).
    Jika TIDAK: Proses dataset, lakukan stemming (Lama), lalu simpan ke disk.
    """
    global DF, PREPROCESSED_DATA, TFIDF_MODELS

    # Load Dataset CSV
    if not os.path.exists(DATASET_FILE):
        print(f"❌ Error: File {DATASET_FILE} tidak ditemukan!")
        return False
    
    DF = pd.read_csv(DATASET_FILE)
    DF['Abstrak'] = DF['Abstrak'].fillna("")
    DF['Judul'] = DF['Judul'].fillna("")
    corpus_raw = DF['Abstrak'].tolist()

    # Cek File Cache
    if os.path.exists(CACHE_FILE):
        print(f"📂 Menguat cache dari disk: {CACHE_FILE} ...")
        try:
            with open(CACHE_FILE, 'rb') as f:
                data = pickle.load(f)
                PREPROCESSED_DATA = data['preprocessed']
                TFIDF_MODELS = data['tfidf']
            print("✅ Cache berhasil dimuat! Aplikasi siap.")
            return True
        except Exception as e:
            print(f"⚠️ Gagal memuat cache (mungkin korup): {e}. Membangun ulang...")

    # Jika Cache tidak ada/gagal, bangun ulang
    print("🔨 Membangun ulang cache (Ini akan memakan waktu untuk Stemming)...")
    
    variations = [
        (False, False), # A
        (True, False),  # B
        (False, True),  # C
        (True, True)    # D
    ]

    for use_sw, use_stem in variations:
        key = get_variation_key(use_sw, use_stem)
        print(f"   ➡️ Processing Variasi: {key} (SW={use_sw}, STEM={use_stem})...")
        
        # 1. Preprocess Corpus
        processed_corpus = []
        total = len(corpus_raw)
        for i, doc in enumerate(corpus_raw):
            processed_corpus.append(preprocess_document(doc, use_sw, use_stem))
            if (i+1) % 50 == 0: print(f"      Processed {i+1}/{total} docs...")
        
        PREPROCESSED_DATA[key] = processed_corpus

        # 2. Build TF-IDF
        print(f"      Building TF-IDF Matrix...")
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(processed_corpus)
        
        TFIDF_MODELS[key] = {
            'vectorizer': vectorizer,
            'matrix': matrix
        }

    # Simpan ke Disk
    print("💾 Menyimpan cache ke disk...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({
            'preprocessed': PREPROCESSED_DATA,
            'tfidf': TFIDF_MODELS
        }, f)
    
    print("✅ Selesai! Cache tersimpan.")
    return True


# --- 4. LOGIKA PENCARIAN & EVALUASI ---

def generate_ground_truth(query_raw, df_data):
    """
    Membuat Ground Truth dinamis.
    Dokumen dianggap RELEVAN jika Judul-nya mengandung salah satu kata kunci query.
    """
    query_clean = clean_text(query_raw)
    query_terms = set(query_clean.split())
    
    relevant_indices = set()
    
    for idx, row in df_data.iterrows():
        title_clean = clean_text(row['Judul'])
        title_tokens = set(title_clean.split())
        
        # Cek irisan kata (intersection)
        if not query_terms.isdisjoint(title_tokens):
            relevant_indices.add(idx)
            
    return relevant_indices

def run_search(query, method, variation_code):
    start_time = time.time()
    
    # Mapping kode dropdown ke boolean
    configs = {
        'A': (False, False),
        'B': (True, False),
        'C': (False, True),
        'D': (True, True)
    }
    use_sw, use_stem = configs.get(variation_code, (True, True))
    var_key = get_variation_key(use_sw, use_stem)

    # 1. Preprocess Query
    processed_query = preprocess_document(query, use_sw, use_stem)
    
    # Jika query kosong setelah preprocessing (misal isinya cuma stopword "yang dan di")
    if not processed_query:
        return [], 0.0, processed_query, 0.0, 0

    scores = []
    
    # 2. Hitung Similarity
    if method == 'Cosine':
        model = TFIDF_MODELS[var_key]
        vectorizer = model['vectorizer']
        doc_matrix = model['matrix']
        
        query_vec = vectorizer.transform([processed_query])
        # Cosine similarity
        cosine_sim = cosine_similarity(query_vec, doc_matrix).flatten()
        scores = cosine_sim.tolist()
        
    elif method == 'Jaccard':
        corpus_list = PREPROCESSED_DATA[var_key]
        query_tokens = set(processed_query.split())
        
        for doc in corpus_list:
            doc_tokens = set(doc.split())
            intersection = len(query_tokens.intersection(doc_tokens))
            union = len(query_tokens.union(doc_tokens))
            score = intersection / union if union > 0 else 0
            scores.append(score)

    # 3. Ranking
    # Ambil indeks dokumen dengan skor > 0
    ranked_results = []
    for idx, score in enumerate(scores):
        if score > 0:
            ranked_results.append((idx, score))
    
    # Sort descending by score
    ranked_results.sort(key=lambda x: x[1], reverse=True)
    
    # 4. Evaluasi (Precision @ K)
    top_k = 10
    retrieved_indices = [x[0] for x in ranked_results[:top_k]]
    
    # Generate Ground Truth Dinamis berdasarkan Judul
    relevant_docs_set = generate_ground_truth(query, DF)
    
    # Hitung Precision
    relevant_retrieved = [idx for idx in retrieved_indices if idx in relevant_docs_set]
    precision = len(relevant_retrieved) / len(retrieved_indices) if retrieved_indices else 0

    # 5. Formatting Output
    final_results = []
    query_terms_for_highlight = query.split() # Gunakan query asli untuk highlight visual

    for rank, (idx, score) in enumerate(ranked_results[:top_k]):
        row = DF.iloc[idx]
        
        # Highlight & Snippet
        abstrak_ori = row['Abstrak']
        snippet_text = get_smart_snippet(abstrak_ori, query_terms_for_highlight)
        highlighted_snippet = highlight_terms(snippet_text, query_terms_for_highlight)
        highlighted_judul = highlight_terms(row['Judul'], query_terms_for_highlight)
        
        is_relevant = idx in relevant_docs_set

        final_results.append({
            'rank': rank + 1,
            'no': row['No.'],
            'judul': highlighted_judul,
            'abstrak_snippet': highlighted_snippet,
            'score': score,
            'is_relevant': is_relevant # Untuk memberi tanda di UI
        })

    exec_time = time.time() - start_time
    
    return final_results, precision, processed_query, exec_time, len(relevant_docs_set)


# --- 5. FLASK TEMPLATE (UI MODERN) ---

# --- 5. FLASK TEMPLATE (UI MODERN - JUDUL DISESUAIKAN) ---

HTML_TEMPLATE = """
<!doctype html>
<html lang="id">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Evaluasi Kinerja Cosine & Jaccard Similarity</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root { --primary-color: #2c3e50; --secondary-color: #34495e; --accent-color: #3498db; }
        body { background-color: #ecf0f1; font-family: 'Segoe UI', sans-serif; }
        
        .main-container { max-width: 1100px; margin: 40px auto; background: white; padding: 40px; border-radius: 15px; box-shadow: 0 15px 35px rgba(0,0,0,0.1); }
        
        .header-title { color: var(--primary-color); font-weight: 800; font-size: 1.8rem; letter-spacing: -0.5px; text-transform: uppercase; line-height: 1.4; }
        .header-subtitle { color: #7f8c8d; font-weight: 600; font-size: 1rem; margin-top: 10px; }
        
        .search-box { background: #f7f9fc; padding: 30px; border-radius: 12px; margin: 30px 0; border: 1px solid #e1e8ed; }
        
        .btn-search { background-color: var(--accent-color); border: none; padding: 12px 40px; font-weight: 700; border-radius: 8px; transition: all 0.3s; }
        .btn-search:hover { background-color: #2980b9; transform: translateY(-2px); box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3); }
        
        .result-card { border: 1px solid #eee; border-left: 5px solid var(--accent-color); margin-bottom: 20px; padding: 20px; border-radius: 8px; background: white; transition: transform 0.2s; }
        .result-card:hover { transform: translateX(5px); box-shadow: 0 5px 15px rgba(0,0,0,0.05); }
        
        .score-pill { font-size: 0.9rem; background: var(--primary-color); color: white; font-weight: bold; padding: 6px 15px; border-radius: 50px; }
        
        mark.highlight { background-color: #f1c40f; color: #2c3e50; padding: 0 3px; border-radius: 3px; font-weight: bold; }
        
        .stats-box { background: var(--primary-color); color: white; padding: 20px; border-radius: 12px; margin-bottom: 35px; }
        .stats-value { font-size: 1.8rem; font-weight: bold; color: #f1c40f; }
        
        /* Loading Spinner */
        #loading { position: fixed; width: 100%; height: 100%; top: 0; left: 0; background: rgba(255,255,255,0.9); z-index: 9999; display: none; justify-content: center; align-items: center; flex-direction: column; }
    </style>
</head>
<body>

    <div id="loading">
        <div class="spinner-border text-primary" style="width: 4rem; height: 4rem;" role="status"></div>
        <h4 class="mt-4 text-dark fw-bold">Sedang Memproses Kueri...</h4>
        <p class="text-muted">Menghitung Similaritas & Ranking Dokumen</p>
    </div>

    <div class="main-container">
        <div class="text-center mb-5">
            <h1 class="header-title">
                Evaluasi Kinerja Cosine Similarity dan Jaccard Similarity<br>
                <span style="font-size: 1.4rem; color: var(--accent-color);">dengan Variasi Preprocessing</span>
            </h1>
            <p class="header-subtitle">Untuk Sistem Temu Kembali Informasi Berbahasa Indonesia</p>
            <hr class="mt-4" style="width: 50%; margin: 0 auto; border-top: 3px solid #eee;">
        </div>

        <form method="POST" onsubmit="document.getElementById('loading').style.display = 'flex'">
            <div class="search-box">
                <div class="mb-4">
                    <label class="form-label fw-bold text-dark"><i class="fas fa-keyboard me-2"></i> Masukkan Kueri Pencarian:</label>
                    <input type="text" class="form-control form-control-lg shadow-sm" name="query" 
                           value="{{ query_val }}" placeholder="Contoh: pengembangan media pembelajaran berbasis android..." required>
                </div>
                
                <div class="row g-4">
                    <div class="col-md-6">
                        <label class="form-label fw-bold text-dark"><i class="fas fa-calculator me-2"></i> Metode Perhitungan:</label>
                        <select class="form-select form-select-lg" name="method">
                            <option value="Cosine" {% if method_val == 'Cosine' %}selected{% endif %}>Cosine Similarity (TF-IDF)</option>
                            <option value="Jaccard" {% if method_val == 'Jaccard' %}selected{% endif %}>Jaccard Similarity</option>
                        </select>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label fw-bold text-dark"><i class="fas fa-filter me-2"></i> Skenario Preprocessing:</label>
                        <select class="form-select form-select-lg" name="variation">
                            <option value="A" {% if var_val == 'A' %}selected{% endif %}>A. Tanpa Preprocessing (Raw)</option>
                            <option value="B" {% if var_val == 'B' %}selected{% endif %}>B. Stopword Removal Saja</option>
                            <option value="C" {% if var_val == 'C' %}selected{% endif %}>C. Stemming Saja (Sastrawi)</option>
                            <option value="D" {% if var_val == 'D' %}selected{% endif %}>D. Kombinasi (Stopword + Stemming)</option>
                        </select>
                    </div>
                </div>
                
                <div class="text-center mt-5">
                    <button type="submit" class="btn btn-primary btn-search">
                        <i class="fas fa-search me-2"></i> Jalankan Evaluasi
                    </button>
                </div>
            </div>
        </form>

        {% if results is not none %}
            <div class="stats-box">
                <div class="row text-center">
                    <div class="col-md-3 border-end border-secondary">
                        <div class="stats-label mb-1">Dokumen Relevan</div>
                        <div class="stats-value">{{ results|length }}</div>
                    </div>
                    <div class="col-md-3 border-end border-secondary">
                        <div class="stats-label mb-1">Waktu Komputasi</div>
                        <div class="stats-value">{{ "%.4f"|format(exec_time) }}s</div>
                    </div>
                    <div class="col-md-3 border-end border-secondary">
                        <div class="stats-label mb-1">Precision @ 10</div>
                        <div class="stats-value">{{ "%.1f"|format(precision * 100) }}%</div>
                    </div>
                    <div class="col-md-3">
                        <div class="stats-label mb-1">Total Ground Truth</div>
                        <div class="stats-value">{{ total_relevant }}</div>
                    </div>
                </div>
                <div class="mt-3 text-center text-white-50 small" style="font-family: monospace;">
                    Processed Query Tokens: [ {{ processed_query }} ]
                </div>
            </div>

            <h4 class="mb-4 fw-bold text-dark border-bottom pb-2"><i class="fas fa-list-ol me-2"></i> Hasil Peringkat Dokumen</h4>
            
            {% if results %}
                {% for item in results %}
                <div class="result-card">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <h5 class="mb-2 text-primary fw-bold">
                                <span class="badge bg-dark me-2">Rank #{{ item.rank }}</span>
                                {{ item.judul|safe }}
                            </h5>
                            <div class="mb-2">
                                {% if item.is_relevant %}
                                    <span class="badge bg-success"><i class="fas fa-check me-1"></i> Relevan (Ground Truth)</span>
                                {% else %}
                                    <span class="badge bg-secondary">Tidak Relevan</span>
                                {% endif %}
                                <span class="text-muted ms-2 small"><i class="fas fa-database me-1"></i> Data No. {{ item.no }}</span>
                            </div>
                            <p class="mb-0 text-dark" style="line-height: 1.7; text-align: justify;">
                                {{ item.abstrak_snippet|safe }}
                            </p>
                        </div>
                        <div class="ms-4 text-end" style="min-width: 100px;">
                            <div class="text-muted small mb-1">Similarity Score</div>
                            <div class="score-pill">{{ "%.5f"|format(item.score) }}</div>
                        </div>
                    </div>
                </div>
                {% endfor %}
            {% else %}
                <div class="alert alert-warning text-center p-5">
                    <i class="fas fa-exclamation-circle fa-3x mb-3"></i><br>
                    <h4 class="fw-bold">Tidak Ditemukan Dokumen yang Cocok</h4>
                    <p>Cobalah mengubah kata kunci atau mengganti metode preprocessing.</p>
                </div>
            {% endif %}

        {% endif %}
        
        <div class="text-center mt-5 text-muted small">
            &copy; 2025 Sistem Temu Kembali Informasi - Kelompok Penelitian JTIK UNM
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
# --- 6. FLASK ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form.get('query', '')
        method = request.form.get('method', 'Cosine')
        variation = request.form.get('variation', 'D')
        
        results, precision, proc_query, exec_time, tot_rel = run_search(query, method, variation)
        
        return render_template_string(HTML_TEMPLATE, 
                                      query_val=query, method_val=method, var_val=variation,
                                      results=results, precision=precision, 
                                      processed_query=proc_query, exec_time=exec_time,
                                      total_relevant=tot_rel)
    
    return render_template_string(HTML_TEMPLATE, 
                                  query_val="", method_val="Cosine", var_val="D",
                                  results=None)

# Load cache hanya 1x di startup environment
load_or_build_cache()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
