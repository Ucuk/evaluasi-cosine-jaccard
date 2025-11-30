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

# ============================================================
# 1. KONFIGURASI DAN INISIALISASI SISTEM
# ============================================================

app = Flask(__name__)
CACHE_FILE = "stbi_cache_data.pkl"
DATASET_FILE = "Dataset_Abstrak_Final_Renumbered.csv"

# Sastrawi
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()
factory_sw = StopWordRemoverFactory()
stop_words_sastrawi = set(factory_sw.get_stop_words())

DF = None
PREPROCESSED_DATA = {}
TFIDF_MODELS = {}

# ============================================================
# 2. TEXT PROCESSING ULTIMATE VERSION
# ============================================================

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_document(text, use_stopword=True, use_stemming=True):
    text = clean_text(text)
    tokens = text.split()

    if use_stopword:
        tokens = [t for t in tokens if t not in stop_words_sastrawi]

    if use_stemming:
        tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)

def highlight_terms(text, query_terms):
    if not text:
        return ""
    text = html.escape(text)

    for term in query_terms:
        if len(term) > 2:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            text = pattern.sub(lambda m: f"<mark class='highlight'>{m.group(0)}</mark>", text)
    return text

def get_smart_snippet(text, query_terms, length=200):
    text_lower = text.lower()
    best_pos = 0

    for term in query_terms:
        pos = text_lower.find(term.lower())
        if pos != -1:
            best_pos = pos
            break

    start = max(0, best_pos - 50)
    end = min(len(text), start + length)
    snippet = text[start:end]

    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."

    return snippet

# ============================================================
# 3. CACHE SYSTEM (Sangat penting untuk speed)
# ============================================================

def get_variation_key(use_sw, use_stem):
    return f"sw_{use_sw}_stem_{use_stem}"

def load_or_build_cache():
    global DF, PREPROCESSED_DATA, TFIDF_MODELS

    if not os.path.exists(DATASET_FILE):
        print("❌ Dataset tidak ditemukan!")
        return False

    DF = pd.read_csv(DATASET_FILE)
    DF["Abstrak"] = DF["Abstrak"].fillna("")
    DF["Judul"] = DF["Judul"].fillna("")
    corpus_raw = DF["Abstrak"].tolist()

    # Jika cache ada → load
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)
                PREPROCESSED_DATA = data["preprocessed"]
                TFIDF_MODELS = data["tfidf"]
            print("✅ Cache berhasil dimuat!")
            return True
        except:
            print("⚠️ Cache rusak → rebuild...")

    # Build ulang (hanya variasi D = SW + Stem)
    print("🔨 Membangun cache baru (Stopword + Stemming)...")
    use_sw, use_stem = True, True
    key = get_variation_key(use_sw, use_stem)

    processed_corpus = []
    total = len(corpus_raw)

    for i, doc in enumerate(corpus_raw):
        processed_corpus.append(preprocess_document(doc, use_sw, use_stem))
        if (i + 1) % 50 == 0:
            print(f"   Processed {i+1}/{total}")

    PREPROCESSED_DATA[key] = processed_corpus

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(processed_corpus)

    TFIDF_MODELS[key] = {
        "vectorizer": vectorizer,
        "matrix": matrix
    }

    with open(CACHE_FILE, "wb") as f:
        pickle.dump({
            "preprocessed": PREPROCESSED_DATA,
            "tfidf": TFIDF_MODELS
        }, f)

    print("✅ Cache selesai dibuat!")
    return True

# ============================================================
# 4. GROUND TRUTH SYSTEM
# ============================================================

def generate_ground_truth(query_raw, df_data):
    query_clean = clean_text(query_raw)
    query_terms = set(query_clean.split())

    relevant_indices = set()

    for idx, row in df_data.iterrows():
        title_clean = clean_text(row["Judul"])
        title_tokens = set(title_clean.split())
        if not query_terms.isdisjoint(title_tokens):
            relevant_indices.add(idx)

    return relevant_indices

# ============================================================
# 5. SISTEM PENCARIAN ULTIMATE (COSINE ONLY)
# ============================================================

def run_search(query):
    start_time = time.time()

    # Only Variasi D
    use_sw, use_stem = True, True
    key = get_variation_key(use_sw, use_stem)

    processed_query = preprocess_document(query, use_sw, use_stem)

    if not processed_query:
        return [], 0.0, processed_query, 0.0, 0, []

    model = TFIDF_MODELS[key]
    vectorizer = model["vectorizer"]
    doc_matrix = model["matrix"]

    query_vec = vectorizer.transform([processed_query])
    cosine_sim = cosine_similarity(query_vec, doc_matrix).flatten()

    ranked = [(i, s) for i, s in enumerate(cosine_sim) if s > 0]
    ranked.sort(key=lambda x: x[1], reverse=True)

    top_k = 10
    retrieved = [x[0] for x in ranked[:top_k]]
    relevant_set = generate_ground_truth(query, DF)

    precision = len([i for i in retrieved if i in relevant_set]) / len(retrieved) if retrieved else 0

    results = []
    query_terms = query.split()

    for rank, (idx, score) in enumerate(ranked[:top_k]):
        row = DF.iloc[idx]

        snippet = get_smart_snippet(row["Abstrak"], query_terms)
        highlighted_snip = highlight_terms(snippet, query_terms)
        highlighted_title = highlight_terms(row["Judul"], query_terms)

        results.append({
            "rank": rank + 1,
            "no": row["No."],
            "judul": highlighted_title,
            "abstrak_snippet": highlighted_snip,
            "score": score,
            "is_relevant": idx in relevant_set
        })

    exec_time = time.time() - start_time
    return results, precision, processed_query, exec_time, len(relevant_set), list(relevant_set)

# ============================================================
# 6. UI HTML ULTIMATE DEWA
# ============================================================

HTML_TEMPLATE = """
<!doctype html>
<html lang="id">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>TF-IDF Cosine Similarity - Ultimate Version</title>

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <style>
        body { background: #ecf0f1; }
        .main-container { max-width: 1100px; margin: 40px auto; background: white; padding: 35px; border-radius: 15px; }
        mark.highlight { background: #f1c40f; padding: 2px 4px; border-radius: 4px; }
        .result-card { border-left: 5px solid #3498db; padding: 18px; margin-bottom: 20px; border-radius: 8px; }
        .badge-score { background: #3498db; }
    </style>
</head>
<body>

<div class="main-container">

    <h2 class="fw-bold text-center mb-4">
        Sistem Temu Kembali Informasi<br>
        <span class="text-primary">TF-IDF + Cosine Similarity (Stopword + Stemming)</span>
    </h2>

    <form method="POST">
        <label class="fw-bold">Masukkan Kueri:</label>
        <input type="text" class="form-control form-control-lg" name="query" value="{{ query_val }}" required>

        <button class="btn btn-primary mt-4 w-100 py-2 fw-bold">
            🔍 Jalankan Pencarian
        </button>
    </form>

    {% if results is not none %}
    <div class="mt-5 p-3 bg-dark text-white rounded">
        <div class="row text-center">
            <div class="col">
                <h6>Total Ground Truth (Relevan)</h6>
                <h3>{{ total_relevant }}</h3>
            </div>
            <div class="col">
                <h6>Precision @ 10</h6>
                <h3>{{ "%.1f"|format(precision * 100) }}%</h3>
            </div>
            <div class="col">
                <h6>Waktu Komputasi</h6>
                <h3>{{ "%.4f"|format(exec_time) }}s</h3>
            </div>
        </div>

        <div class="text-center small mt-2">
            Processed Query: <b>{{ processed_query }}</b>
        </div>
    </div>

    <!-- Tombol Ground Truth -->
    <div class="text-center mt-4">
        <button class="btn btn-warning fw-bold" data-bs-toggle="modal" data-bs-target="#gtModal">
            📘 Lihat Dokumen Ground Truth
        </button>
    </div>

    <!-- Modal Ground Truth -->
    <div class="modal fade" id="gtModal" tabindex="-1">
      <div class="modal-dialog modal-lg">
        <div class="modal-content">
          
          <div class="modal-header">
            <h5 class="modal-title fw-bold">Daftar Dokumen Ground Truth</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
          </div>

          <div class="modal-body">

            {% if ground_truth_list %}
                <ul class="list-group">
                {% for idx in ground_truth_list %}
                    <li class="list-group-item">
                        <b>No {{ DF.iloc[idx]['No.'] }}:</b>
                        {{ DF.iloc[idx]['Judul'] }}
                    </li>
                {% endfor %}
                </ul>
            {% else %}
                <div class="alert alert-info">Tidak ada dokumen relevan berdasarkan Ground Truth.</div>
            {% endif %}

          </div>

          <div class="modal-footer">
            <button class="btn btn-secondary" data-bs-dismiss="modal">Tutup</button>
          </div>

        </div>
      </div>
    </div>

    <h4 class="mt-5 fw-bold">Hasil Peringkat Dokumen</h4>

    {% for item in results %}
    <div class="result-card">
        <h5 class="fw-bold">
            #{{ item.rank }} — {{ item.judul|safe }}
        </h5>

        {% if item.is_relevant %}
            <span class="badge bg-success mb-2">Relevan (Ground Truth)</span>
        {% else %}
            <span class="badge bg-secondary mb-2">Tidak Relevan</span>
        {% endif %}

        <span class="badge badge-score ms-2">{{ "%.5f"|format(item.score) }}</span>

        <p class="mt-2">{{ item.abstrak_snippet|safe }}</p>
    </div>
    {% endfor %}

    {% endif %}

</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# ============================================================
# 7. ROUTE FLASK
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query", "")
        results, precision, proc_query, exec_time, tot_rel, gt_list = run_search(query)

        return render_template_string(
            HTML_TEMPLATE,
            query_val=query,
            results=results,
            precision=precision,
            processed_query=proc_query,
            exec_time=exec_time,
            total_relevant=tot_rel,
            ground_truth_list=gt_list,
            DF=DF
        )

    return render_template_string(
        HTML_TEMPLATE,
        query_val="",
        results=None,
        DF=DF
    )

# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":
    print("\n🚀 MENYIAPKAN SISTEM ULTIMATE DEWA...")
    load_or_build_cache()
    print("🌐 Aplikasi berjalan di http://127.0.0.1:5002")
    app.run(debug=True, port=5002, use_reloader=False)
