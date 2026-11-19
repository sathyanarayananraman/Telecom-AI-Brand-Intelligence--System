# ============================================================
#  ZENDS Brand Intelligence  –  Local Model Edition
#  Run:  streamlit run app_local.py
#
#  Requirements
#  ─────────────────────────────────────────────────────────────
#  pip install streamlit pandas scikit-learn joblib pypdf
#            chromadb openai
#  Ollama running locally on port 11434 with:
#    - mxbai-embed-large  (embeddings)
#    - llama3.2:1b        (LLM replies)
#
#  Folder layout expected beside this file
#  ─────────────────────────────────────────────────────────────
#  models/
#    tfidf_sent.pkl
#    sentiment_model.pkl
#    tfidf_cat.pkl
#    category_model.pkl
#  local/
#    vector_db/
#      chroma_db/        ← your persisted ChromaDB
#  data/
#    zends_feedback_clean.csv   ← historical synthetic feedback
#    feedback_log.csv           ← written at runtime (auto-created)
# ============================================================

import os, re
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
CHROMA_DIR = r"D:\python_projects\zends_project\local\vector_db\chroma_db"
DATA_DIR   = os.path.join(BASE_DIR, "data")
# os.path.join(DATA_DIR, "feedback_log.csv")
HIST_FILE  = os.path.join(DATA_DIR, "zends_feedback_clean.csv")
LOG_FILE   = HIST_FILE


OLLAMA_BASE = "http://localhost:11434/v1"
EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL   = "llama3.2:1b"

# ── Keyword override (kept identical to original app.py) ───────────────────────
def override_category_by_keywords(text: str, predicted: str) -> str:
    t = text.lower()
    if any(k in t for k in ["virtual machine", "cloud", "storage", "vm"]):
        return "Cloud & Data Center Services"
    if any(k in t for k in ["sensor", "iot", "traffic", "parking", "fleet", "predictive maintenance"]):
        return "IoT & Smart Solutions"
    if any(k in t for k in ["bill", "payment", "charge", "refund", "invoice"]):
        return "Billing & Payments"
    if any(k in t for k in ["sim", "roaming", "5g", "call", "signal"]):
        return "Mobile Network"
    if any(k in t for k in ["broadband", "router", "fiber", "wifi", "internet speed"]):
        return "Broadband Service"
    return predicted

# ── Model loading (cached once per session) ────────────────────────────────────
@st.cache_resource
def load_tfidf_models():
    return (
        joblib.load(os.path.join(MODEL_DIR, "tfidf_sent.pkl")),
        joblib.load(os.path.join(MODEL_DIR, "sentiment_model.pkl")),
        joblib.load(os.path.join(MODEL_DIR, "tfidf_cat.pkl")),
        joblib.load(os.path.join(MODEL_DIR, "category_model.pkl")),
    )

@st.cache_resource
def load_ollama_client():
    try:
        from openai import OpenAI
        return OpenAI(base_url=OLLAMA_BASE, api_key="ollama")
    except Exception:
        return None

@st.cache_resource
# def load_chroma():
#     try:
#         import chromadb
#         client     = chromadb.PersistentClient(path=CHROMA_DIR)
#         collection = client.get_or_create_collection(name="mannual")
#         return collection
#     except Exception:
#         return None
@st.cache_resource
def load_chroma():
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name="mannual")

# ── Sentiment + Category helpers ──────────────────────────────────────────────
# ── Neutral-intent override constants ─────────────────────────────────────────
# If model predicts Negative/Positive with confidence below this threshold
# AND the text looks like a plain question, override to Neutral.
NEUTRAL_CONFIDENCE_THRESHOLD = 0.75

# Opening words/phrases that strongly signal an informational question
QUESTION_OPENERS = (
    "what ", "which ", "how ", "when ", "where ", "who ", "why ",
    "is there", "are there", "do you", "does the", "can i", "can you",
    "could you", "tell me", "explain", "list ", "show me",
    "i would like to know", "i want to know", "do you offer",
    "what services", "which services", "what internet", "what plans",
    "what are the", "what is the",
)

def _is_informational(text: str) -> bool:
    """True if the text looks like a neutral enquiry rather than a complaint."""
    t = text.lower().strip()
    return t.endswith("?") or any(t.startswith(op) for op in QUESTION_OPENERS)

def predict_sentiment(text: str, tfidf_sent, sent_mdl) -> tuple[str, float | None]:
    """Returns (label, confidence | None) using local sentiment TF-IDF model.
    Label is title-cased so it always matches 'Negative'/'Positive'/'Neutral'.
    """
    vec   = tfidf_sent.transform([text])
    raw   = sent_mdl.predict(vec)[0]
    # Normalize regardless of how the model stored the class (lower / UPPER / Title)
    label = str(raw).strip().title()
    try:
        proba = float(sent_mdl.predict_proba(vec).max())
    except Exception:
        proba = None
    return label, proba

def predict_category(text: str, tfidf_cat, cat_mdl) -> str:
    """Returns predicted service category using local category TF-IDF model."""
    vec = tfidf_cat.transform([text])
    return str(cat_mdl.predict(vec)[0])

def analyse(
    text: str,
    tfidf_sent,
    sent_mdl,
    tfidf_cat,
    cat_mdl,
) -> tuple[str, float | None, str]:
    """
    Analyses sentiment and service category using separate vectorizers/models.
    Handles mixed-sentiment messages by splitting on 'but / however / although'.
    Returns (sentiment, confidence, category).
    """
    connectors = re.split(r"\b(but|however|although|yet|though)\b", text, flags=re.IGNORECASE)
    parts = [
        p.strip() for p in connectors
        if p.strip() and p.lower() not in ("but", "however", "although", "yet", "though")
    ]

    if len(parts) >= 2:
        sentiments = [predict_sentiment(p, tfidf_sent, sent_mdl)[0] for p in parts]
        unique = set(sentiments)
        if len(unique) > 1 and "Negative" in unique:
            neg_part = parts[sentiments.index("Negative")]
            cat_raw  = predict_category(neg_part, tfidf_cat, cat_mdl)
            category = override_category_by_keywords(neg_part, cat_raw)
            return "Mixed", None, category

    label, conf = predict_sentiment(text, tfidf_sent, sent_mdl)
    cat_raw     = predict_category(text, tfidf_cat, cat_mdl)
    category    = override_category_by_keywords(text, cat_raw)

    # ── Neutral-intent override ────────────────────────────────────────────────
    # Catches questions like "what internet service does the company provide?"
    # that the TF-IDF model mislabels as Negative/Positive at low confidence.
    if label in ("Negative", "Positive") and _is_informational(text):
        if conf is None or conf < NEUTRAL_CONFIDENCE_THRESHOLD:
            label = "Neutral"

    return label, conf, category

# ── RAG helpers ────────────────────────────────────────────────────────────────
def get_embedding(text: str, client) -> list[float] | None:
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
        emb = resp.data[0].embedding
        if isinstance(emb, tuple):
            emb = emb[0]
        return [float(x) for x in emb]
    except Exception:
        return None
    
# def get_embedding(text: str, client) -> tuple[list[float] | None, str | None]:
#     """Returns (embedding, error_message). error_message is None on success."""
#     try:
#         resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
#         return resp.data[0].embedding, None
#     except Exception as e:
#         return None, str(e)

# def retrieve(text: str, category: str = "", top_k: int = 3) -> tuple[list[dict], str | None]:
#     """Query ChromaDB and return (chunks, error_message).
#     error_message is None on success; non-None explains exactly what failed.
#     """
#     collection = load_chroma()
#     ollama     = load_ollama_client()

#     if ollama is None:
#         return [], "Ollama client could not be created — is Ollama running on port 11434?"
#     if collection is None:
#         return [], "ChromaDB collection could not be opened — check the CHROMA_DIR path."

#     # Check the collection actually has documents
#     try:
#         count = collection.count()
#         if count == 0:
#             return [], "ChromaDB collection is empty — re-run your rag_1.ipynb to populate it."
#     except Exception as e:
#         return [], f"ChromaDB count failed: {e}"

#     embedding, emb_err = get_embedding(text, ollama)
#     if embedding is None:
#         return [], (
#             f"Embedding failed — make sure the '{EMBED_MODEL}' model is pulled in Ollama "
#             f"(`ollama pull {EMBED_MODEL}`). Error: {emb_err}"
#         )

#     try:
#         results   = collection.query(query_embeddings=[embedding], n_results=top_k)
#         chunks    = []
#         docs      = results.get("documents", [[]])[0]
#         ids       = results.get("ids",       [[]])[0]
#         distances = results.get("distances", [[]])[0]
#         metas     = results.get("metadatas", [[]])[0]
#         for doc, cid, dist, meta in zip(docs, ids, distances, metas):
#             chunks.append({
#                 "text":     doc,
#                 "source":   meta.get("source", "policy_doc"),
#                 "page":     meta.get("page", "—"),
#                 "distance": round(dist, 4),
#             })
#         return chunks, None
#     except Exception as e:
#         return [], f"ChromaDB query failed: {e}"
def retrieve(text: str, category: str = "", top_k: int = 3) -> tuple[list[dict], str | None]:
    collection = load_chroma()
    ollama = load_ollama_client()
    if collection is None:
        return [], "ChromaDB collection unavailable."
    if ollama is None:
        return [], "Ollama client unavailable."
    if collection.count() == 0:
        return [], "ChromaDB collection has 0 chunks."

    try:
        raw_emb = get_embedding(text, ollama)
        if isinstance(raw_emb, tuple):
            raw_emb = raw_emb[0]
        if not raw_emb:
            return [], "Failed to generate query embedding."

        query_vec = [float(x) for x in raw_emb]
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, collection.count()),
        )
    except Exception as e:
        return [], str(e)

    docs = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    chunks = []
    for doc, cid, dist, meta in zip(docs, ids, distances, metas):
        meta = meta or {}
        chunks.append({
            "text": doc,
            "source": meta.get("source", "policy_doc"),
            "page": meta.get("page", "—"),
            "distance": round(float(dist), 4),
        })
    return chunks, None

def build_reply(text: str, sentiment: str, category: str, chunks: list[dict]) -> str:
    """Template-based reply (no LLM, instant)."""
    if not chunks:
        policy_note = "Our team will follow up based on our standard service policies."
    else:
        policy_note = (
            f"Based on our policy documents, here is what we found: "
            f"{chunks[0]['text'][:300].strip()}…"
        )

    greetings = {
        "Negative": "We sincerely apologise for the inconvenience you've experienced.",
        "Positive": "Thank you for your kind words — we're delighted to hear this!",
        "Neutral" : "Thank you for getting in touch with us.",
        "Mixed"   : "Thank you for your honest feedback. We're glad parts of your experience were positive.",
    }
    opening = greetings.get(sentiment, "Thank you for your feedback.")
    return (
        f"{opening} Regarding your {category} query: {policy_note} "
        "If you need further assistance, our support team is available 24/7."
    )

def build_reply_llm(text: str, sentiment: str, category: str, chunks: list[dict]) -> str:
    """Returns a 1-2 sentence policy-grounded answer without mixing unrelated issues."""
    if sentiment == "Positive" and not _is_informational(text):
        return (
            f"Thank you for your kind feedback! We're delighted to hear that you're satisfied "
            f"with our {category}. Please reach out anytime if you need assistance."
        )

    ollama = load_ollama_client()
    if ollama is None:
        return build_reply(text, sentiment, category, chunks)

    if not chunks:
        return "Thank you for reaching out. Our support team will get back to you shortly."

    # Use the top 2 closest chunks so split activation rules stay together
    top_chunks = sorted(chunks, key=lambda c: c.get("distance", 1.0))[:2]
    context = "\n\n".join(c["text"].strip() for c in top_chunks)

    prompt = f"""Policy excerpts:
{context}

Customer message: {text}

Extract ONLY the exact policy rule, timeframe, or troubleshooting step that matches the customer's issue in 1-2 sentences.
Ignore unrelated issues in the excerpts (such as payment failures or broadband when the customer asks about SIM/eSIM activation), and do not claim to check their account.""".strip()

    try:
        resp = ollama.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.0,
            max_tokens=90,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise telecom policy assistant. "
                        "State only the matching policy rule or timeframe from the excerpts in 1-2 short sentences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return build_reply(text, sentiment, category, chunks) + f"\n\n*(LLM unavailable: {e})*"


# def build_reply_llm(text: str, sentiment: str, category: str, chunks: list[dict]) -> str:
    """Returns a 1-2 sentence answer grounded strictly in the best retrieved chunk."""
    if sentiment == "Positive" and not _is_informational(text):
        return (
            f"Thank you for your kind feedback! We're delighted to hear that you're satisfied "
            f"with our {category}. Please reach out anytime if you need assistance."
        )
    elif sentiment == "Neutral" and not _is_informational(text):
            return (
                f"Thank you for getting in touch with us"
                f"with our {category}. Please reach out anytime if you need assistance."
            )
    ollama = load_ollama_client()
    if ollama is None:
        return build_reply(text, sentiment, category, chunks)

    if not chunks:
        return "Thank you for reaching out. Our support team will get back to you shortly."

    # Best chunk = lowest distance score
    best_chunk = min(chunks, key=lambda c: c.get("distance", 1.0))
    context    = best_chunk["text"].strip()

    prompt = f"""Policy:
{context}

Customer: {text}

Using the policy above, reply in 1-2 sentences. No lists. No bullet points.""".strip()

    try:
        resp = ollama.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.0,
            max_tokens=90,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a telecom support agent. "
                        "Read the policy, then answer the customer in 1-2 sentences. "
                        "Never say information is unavailable if the policy answers the question."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return build_reply(text, sentiment, category, chunks) + f"\n\n*(LLM unavailable: {e})*"


# ── Log & Data helpers ─────────────────────────────────────────────────────────
# def save_log(text: str, sentiment: str, category: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    row = pd.DataFrame([{
        "time":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feedback_text":    text,
        "service_category": category,
        "sentiment":        sentiment,
    }])
    row.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)
def save_log(text: str, sentiment: str, category: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    row = pd.DataFrame([{
        "feedback_text":    text,
        "service_category": category,
        "sentiment":        sentiment,
    }])
    row.to_csv(HIST_FILE, mode="a", header=not os.path.exists(HIST_FILE), index=False)

LOG_COLUMNS  = ["time", "feedback_text", "service_category", "sentiment"]
DATA_COLUMNS = ["feedback_text", "service_category", "sentiment"]

def _safe_read_csv(path: str, columns: list) -> pd.DataFrame:
    """Always returns a DataFrame with exactly `columns`.
    Survives: file missing, file empty (deleted contents),
    corrupted rows, or columns wiped from header.
    """
    empty = pd.DataFrame(columns=columns)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return empty
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return empty
    # Inject any missing columns so downstream code never KeyErrors
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[columns]

def load_live() -> pd.DataFrame:
    return _safe_read_csv(LOG_FILE, LOG_COLUMNS)

def load_combined_data() -> pd.DataFrame:
    history = _safe_read_csv(HIST_FILE, DATA_COLUMNS)
    live    = _safe_read_csv(LOG_FILE,  LOG_COLUMNS)
    return pd.concat([history, live[DATA_COLUMNS]], ignore_index=True)

# ── Recommendations ──────────────────────────────────────────────────────────
ISSUE_KEYWORDS = {
    "slow speed"      : ["slow", "speed", "lag", "latency"],
    "disconnection"   : ["disconnect", "drop", "unstable", "interrupt"],
    "billing error"   : ["bill", "charge", "overcharg", "invoice", "payment"],
    "no signal"       : ["signal", "coverage", "dead zone", "no network"],
    "poor support"    : ["support", "agent", "rude", "unhelpful", "wait"],
    "activation issue": ["activat", "sim", "setup", "configure"],
    "outage"          : ["outage", "down", "unavailabl"],
}

def detect_issues(texts: list[str]) -> list[str]:
    found = []
    combined = " ".join(texts).lower()
    for label, kws in ISSUE_KEYWORDS.items():
        if any(kw in combined for kw in kws):
            found.append(label)
    return found

def build_recommendations(df: pd.DataFrame, retrieve_fn=None) -> list[dict]:
    recs = []
    negative_df = df[df["sentiment"].isin(["Negative", "Mixed"])]
    for cat, group in negative_df.groupby("service_category"):
        total_cat = len(df[df["service_category"] == cat])
        neg_count = len(group)
        neg_share = round(neg_count / total_cat * 100, 1) if total_cat else 0
        if neg_count < 3:
            continue
        issues = detect_issues(group["feedback_text"].tolist())
        action = (
            f"Prioritise resolving {', '.join(issues[:2])} in {cat}."
            if issues else f"Review recent complaints in {cat} for recurring patterns."
        )
        policy = None
        if retrieve_fn:
            try:
                rec_chunks, _ = retrieve_fn(f"{cat} complaints: {' '.join(issues)}", category=cat, top_k=1)
                policy = rec_chunks[0] if rec_chunks else None
            except Exception:
                pass
        recs.append({
            "category":       cat,
            "negative_count": neg_count,
            "negative_share": neg_share,
            "issues":         issues,
            "action":         action,
            "policy":         policy,
        })
    recs.sort(key=lambda r: r["negative_count"], reverse=True)
    return recs

# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════
import base64

st.set_page_config(page_title="ZENDS Brand Intelligence", layout="wide")

image_path = r"D:\python_projects\zends_project\local\logo.png"


def set_background_logo(img_path: str, opacity: float = 0.08, size: str = "420px"):
    """Applies a subtle centered logo watermark behind the Streamlit app canvas."""
    if not os.path.exists(img_path):
        return

    ext = os.path.splitext(img_path)[1].lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext or 'png'}"

    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: transparent;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background-image: url("data:{mime};base64,{encoded}");
            background-repeat: no-repeat;
            background-position: center;
            background-size: {size};
            opacity: {opacity};
            pointer-events: none;
            z-index: 0;
        }}
        /* Keep main content and sidebar above the background watermark */
        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stSidebar"] {{
            position: relative;
            z-index: 1;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


set_background_logo(image_path, opacity=0.08, size="420px")

tfidf_sent = sentiment_model = tfidf_cat = category_model = None
models_ok = False

try:
    tfidf_sent, sentiment_model, tfidf_cat, category_model = load_tfidf_models()
    models_ok = True
except Exception as e:
    models_ok = False
    st.sidebar.error(f"Could not load local models: {e}\nCheck that models/*.pkl files exist.")

ollama_ok = load_ollama_client() is not None
_chroma_col = load_chroma()
chroma_ok = _chroma_col is not None
chroma_count = 0
if chroma_ok:
    try:
        chroma_count = _chroma_col.count()
    except Exception:
        pass
st.sidebar.markdown("### System status")
st.sidebar.markdown(f"{'✅' if models_ok  else '❌'} TF-IDF models")
st.sidebar.markdown(f"{'✅' if ollama_ok  else '❌'} Ollama (localhost:11434)")
chroma_label = f"{'✅' if chroma_ok and chroma_count > 0 else '⚠️' if chroma_ok else '❌'} ChromaDB"
chroma_detail = f"({chroma_count} chunks)" if chroma_ok else "(not found)"
st.sidebar.markdown(f"{chroma_label} {chroma_detail}")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Customer Feedback", "Brand Analyst Dashboard"],
)

# ═══════════════════════════════════════════════════════════════════════
# 0) HOME PAGE
# ═══════════════════════════════════════════════════════════════════════
if page == "Home":
    st.title("ZENDS Communications – Brand Intelligence")
    st.markdown(
        "Local customer feedback intelligence combining **TF-IDF + Logistic Regression** "
        "classifiers for sentiment and service category detection with **ChromaDB + Ollama** "
        "policy-grounded responses."
    )

    combined_df = load_combined_data()
    live_df = load_live()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Feedback Records", f"{len(combined_df):,}")
    m2.metric("Live Logged Submissions", f"{len(live_df):,}")
    m3.metric("Service Categories", f"{combined_df['service_category'].nunique()}")
    m4.metric("Models & RAG Status", "Ready" if (models_ok and chroma_ok) else "Check Sidebar")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        with st.container(border=True):
            st.subheader("💬 Customer Feedback")
            st.write(
                "Submit customer experience notes or complaints, detect sentiment and "
                "service category locally, retrieve matching policy clauses from ChromaDB, "
                "and generate instant or Ollama-assisted replies."
            )
            st.caption("Select **Customer Feedback** in the left sidebar to test or log feedback.")

    with col_right:
        with st.container(border=True):
            st.subheader("📊 Brand Analyst Dashboard")
            st.write(
                "Explore sentiment distributions, category-level complaint volume, "
                "prioritised service recommendations, and live 5-second feedback monitoring."
            )
            st.caption("Select **Brand Analyst Dashboard** in the left sidebar to inspect analytics.")

    st.subheader("Active Pipeline Architecture")
    st.markdown(
        "* **Sentiment & Category Models:** Separate TF-IDF (`1–2` n-grams, `5,000` features) + "
        "balanced Logistic Regression (`C=10.0`) models loaded from `models/`.\n"
        "* **Policy RAG Store:** Persisted ChromaDB (`mannual` collection) queried with "
        "`mxbai-embed-large` embeddings via local Ollama.\n"
        "* **Response & Monitoring:** Template or `llama3.2:1b` grounded replies with "
        "automatic logging to `data/feedback_log.csv`."
    )

# ═══════════════════════════════════════════════════════════════════════
# 1) CUSTOMER INTERFACE
# ═══════════════════════════════════════════════════════════════════════
elif page == "Customer Feedback":
    st.title("ZENDS Communications – Share your feedback")

    if not models_ok:
        st.error("Models not loaded. Please check the models/ folder.")
        st.stop()

    text    = st.text_area("Tell us about your experience or complaint", height=150)
    use_llm = st.checkbox(
        "Generate reply with Ollama LLM (requires Ollama running locally)"
    )

    if st.button("Submit") and text.strip():
        sentiment, confidence, category = analyse(
            text, tfidf_sent, sentiment_model, tfidf_cat, category_model
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Sentiment",        sentiment)
        c2.metric("Confidence",       f"{confidence * 100:.0f}%" if confidence is not None else "n/a")
        c3.metric("Service category", category)

        if sentiment == "Mixed":
            st.caption(
                "Mixed feedback detected: part of your message is positive and part is a complaint. "
                "The category shown is the one with the complaint."
            )

        chunks = []
        rag_error = None
        with st.spinner("Searching policy documents…"):
            chunks, rag_error = retrieve(text, category=category)
        if rag_error:
            st.warning(f"⚠️ Policy search issue: {rag_error}")

        with st.spinner("Generating response…"):
            try:
                if use_llm:
                    reply = build_reply_llm(text, sentiment, category, chunks)
                else:
                    reply = build_reply(text, sentiment, category, chunks)
            except Exception as e:
                st.warning(f"Reply generation failed, using template fallback: {e}")
                reply = build_reply(text, sentiment, category, chunks)

        st.subheader("Our response")
        safe_reply = reply.replace("$", r"\$")
        st.info(safe_reply)

        with st.expander("Policy documents used (RAG)"):
            if not chunks:
                st.write("No matching policy document found.")
            for ch in chunks:
                st.write(f"**{ch['source']}** (page {ch['page']}, distance {ch['distance']})")
                st.caption(ch["text"][:300])

        save_log(text, sentiment, category)
        st.success("Your feedback has been recorded. Thank you!")

# ═══════════════════════════════════════════════════════════════════════
# 2) BRAND ANALYST DASHBOARD
# ═══════════════════════════════════════════════════════════════════════
else:
    st.title("Brand Analyst Dashboard")

    try:
        history = pd.read_csv(HIST_FILE)
    except FileNotFoundError:
        history = pd.DataFrame(columns=["feedback_text", "service_category", "sentiment"])
        st.warning(f"Historical data not found at {HIST_FILE}. Showing live data only.")

    live     = load_live()
    all_data = pd.concat(
        [history, live[["feedback_text", "service_category", "sentiment"]]],
        ignore_index=True
    )

    cats   = sorted(all_data["service_category"].dropna().unique())
    chosen = st.multiselect("Filter service category", cats, default=cats)
    df     = all_data[all_data["service_category"].isin(chosen)]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total feedback", len(df))
    k2.metric("Negative", f"{(df['sentiment'] == 'Negative').mean() * 100:.1f}%" if len(df) else "0.0%")
    k3.metric("Neutral",  f"{(df['sentiment'] == 'Neutral' ).mean() * 100:.1f}%" if len(df) else "0.0%")
    k4.metric("Positive", f"{(df['sentiment'] == 'Positive').mean() * 100:.1f}%" if len(df) else "0.0%")
    k5.metric("Mixed",    f"{(df['sentiment'] == 'Mixed'   ).mean() * 100:.1f}%" if len(df) else "0.0%")

    left, right = st.columns(2)

    left.subheader("Sentiment distribution")
    left.bar_chart(df["sentiment"].value_counts())

    right.subheader("Topic-wise complaint frequency (Negative + Mixed)")
    complaint_df = df[df["sentiment"].isin(["Negative", "Mixed"])]
    right.bar_chart(complaint_df["service_category"].value_counts())

    st.subheader("Service category sentiment breakdown")
    breakdown = pd.crosstab(df["service_category"], df["sentiment"])
    st.bar_chart(breakdown)
    st.dataframe(breakdown)

    st.subheader("Recommendations for improving service quality")
    # Wrap retrieve so build_recommendations gets (chunks, _) correctly via its own unpack
    retrieve_fn = retrieve if (chroma_ok and chroma_count > 0 and ollama_ok) else None
    recs        = build_recommendations(df, retrieve_fn=retrieve_fn)

    if not recs:
        st.write("Not enough negative feedback to analyse yet.")
    for r in recs:
        with st.container(border=True):
            st.markdown(
                f"**{r['category']}** – {r['negative_count']} negative complaints "
                f"({r['negative_share']}% of this category)"
            )
            if r["issues"]:
                st.write("Recurring issues: " + ", ".join(r["issues"]))
            st.write("Recommended action: " + r["action"])
            if r["policy"]:
                st.caption(
                    f"Policy reference ({r['policy']['source']}, p{r['policy']['page']}): "
                    + re.sub(r"^\[.*?\] ", "", r["policy"]["text"])[:250]
                )

    if recs and st.checkbox("Ask Ollama LLM to summarise these recommendations"):
        if not ollama_ok:
            st.warning("Ollama is not running. Start it with `ollama serve`.")
        else:
            ollama = load_ollama_client()
            facts  = "; ".join(
                f"{r['category']} has {r['negative_count']} negative complaints about "
                f"{', '.join(r['issues']) or 'various issues'}"
                for r in recs
            )
            prompt = f"Write 3 short, actionable recommendations for a telecom service team. Facts: {facts}."
            with st.spinner("Asking Ollama…"):
                try:
                    resp = ollama.chat.completions.create(
                        model=LLM_MODEL,
                        temperature=0.3,
                        max_tokens=200,
                        messages=[
                            {"role": "system", "content": "You are a concise telecom service improvement advisor."},
                            {"role": "user",   "content": prompt},
                        ],
                    )
                    st.info(resp.choices[0].message.content.strip())
                except Exception as e:
                    st.warning(f"LLM summarisation failed: {e}")

    st.subheader("Real-time monitoring of new feedback")

    @st.fragment(run_every=5)
    def live_monitor():
        live_now = load_live()
        if live_now.empty:
            st.write("No live feedback yet. Submit one from the Customer Feedback page.")
            return
        a, b = st.columns(2)
        a.metric("Live submissions", len(live_now))
        recent    = live_now.tail(10)
        neg_count = int((recent["sentiment"] == "Negative").sum())
        b.metric("Negative in last 10", neg_count)
        if neg_count >= 5:
            st.error("⚠️ Alert: High volume of negative messages in recent feedback. Check for a service issue.")
        st.dataframe(recent.iloc[::-1])
        st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")

    live_monitor()