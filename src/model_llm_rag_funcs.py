import faiss
import os, json
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from utils import get_token_length

def load_vectorstore_for_rag(vs_dir: str) -> Tuple[faiss.Index, List[str], List[Dict[str, Any]], Dict[str, Any]]:
    idx = faiss.read_index(os.path.join(vs_dir, "index.faiss"))
    texts = [json.loads(l)["text"] for l in open(os.path.join(vs_dir, "texts.jsonl"), encoding="utf-8")]
    metas = [json.loads(l) for l in open(os.path.join(vs_dir, "metadata.jsonl"), encoding="utf-8")]
    model_info = json.load(open(os.path.join(vs_dir, "model.json"), encoding="utf-8"))
    return idx, texts, metas, model_info

# Sentence-Transformers embedder (MiniLM-L6)
_ST_CACHE = {"name": None, "model": None}

def _get_st_model(name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    if _ST_CACHE["model"] is None or _ST_CACHE["name"] != name:
        _ST_CACHE["model"] = SentenceTransformer(name)
        _ST_CACHE["name"] = name
    return _ST_CACHE["model"]

def embed_query_st(text: str, normalize: bool = True,
                   model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    model = _get_st_model(model_name)
    q = model.encode([text], convert_to_numpy=True)  # shape (1, dim)
    if normalize:
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-10)
    return q.astype(np.float32)

# Budget tokens
def truncate_by_token_budget(chunks: List[str], max_tokens: int) -> str:
    out, used = [], 0
    for ch in chunks:
        t = get_token_length(ch) 

        if used == 0 and t > max_tokens:
            # Include at least the first hit even if it exceeds token limits
            out.append(ch)
            used += t
            # Continue to see if we can fit more chunks
            continue
        elif used + t > max_tokens:
            break
        else:
            out.append(ch)
            used += t
    return "\n\n".join(out)


def extract_viral_diagnosis(block):
    if isinstance(block, str):
        try:
            data = json.loads(block)
            return data.get("viral_diagnosis", None)
        except Exception:
            return None
    elif isinstance(block, list):
        results = []
        for item in block:
            try:
                data = json.loads(item)
                results.append(data.get("viral_diagnosis", None))
            except Exception:
                results.append(None)
        return results
    else:
        return None

# Retrieval using the MiniLM query embedding
def retrieve_context(row_text: str, rag_state: Dict[str, Any],
                     top_k: int, max_ctx_tokens: int) -> str:
    """
    Returns a concatenated context string retrieved from a FAISS index
    built with all-MiniLM-L6-v2 embeddings (ideally L2-normalized).
    """
    index = rag_state["index"]
    texts = rag_state["texts"]
    model_info = rag_state.get("model_info", {})
    # Respect whether stored vectors were normalized; default True for cosine search
    normalized = model_info.get("normalized", True)

    q_emb = embed_query_st(
        row_text,
        normalize=normalized,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    D, I = index.search(q_emb, top_k)
    hits = [texts[i] for i in I[0] if i != -1]

    body = hits
    # Uncomment to truncate returned hits based on token length
    # body = truncate_by_token_budget(hits, max_ctx_tokens)

    # Only return viral diagnosis
    body = extract_viral_diagnosis(body)

    if not body:
        return ""
    return body
