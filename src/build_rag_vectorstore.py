# rag_vectorstore.py
from __future__ import annotations
import os, json
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

# uv pip install sentence-transformers faiss-gpu
from sentence_transformers import SentenceTransformer
import faiss

from utils import row_to_json



def df_to_docs(df: pd.DataFrame, cols: Optional[List[str]] = None,
               id_col: Optional[str] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns:
      texts: list[str] (one per row)
      metadatas: list[dict] (per-row metadata: id, row_index, optionally id_col)
    """
    texts, metas = [], []
    for i, row in df[cols].iterrows():
        text = row_to_json(row)
        meta = {"_row_index": int(i)}
        if id_col and id_col in df.columns:
            meta["_id"] = df.at[i, id_col]
        texts.append(text)
        metas.append(meta)
    return texts, metas


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds a simple L2 FAISS index (you can switch to cosine by normalizing).
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype(np.float32))
    return index


def embed_texts(texts: List[str],
                model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                batch_size: int = 128,
                normalize: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Encode texts to embeddings. Returns (embeddings, model_info)
    """
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=normalize)
    info = {"model_name": model_name, "normalized": normalize}
    return embs, info


def save_vectorstore(out_dir: str,
                     index: faiss.Index,
                     texts: List[str],
                     metadatas: List[Dict[str, Any]],
                     model_info: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    with open(os.path.join(out_dir, "texts.jsonl"), "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
        for m in metadatas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "model.json"), "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)


def load_vectorstore(vs_dir: str) -> Tuple[faiss.Index, List[str], List[Dict[str, Any]], Dict[str, Any]]:
    index = faiss.read_index(os.path.join(vs_dir, "index.faiss"))
    texts = [json.loads(l)["text"] for l in open(os.path.join(vs_dir, "texts.jsonl"), encoding="utf-8")]
    metas = [json.loads(l) for l in open(os.path.join(vs_dir, "metadata.jsonl"), encoding="utf-8")]
    model_info = json.load(open(os.path.join(vs_dir, "model.json"), encoding="utf-8"))
    return index, texts, metas, model_info


# def search(query: str,
#            vs_dir: str,
#            top_k: int = 5) -> List[Dict[str, Any]]:
#     """
#     Query the vectorstore with a text query. Returns list of {text, metadata, score}
#     """
#     index, texts, metas, model_info = load_vectorstore(vs_dir)
#     model = SentenceTransformer(model_info["model_name"])
#     q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=model_info.get("normalized", True))
#     D, I = index.search(q_emb.astype(np.float32), top_k)
#     results = []
#     for dist, idx in zip(D[0].tolist(), I[0].tolist()):
#         if idx == -1:  # FAISS may return -1 if not enough vectors
#             continue
#         results.append({"text": texts[idx], "metadata": metas[idx], "score": float(dist)})
#     return results


def build_vector_db_from_df(df: pd.DataFrame,
                            out_dir: str,
                            cols: Optional[List[str]] = None,
                            id_col: Optional[str] = None,
                            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                            normalize: bool = True,
                            batch_size: int = 128) -> str:
    """
    One-shot convenience function:
      1) Convert rows -> texts + metadata
      2) Embed
      3) Build FAISS
      4) Save artifacts
    Returns the directory path containing the vectorstore.
    """
    texts, metas = df_to_docs(df, cols=cols, id_col=id_col)
    embs, info = embed_texts(texts, model_name=model_name, batch_size=batch_size, normalize=normalize)
    index = build_faiss_index(embs)
    save_vectorstore(out_dir, index, texts, metas, info)
    return out_dir
