"""
rag_retrieval.py — ChromaDB RAG retrieval for SepsisAI treatment queries
Adnexus Biotechnologies Inc | INVESTIGATIONAL — Not FDA Cleared

Collection: sepsisai_full_collection_2026 (~391,805 chunks)
Embeddings: NeuML/pubmedbert-base-embeddings
Reranking:  cross-encoder/ms-marco-MiniLM-L-6-v2 (optional, faster without CUDA)
"""

from __future__ import annotations
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Treatment RAG query banks (Cell 12 verbatim) ─────────────────────────────
TREATMENT_QUERIES: Dict[str, List[str]] = {
    "Urea": [
        "urea catabolism sepsis muscle wasting nitrogen balance ICU mortality",
        "hyperureaemia sepsis catabolic crisis negative nitrogen balance outcome",
        "urea independent predictor 90-day mortality sepsis Cox regression",
        "urea nitrogen wasting sepsis muscle proteolysis mechanism review",
        "continuous renal replacement therapy CRRT sepsis urea clearance AKI",
        "nutritional optimisation protein intake sepsis ICU catabolism reversal",
        "high protein nutrition sepsis ICU 1.5 g/kg nitrogen balance",
        "enteral parenteral nutrition sepsis catabolic state protein target",
        "ESPEN ICU nutrition guidelines protein sepsis critically ill 2019",
        "Heyland protein targets critically ill patients ICU nutrition",
        "mTOR resistance sepsis catabolism amino acid supplementation",
    ],
    "D-dimer": [
        "DIC disseminated intravascular coagulation sepsis anticoagulation management",
        "ISTH DIC score criteria platelets PT fibrinogen D-dimer diagnosis",
        "heparin anticoagulation sepsis DIC evidence review ICU",
        "fresh frozen plasma FFP coagulopathy sepsis treatment protocol",
        "recombinant thrombomodulin antithrombin sepsis DIC intervention",
    ],
    "Lactate": [
        "lactate clearance resuscitation bundle sepsis haemodynamic target",
        "vasopressor norepinephrine MAP target septic shock lactate",
        "fluid resuscitation haemodynamic optimisation sepsis lactate guided",
        "early goal directed therapy sepsis lactate clearance 10 percent",
    ],
    "PTX3": [
        "PTX3 endothelial protection vasopressor sepsis haemodynamic support",
        "endothelial barrier dysfunction therapeutic target sepsis ICU intervention",
        "early vasopressor initiation septic shock MAP 65 mmHg evidence",
    ],
    "SOFA": [
        "multi-organ failure sepsis organ support escalation ICU protocol",
        "SOFA score progression organ dysfunction reversal therapy sepsis",
    ],
    "IL-10": [
        "immunoparalysis IL-10 elevated sepsis immunostimulant therapy",
        "GM-CSF interferon gamma immunotherapy sepsis immunosuppression reversal",
        "corticosteroid avoidance immunoparalysis IL-10 sepsis management",
        "corticosteroid harm immunosuppressed sepsis IL-10 elevated",
    ],
    "sIL-1R2": [
        "anakinra IL-1 blockade NLRP3 inflammasome sepsis intervention review",
        "IL-1 receptor antagonist sepsis hyperferritinaemic inflammasome treatment",
        "NLRP3 inflammasome late phase sepsis monitoring Day 5 intervention",
    ],
    "IL-8": [
        "neutrophil directed therapy sepsis IL-8 CXCL8 intervention",
        "antibiotic source control neutrophil driven sepsis management",
    ],
    "IL-18": [
        "IL-18 NLRP3 inflammasome activation sepsis monitoring treatment",
        "anakinra tadekinig alfa IL-18 inflammasome blockade sepsis evidence",
        "inflammasome pathway sepsis catabolic state IL-18 intervention",
    ],
    "PaO2_FiO2": [
        "lung protective ventilation ARDS sepsis low tidal volume evidence",
        "prone positioning ARDS sepsis PaO2 FiO2 improvement mortality",
    ],
    "Creatinine": [
        "CRRT acute kidney injury sepsis renal replacement therapy indication KDIGO",
        "nephrotoxin avoidance acute kidney injury sepsis management review",
        "fluid balance renal protection sepsis AKI intervention evidence",
        "CRRT indication refractory fluid overload acidosis hyperkalaemia",
    ],
    "MAP": [
        "vasopressor MAP target 65 mmHg septic shock evidence review",
        "norepinephrine vasopressor titration septic shock haemodynamic",
    ],
    "CRP": [
        "CRP antibiotic guided monitoring sepsis serial measurement review",
    ],
    "IL-6": [
        "IL-6 sepsis marker downstream effector multi-organ failure monitoring",
    ],
    "IL-1ra": [
        "IL-1ra counter-regulation sepsis futile inflammation management",
    ],
    "TNF-alpha": [
        "TNF-alpha macrophage amplification sepsis anti-inflammatory source control",
    ],
    "GCS": [
        "septic encephalopathy GCS impairment neuroprotection management ICU",
    ],
    "Platelets": [
        "thrombocytopaenia sepsis monitoring coagulopathy platelet transfusion",
    ],
    "sIL-1R2": [
        "anakinra IL-1 blockade NLRP3 inflammasome sepsis intervention review",
        "NLRP3 inflammasome late phase sepsis monitoring Day 5 intervention",
    ],
}

SHARED_TREATMENT_QUERIES: List[str] = [
    "surviving sepsis campaign 2021 guidelines management ICU protocol",
    "sepsis bundle hour-1 lactate blood culture antibiotic vasopressor",
    "biomarker guided therapy sepsis personalised treatment precision medicine",
    "sepsis ICU management review evidence based intervention 2022 2023",
]


def build_treatment_rag_queries(bei: dict) -> List[str]:
    """Build ordered, deduplicated query list for a given BEI result."""
    queries: List[str] = []

    # Primary tipping dimension queries first
    tipping = bei.get("tipping_marker", "")
    if tipping in TREATMENT_QUERIES:
        queries.extend(TREATMENT_QUERIES[tipping])

    # Secondary failing dimensions (top 3 after primary)
    tc = bei.get("tipping_contributions", {})
    neg_tcs = sorted(
        [(m, v) for m, v in tc.items() if v < 0 and m != tipping],
        key=lambda x: x[1]
    )[:3]
    for marker, _ in neg_tcs:
        if marker in TREATMENT_QUERIES:
            queries.extend(TREATMENT_QUERIES[marker][:2])

    queries.extend(SHARED_TREATMENT_QUERIES)

    # MNAR and alarm-specific queries
    if bei.get("lactate_missing_warning"):
        queries.extend([
            "lactate measurement mandatory sepsis resuscitation MNAR bias",
            "lactate mandatory sepsis bundle haemodynamic monitoring",
        ])
    if bei.get("gcs_alarm"):
        queries.append("septic encephalopathy GCS impairment neuroprotection management ICU")

    # Deduplicate while preserving order
    seen: set = set()
    out: List[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


class RagRetriever:
    """
    Wrapper around a ChromaDB collection for BEI-oriented RAG retrieval.
    Stateless after construction — safe for concurrent API requests.
    """

    def __init__(self, chroma_path: str, collection_name: str, embed_model: str):
        self._chroma_path      = chroma_path
        self._collection_name  = collection_name
        self._embed_model      = embed_model
        self._collection       = None
        self._cross_encoder    = None
        self._ce_loaded        = False

    def load(self) -> None:
        """Initialise ChromaDB client and load collection."""
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        os.environ["ANONYMIZED_TELEMETRY"]     = "False"
        os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

        device = "cuda" if os.path.exists("/usr/local/cuda") else "cpu"
        embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=self._embed_model, device=device
        )
        client = chromadb.PersistentClient(path=self._chroma_path)
        self._collection = client.get_collection(
            name=self._collection_name, embedding_function=embed_fn
        )
        logger.info(
            f"ChromaDB loaded: {self._collection.count():,} chunks "
            f"[{self._collection_name}]"
        )

    @property
    def is_loaded(self) -> bool:
        return self._collection is not None

    @property
    def chunk_count(self) -> Optional[int]:
        if self._collection is None:
            return None
        return self._collection.count()

    def _load_cross_encoder(self) -> None:
        if self._ce_loaded:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self._ce_loaded = True
            logger.info("Cross-encoder loaded for reranking")
        except Exception as e:
            logger.warning(f"Cross-encoder unavailable (cosine ranking only): {e}")
            self._ce_loaded = True  # mark as attempted so we don't retry

    def retrieve(
        self,
        queries: List[str],
        top_k: int = 25,
        final_k: int = 10,
        min_year: int = 2015,
    ) -> List[dict]:
        """
        Retrieve and rerank literature chunks from ChromaDB.

        Returns list of dicts: {text, score, source, year, chunk_id}
        Sorted descending by cross-encoder score (or cosine if CE unavailable).
        """
        if self._collection is None:
            raise RuntimeError("RagRetriever.load() must be called before retrieve()")

        results: Dict[str, dict] = {}
        n_results = min(top_k, self._collection.count())

        for q in queries:
            try:
                res = self._collection.query(
                    query_texts=[q],
                    n_results=n_results,
                    where={"year": {"$gte": min_year}},  # int — required by ChromaDB
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                # Fallback without year filter (handles old string-keyed collections)
                try:
                    res = self._collection.query(
                        query_texts=[q],
                        n_results=n_results,
                        include=["documents", "metadatas", "distances"],
                    )
                except Exception as e:
                    logger.warning(f"RAG query failed: {q[:60]}... — {e}")
                    continue

            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            ):
                cid    = meta.get("chunk_id", doc[:40])
                cosine = max(0.0, 1.0 - dist)
                if cid not in results or cosine > results[cid]["score"]:
                    results[cid] = {
                        "text":     doc,
                        "score":    round(cosine, 4),
                        "source":   meta.get("file_name", "unknown"),
                        "year":     meta.get("year", ""),
                        "chunk_id": cid,
                    }

        ranked = sorted(results.values(), key=lambda x: x["score"], reverse=True)

        # Cross-encoder reranking
        self._load_cross_encoder()
        if self._cross_encoder is not None and len(ranked) > 0:
            try:
                rerank_q = (
                    f"sepsis treatment intervention {queries[0] if queries else 'sepsis'}"
                )
                scores = self._cross_encoder.predict(
                    [(rerank_q, c["text"]) for c in ranked[:40]]
                )
                for i, c in enumerate(ranked[:40]):
                    c["ce"] = float(scores[i])
                ranked = sorted(
                    ranked[:40],
                    key=lambda x: x.get("ce", x["score"]),
                    reverse=True,
                )
            except Exception as e:
                logger.warning(f"Cross-encoder reranking failed: {e}")

        return ranked[:final_k]

    def retrieve_for_bei(self, bei: dict, top_k: int = 25, final_k: int = 10) -> List[dict]:
        """Convenience wrapper: build queries from BEI result and retrieve."""
        queries = build_treatment_rag_queries(bei)
        logger.info(f"RAG: {len(queries)} queries → top_k={top_k} → final_k={final_k}")
        return self.retrieve(queries, top_k=top_k, final_k=final_k)


def format_chunks_for_claude(chunks: List[dict]) -> str:
    """Format retrieved chunks into numbered literature block for Claude prompt."""
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[{i}] ({c.get('year', '')}) "
            f"{c.get('source', '?').replace('.pdf', '')[:60]}\n"
            f"{c['text'][:500].strip()}"
        )
    return "\n\n---\n\n".join(lines)
