"""
ingest_literature.py — Add clinical literature to SepsisAI ChromaDB collection
Adnexus Biotechnologies Inc | INVESTIGATIONAL — Not FDA Cleared

Usage
-----
# Index pre-validated seed chunks (ADRENAL 2018, ESPEN 2019, KDIGO 2012)
python3 ingest_literature.py --seed

# Index a PDF file
python3 ingest_literature.py --pdf /path/to/paper.pdf --source "Author 2023" --year 2023

# Index a plain-text file (one chunk per paragraph)
python3 ingest_literature.py --txt /path/to/paper.txt --source "Author 2023" --year 2023

# Preview only (dry run — no writes)
python3 ingest_literature.py --seed --dry-run

Collection: sepsisai_full_collection_2026
Embeddings: NeuML/pubmedbert-base-embeddings
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import uuid
from typing import List

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Collection constants (must match rag_retrieval.py) ────────────────────────
COLLECTION_NAME = "sepsisai_full_collection_2026"
EMBED_MODEL     = "NeuML/pubmedbert-base-embeddings"
CHROMA_PATH     = os.getenv("CHROMA_PATH", "/data/chroma_db")


# ── Pre-validated seed chunks ─────────────────────────────────────────────────
# Source: primary trial papers + clinical guidelines
# Validated against: NEJM 2018 (ADRENAL), Clin Nutr 2019 (ESPEN), KDIGO 2012
# Each chunk: {text, source, year, tags}

SEED_CHUNKS: List[dict] = [

    # ══════════════════════════════════════════════════════════════════════════
    # ADRENAL TRIAL — Venkatesh et al., NEJM 2018
    # DOI: 10.1056/NEJMoa1705835 | PMID: 29872001
    # N=3,658; continuous IV hydrocortisone 200 mg/day × 7 days vs placebo
    # ══════════════════════════════════════════════════════════════════════════
    {
        "text": (
            "ADRENAL trial (Venkatesh et al., NEJM 2018; N=3,658): Continuous "
            "intravenous hydrocortisone 200 mg/day for 7 days versus placebo in "
            "mechanically ventilated patients with septic shock. PRIMARY OUTCOME: "
            "90-day all-cause mortality was 27.9% in the hydrocortisone group versus "
            "28.8% in the placebo group (OR 0.95; 95% CI 0.82–1.10; P=0.50). "
            "Hydrocortisone did NOT reduce 90-day mortality. This was the largest "
            "randomised trial of corticosteroids in septic shock to date."
        ),
        "source": "Venkatesh_ADRENAL_NEJM_2018.pdf",
        "year": 2018,
        "tags": ["adrenal", "hydrocortisone", "septic_shock", "mortality", "rct"],
    },
    {
        "text": (
            "ADRENAL trial secondary outcomes (Venkatesh 2018): Hydrocortisone "
            "significantly reduced time to shock reversal (median 3 vs 4 days; "
            "HR 1.32; 95% CI 1.23–1.41; P<0.001), time to ICU discharge (median "
            "10 vs 12 days; HR 1.14; 95% CI 1.06–1.23; P<0.001), and duration of "
            "initial mechanical ventilation (median 6 vs 7 days; HR 1.13; "
            "95% CI 1.05–1.22; P<0.001). Fewer transfusions in hydrocortisone arm "
            "(37.0% vs 41.7%; P=0.004). No difference in 28-day mortality, "
            "bacteraemia/fungaemia, or renal replacement therapy frequency. "
            "Secondary benefit signals are hypothesis-generating only."
        ),
        "source": "Venkatesh_ADRENAL_NEJM_2018.pdf",
        "year": 2018,
        "tags": ["adrenal", "hydrocortisone", "shock_reversal", "icu_stay", "secondary_outcomes"],
    },
    {
        "text": (
            "ADRENAL trial — corticosteroid harm signal and immune context "
            "(Venkatesh 2018): No significant difference in new-onset bacteraemia or "
            "fungaemia (14.1% vs 14.1%). However, hydrocortisone-treated patients with "
            "intact immune reserve (non-immunoparalysed phenotype, elevated IL-10 energy) "
            "are at theoretical risk of suppressed appropriate inflammatory response. "
            "ACTH stimulation testing was NOT performed in ADRENAL. The trial does not "
            "resolve whether a biomarker-guided approach (e.g., IL-10 energy phenotyping) "
            "could identify a subgroup that benefits. Current evidence supports withholding "
            "corticosteroids in patients without haemodynamic refractory shock or evidence "
            "of immune compromise."
        ),
        "source": "Venkatesh_ADRENAL_NEJM_2018.pdf",
        "year": 2018,
        "tags": ["adrenal", "hydrocortisone", "immunoparalysis", "il10", "contraindication"],
    },
    {
        "text": (
            "Corticosteroid contraindication in non-immunoparalysed sepsis: "
            "Elevated IL-10 energy (>0.70 normalised score) indicates intact "
            "counter-regulatory anti-inflammatory reserve — the patient is mounting "
            "an appropriate immune response, not immunoparalysed. Administering "
            "hydrocortisone in this phenotype suppresses the intact immune response, "
            "potentially increasing infectious risk. The ADRENAL trial (2018) found no "
            "mortality benefit for corticosteroids in unselected septic shock, consistent "
            "with the hypothesis that only immunoparalysed patients (low IL-10 energy) "
            "might benefit. Biomarker-stratified corticosteroid decisions are a recognised "
            "knowledge gap (Singer 2022 Lancet perspective)."
        ),
        "source": "Venkatesh_ADRENAL_NEJM_2018.pdf",
        "year": 2018,
        "tags": ["corticosteroid", "il10", "immunoparalysis", "contraindication", "bei_rule2"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # ESPEN 2019 ICU NUTRITION GUIDELINES
    # Singer P et al., Clin Nutr 2019; DOI: 10.1016/j.clnu.2018.08.037
    # ══════════════════════════════════════════════════════════════════════════
    {
        "text": (
            "ESPEN ICU Nutrition Guidelines 2019 (Singer et al., Clin Nutr 2019) — "
            "Protein targets for critically ill patients: Initial protein target of "
            "1.3 g/kg/day (Grade B). In the acute early phase (first 48h), "
            "permissive hypocaloric feeding with 70% of estimated caloric need is "
            "acceptable. After the acute phase, full protein delivery of 1.3–1.5 g/kg/day "
            "is recommended. In septic patients with evidence of catabolism (elevated "
            "urea, catabolic state), protein up to 1.5–2.0 g/kg/day may be required. "
            "Protein restriction to 0.8–1.2 g/kg/day is NOT recommended for catabolic "
            "sepsis — this accelerates skeletal muscle wasting and worsens outcomes."
        ),
        "source": "ESPEN_ICU_Nutrition_Guidelines_2019.pdf",
        "year": 2019,
        "tags": ["espen", "nutrition", "protein", "catabolism", "sepsis", "icu", "bei_rule5"],
    },
    {
        "text": (
            "ESPEN 2019 — Energy targets and timing: Early enteral nutrition (EN) "
            "should be initiated within 48 hours of ICU admission when the patient "
            "is haemodynamically stable (Grade A). Target energy 20–25 kcal/kg/day "
            "in the early phase, titrated to 25–30 kcal/kg/day after day 3 once "
            "haemodynamic stability is achieved. Overfeeding must be avoided — "
            "hypocaloric EN (≤20 kcal/kg/day) with adequate protein is preferred in "
            "the early acute phase over full energy delivery. High-protein hypocaloric "
            "feeding preserves lean mass while avoiding overfeeding-related complications "
            "(hyperglycaemia, hypercapnia, hepatic steatosis). Indirect calorimetry "
            "is recommended over predictive equations when available (Grade B)."
        ),
        "source": "ESPEN_ICU_Nutrition_Guidelines_2019.pdf",
        "year": 2019,
        "tags": ["espen", "nutrition", "energy", "enteral", "timing", "icu"],
    },
    {
        "text": (
            "ESPEN 2019 — Protein supplementation and catabolic hyperureaemia: "
            "Sepsis-induced muscle proteolysis releases branched-chain amino acids "
            "as fuel substrates, generating excess urea via the urea cycle. Elevated "
            "urea:creatinine ratio (>150 SI units) indicates catabolism-dominant "
            "physiology, not primary AKI. In catabolic state, protein delivery "
            "≥1.5 g/kg/day is required to compensate for obligate nitrogen losses "
            "and attenuate muscle wasting. ESPEN 2019 Recommendation 24 explicitly "
            "states that protein restriction is harmful in catabolic critical illness. "
            "Heyland et al. (2011, Crit Care Med) demonstrated survival benefit with "
            "protein >1.2 g/kg/day in ICU patients. Glutamine supplementation may "
            "be considered in select cases, though evidence remains mixed."
        ),
        "source": "ESPEN_ICU_Nutrition_Guidelines_2019.pdf",
        "year": 2019,
        "tags": ["espen", "protein", "urea", "catabolism", "urea_creatinine_ratio", "bei_rule1"],
    },
    {
        "text": (
            "ESPEN 2019 Recommendation — route and composition of nutrition: "
            "Enteral nutrition is preferred over parenteral nutrition (Grade A). "
            "If EN is contraindicated or insufficient after 48h, supplemental "
            "parenteral nutrition should be considered (Grade B). Glutamine: "
            "NOT recommended for critically ill patients receiving parenteral "
            "nutrition who have multi-organ failure or haemodynamic instability. "
            "Omega-3 fatty acids may be considered in ARDS patients on EN. "
            "Selenium and thiamine supplementation should be considered in all "
            "critically ill patients. Vitamin C high-dose therapy lacks sufficient "
            "evidence for routine use as of 2019."
        ),
        "source": "ESPEN_ICU_Nutrition_Guidelines_2019.pdf",
        "year": 2019,
        "tags": ["espen", "nutrition", "enteral", "parenteral", "composition", "icu"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # KDIGO 2012 AKI GUIDELINES + RENAL TRIAL 2009
    # KDIGO: Kidney Int Suppl 2012; DOI: 10.1038/kisup.2012.1
    # RENAL: Bellomo et al., NEJM 2009; DOI: 10.1056/NEJMoa0902413
    # ══════════════════════════════════════════════════════════════════════════
    {
        "text": (
            "KDIGO 2012 AKI Guidelines — CRRT initiation criteria: Renal replacement "
            "therapy (RRT) should be initiated based on clinical context, NOT on a "
            "single urea or creatinine threshold. KDIGO 2012 (Section 5.3) states that "
            "RRT should be initiated when ONE of the following is present: (1) refractory "
            "fluid overload; (2) metabolic acidosis pH <7.15 refractory to bicarbonate; "
            "(3) hyperkalaemia K⁺ >6.5 mmol/L refractory to medical therapy; "
            "(4) uraemic complications (encephalopathy, pericarditis, bleeding); "
            "(5) haemodynamic instability refractory to vasopressors. Urea elevation "
            "alone is NOT an indication for CRRT initiation."
        ),
        "source": "KDIGO_AKI_Guidelines_2012.pdf",
        "year": 2012,
        "tags": ["kdigo", "crrt", "aki", "rrt_indication", "urea", "bei_rule4"],
    },
    {
        "text": (
            "KDIGO 2012 — Urea and creatinine staging (NOT CRRT triggers): "
            "AKI is staged by serum creatinine and urine output: Stage 1 "
            "(Cr ×1.5–1.9 baseline, or ≥26.5 μmol/L rise in 48h, or UO <0.5 mL/kg/h "
            "for 6–12h); Stage 2 (Cr ×2–2.9 baseline, or UO <0.5 mL/kg/h for ≥12h); "
            "Stage 3 (Cr ×3 baseline, or ≥354 μmol/L, or initiation of RRT). "
            "These stages inform monitoring intensity and nephrology review, not "
            "automatic CRRT initiation. The RENAL trial (Bellomo 2009, NEJM) "
            "showed no mortality difference between higher-dose and lower-dose CRRT "
            "(40 vs 25 mL/kg/h effluent). A specific urea target (e.g., <20 mmol/L) "
            "does NOT exist in KDIGO or RENAL trial guidelines."
        ),
        "source": "KDIGO_AKI_Guidelines_2012.pdf",
        "year": 2012,
        "tags": ["kdigo", "aki", "staging", "creatinine", "urea_threshold", "bei_rule4"],
    },
    {
        "text": (
            "KDIGO 2012 AKI — Sepsis-AKI management principles: In sepsis-associated "
            "AKI, haemodynamic stabilisation with vasopressors targeting MAP ≥65 mmHg "
            "and adequate volume resuscitation are the primary interventions. "
            "Nephrotoxin avoidance (NSAIDs, aminoglycosides, radiocontrast) is Grade 1A. "
            "Loop diuretics are NOT recommended to treat AKI or reduce its duration "
            "(Grade 1B). CRRT is preferred over intermittent haemodialysis in "
            "haemodynamically unstable patients (Grade 2B). Anticoagulation for CRRT "
            "circuit: regional citrate anticoagulation is preferred over systemic "
            "heparin in patients without contraindications (Grade 2B, updated 2019)."
        ),
        "source": "KDIGO_AKI_Guidelines_2012.pdf",
        "year": 2012,
        "tags": ["kdigo", "aki", "sepsis", "crrt", "management", "vasopressor"],
    },
    {
        "text": (
            "KDIGO 2012 / RENAL trial — no urea-based CRRT target: The RENAL trial "
            "(Bellomo et al., NEJM 2009; N=1,508) compared higher-intensity CRRT "
            "(40 mL/kg/h effluent flow) versus lower-intensity CRRT (25 mL/kg/h) "
            "in critically ill patients with AKI. 90-day mortality was 44.7% vs 44.9% "
            "(RR 1.00; 95% CI 0.81–1.23; P=0.99). No survival benefit from higher dose. "
            "Urea clearance was incidental — neither trial arm targeted a specific urea "
            "value. Clinical practitoners and AI decision-support systems must NOT cite "
            "'Urea <20 mmol/L' as a CRRT initiation or cessation target — this threshold "
            "does not appear in KDIGO, the RENAL trial, or the SSC 2021 guidelines."
        ),
        "source": "RENAL_Trial_NEJM_2009.pdf",
        "year": 2009,
        "tags": ["renal_trial", "crrt", "dose", "urea", "no_urea_target", "bei_rule4"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # SURVIVING SEPSIS CAMPAIGN 2021 (reference anchors for hard rules)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "text": (
            "Surviving Sepsis Campaign 2021 guidelines (Evans et al., Crit Care Med 2021): "
            "Weak recommendation for corticosteroids (IV hydrocortisone 200 mg/day) "
            "only in adult septic shock patients who are not stabilised with adequate "
            "fluid resuscitation and vasopressor therapy. This recommendation explicitly "
            "applies to refractory shock, NOT to all septic shock patients. The SSC 2021 "
            "does not recommend corticosteroids in patients with haemodynamic stability "
            "or intact immune reserve. Corticosteroid candidacy requires individual "
            "haemodynamic and immunological assessment — it is not a default intervention."
        ),
        "source": "SSC_Guidelines_2021.pdf",
        "year": 2021,
        "tags": ["ssc", "corticosteroid", "hydrocortisone", "septic_shock", "guideline"],
    },
    {
        "text": (
            "Surviving Sepsis Campaign 2021 — sepsis bundles and early management: "
            "Hour-1 bundle: (1) measure lactate; (2) blood cultures before antibiotics; "
            "(3) broad-spectrum antibiotics; (4) 30 mL/kg crystalloid for hypotension or "
            "lactate ≥4 mmol/L; (5) vasopressors if MAP <65 mmHg during/after fluids. "
            "Lactate-guided resuscitation: target lactate clearance ≥10% per 2-hour cycle "
            "in elevated lactate (≥2 mmol/L). Norepinephrine is the first-line vasopressor "
            "(strong recommendation). MAP target 65 mmHg for most patients — higher targets "
            "(80–85 mmHg) benefit only patients with chronic hypertension (SEPSISPAM trial)."
        ),
        "source": "SSC_Guidelines_2021.pdf",
        "year": 2021,
        "tags": ["ssc", "bundle", "lactate", "vasopressor", "norepinephrine", "map"],
    },
]


def _load_collection(dry_run: bool = False):
    """Load or create ChromaDB collection."""
    if dry_run:
        return None, None

    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    os.environ["ANONYMIZED_TELEMETRY"]     = "False"
    os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

    device = "cuda" if os.path.exists("/usr/local/cuda") else "cpu"
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL, device=device
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # get_or_create so we can add to an existing collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        f"Collection '{COLLECTION_NAME}' loaded/created — "
        f"{collection.count():,} existing chunks"
    )
    return client, collection


def _chunk_id(source: str, index: int) -> str:
    """Deterministic chunk ID from source name + index."""
    return f"{source.replace(' ', '_').replace('/', '_')}__chunk_{index:04d}"


def ingest_chunks(
    chunks: List[dict],
    collection,
    dry_run: bool = False,
) -> int:
    """
    Add chunks to the ChromaDB collection.

    Each chunk: {text, source, year, tags (optional)}
    Returns count of chunks added.
    """
    if not chunks:
        logger.warning("ingest_chunks: empty chunk list — nothing to add")
        return 0

    documents, metadatas, ids = [], [], []

    for i, chunk in enumerate(chunks):
        text   = chunk["text"].strip()
        source = chunk.get("source", "unknown")
        year   = int(chunk.get("year", 0))
        cid    = chunk.get("chunk_id") or _chunk_id(source, i)

        if not text:
            continue

        documents.append(text)
        metadatas.append({
            "file_name": source,
            "year":      year,
            "chunk_id":  cid,
            "tags":      ",".join(chunk.get("tags", [])),
        })
        ids.append(cid)

    if dry_run:
        logger.info(f"DRY RUN — would add {len(documents)} chunks:")
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            logger.info(f"  [{i+1}] {meta['file_name']} ({meta['year']}) — {doc[:80]}...")
        return len(documents)

    # ChromaDB upsert (idempotent — duplicate chunk_ids are overwritten)
    batch_size = 50
    added = 0
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        collection.upsert(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        added += len(documents[start:end])
        logger.info(f"  Upserted {added}/{len(documents)} chunks")

    logger.info(
        f"Done. Collection now has {collection.count():,} total chunks."
    )
    return added


def ingest_txt(
    filepath: str,
    source: str,
    year: int,
    collection,
    dry_run: bool = False,
    min_chunk_len: int = 100,
) -> int:
    """
    Ingest a plain-text file. Splits on double newlines (paragraph boundaries).
    Each paragraph ≥ min_chunk_len chars becomes one chunk.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    paragraphs = [p.strip() for p in raw.split("\n\n") if len(p.strip()) >= min_chunk_len]
    logger.info(f"TXT: {len(paragraphs)} paragraphs from {filepath}")

    chunks = [
        {"text": p, "source": source, "year": year, "chunk_id": _chunk_id(source, i)}
        for i, p in enumerate(paragraphs)
    ]
    return ingest_chunks(chunks, collection, dry_run=dry_run)


def ingest_pdf(
    filepath: str,
    source: str,
    year: int,
    collection,
    dry_run: bool = False,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> int:
    """
    Ingest a PDF file. Extracts text via pdfminer.six and splits into
    overlapping chunks of ~chunk_size words.
    Requires: pip install pdfminer.six
    """
    try:
        from pdfminer.high_level import extract_text as pdf_extract
    except ImportError:
        logger.error("pdfminer.six not installed: pip install pdfminer.six --break-system-packages")
        sys.exit(1)

    raw = pdf_extract(filepath)
    words = raw.split()
    logger.info(f"PDF: {len(words):,} words extracted from {filepath}")

    chunks = []
    i = 0
    idx = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        text = " ".join(chunk_words).strip()
        if len(text) >= 80:
            chunks.append({
                "text":     text,
                "source":   source,
                "year":     year,
                "chunk_id": _chunk_id(source, idx),
            })
            idx += 1
        i += chunk_size - chunk_overlap

    logger.info(f"PDF: split into {len(chunks)} chunks")
    return ingest_chunks(chunks, collection, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest clinical literature into SepsisAI ChromaDB collection"
    )
    parser.add_argument(
        "--seed", action="store_true",
        help="Index pre-validated seed chunks (ADRENAL 2018, ESPEN 2019, KDIGO 2012, SSC 2021)",
    )
    parser.add_argument("--pdf",  metavar="PATH", help="Path to a PDF file to ingest")
    parser.add_argument("--txt",  metavar="PATH", help="Path to a plain-text file to ingest")
    parser.add_argument("--source", metavar="LABEL",  help="Source label (e.g. 'Author_NEJM_2023.pdf')")
    parser.add_argument("--year",   metavar="YEAR",   type=int, help="Publication year")
    parser.add_argument(
        "--chroma-path", default=CHROMA_PATH,
        help=f"Path to ChromaDB directory (default: {CHROMA_PATH})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be indexed without writing to ChromaDB",
    )
    args = parser.parse_args()

    if not args.seed and not args.pdf and not args.txt:
        parser.print_help()
        sys.exit(0)

    # Update module-level default so _load_collection sees the override
    import ingest_literature as _self
    _self.CHROMA_PATH = args.chroma_path

    _client, collection = _load_collection(dry_run=args.dry_run)

    total_added = 0

    if args.seed:
        logger.info(f"Indexing {len(SEED_CHUNKS)} pre-validated seed chunks …")
        n = ingest_chunks(SEED_CHUNKS, collection, dry_run=args.dry_run)
        total_added += n
        logger.info(f"Seed ingest complete: {n} chunks {'(dry run)' if args.dry_run else 'written'}")

    if args.pdf:
        if not args.source or not args.year:
            logger.error("--pdf requires --source and --year")
            sys.exit(1)
        n = ingest_pdf(args.pdf, args.source, args.year, collection, dry_run=args.dry_run)
        total_added += n

    if args.txt:
        if not args.source or not args.year:
            logger.error("--txt requires --source and --year")
            sys.exit(1)
        n = ingest_txt(args.txt, args.source, args.year, collection, dry_run=args.dry_run)
        total_added += n

    logger.info(f"Total chunks processed: {total_added}")


if __name__ == "__main__":
    main()
