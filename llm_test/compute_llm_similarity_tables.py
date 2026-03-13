#!/usr/bin/env python3
"""
Compute similarity metrics between `original_ans` and `llm_cond{1,2,3}` in CSV tables.

Outputs 3 new columns per row:
  - similarity_cond1
  - similarity_cond2
  - similarity_cond3

Each cell is a JSON object:
  {
    "score1": ...,
    "score2": {"entailment": ..., "contradiction": ...},
    "score3": ...,
    "score4": ...
  }

Score definitions
  score1: semantic similarity (embedding cosine)
  score2: stance/NLI scores (entailment and contradiction probabilities)
  score3: lexical similarity (char n-gram TF-IDF cosine or Jaccard)
  score4: length similarity = 1 - |len1-len2| / max(len1, len2)

Dependencies (recommended):
  pip install sentence-transformers transformers torch scikit-learn
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch


def _extract_llm_text(value) -> Optional[str]:
    """Best-effort extraction of plain text from nested/json-string cells."""
    if isinstance(value, dict):
        if isinstance(value.get("llm_text"), str):
            return value["llm_text"].strip()
        for key in ("answer", "response", "text", "content", "result", "data"):
            if key in value:
                extracted = _extract_llm_text(value[key])
                if extracted is not None:
                    return extracted
        return None

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return ""
        if s.startswith("{") or s.startswith("["):
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                return s
            extracted = _extract_llm_text(parsed)
            return extracted if extracted is not None else s
        return s

    if isinstance(value, list):
        for item in value:
            extracted = _extract_llm_text(item)
            if extracted is not None:
                return extracted
        return None

    return None


def normalize_cell_text(text: str) -> str:
    if text is None:
        return ""
    extracted = _extract_llm_text(text)
    return extracted if extracted is not None else str(text)


def clip_score(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def round_score(x: float) -> float:
    return round(float(x), 6)


def autodetect_device(preference: str = "auto") -> str:
    """Choose best available torch device for SentenceTransformer (cuda/mps/cpu)."""
    pref = (preference or "auto").lower()
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_hf_device(preference: str = "auto", device_id: Optional[int] = None):
    """
    Determine the device argument for transformers.pipeline that works across
    Windows/macOS/Linux. Returns an int (CUDA), torch.device(\"mps\"), or -1 for CPU.
    """
    if device_id is not None:
        return device_id

    pref = (preference or "auto").lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return 0
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return -1

    if pref == "cpu":
        return -1
    if pref.startswith("cuda"):
        try:
            return int(pref.split(":")[1]) if ":" in pref else 0
        except ValueError:
            return 0
    if pref == "mps":
        return torch.device("mps")
    return pref


def length_similarity(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    denom = max(la, lb)
    return 1.0 - abs(la - lb) / denom


def jaccard_char_ngrams(a: str, b: str, n: int = 3) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    def grams(s: str) -> set[str]:
        if len(s) < n:
            return {s}
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    ga, gb = grams(a), grams(b)
    union = ga | gb
    if not union:
        return 1.0
    return len(ga & gb) / len(union)


@dataclass
class SimilarityEvaluator:
    embedder: object
    nli_pipe: object
    lexical_mode: str = "tfidf"
    char_ngram_range: Tuple[int, int] = (3, 5)
    jaccard_n: int = 3

    def __post_init__(self) -> None:
        self._embedding_cache: Dict[str, object] = {}
        self._tfidf_vectorizer_cls = None
        self._cosine_similarity_fn = None
        if self.lexical_mode == "tfidf":
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
            except ImportError as e:
                raise ImportError(
                    "scikit-learn is required for lexical_mode='tfidf'. "
                    "Install with: pip install scikit-learn"
                ) from e
            self._tfidf_vectorizer_cls = TfidfVectorizer
            self._cosine_similarity_fn = cosine_similarity

    def _embedding(self, text: str):
        cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached
        emb = self.embedder.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self._embedding_cache[text] = emb
        return emb

    def semantic_score(self, a: str, b: str) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        emb_a = self._embedding(a)
        emb_b = self._embedding(b)
        # embeddings are normalized, so dot product == cosine
        return float((emb_a * emb_b).sum())

    def lexical_score(self, a: str, b: str) -> float:
        if self.lexical_mode == "jaccard":
            return jaccard_char_ngrams(a, b, n=self.jaccard_n)

        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0

        vectorizer = self._tfidf_vectorizer_cls(
            analyzer="char",
            ngram_range=self.char_ngram_range,
        )
        mat = vectorizer.fit_transform([a, b])
        sim = self._cosine_similarity_fn(mat[0], mat[1])[0, 0]
        return float(sim)

    @staticmethod
    def _normalize_nli_output(raw) -> List[Dict[str, float]]:
        # transformers pipeline output can be list[dict] or [list[dict]]
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            return raw[0]
        if isinstance(raw, list):
            return raw
        return [raw]

    @staticmethod
    def _label_key(label: str) -> str:
        l = (label or "").lower()
        if "entail" in l:
            return "entailment"
        if "contrad" in l:
            return "contradiction"
        if "neutral" in l:
            return "neutral"
        return l

    def nli_stance_scores(self, premise: str, hypothesis: str) -> Dict[str, float]:
        if not premise and not hypothesis:
            return {"entailment": 1.0, "contradiction": 0.0}
        if not premise or not hypothesis:
            return {"entailment": 0.0, "contradiction": 0.0}

        raw = self.nli_pipe({"text": premise, "text_pair": hypothesis})
        scores = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
        for item in self._normalize_nli_output(raw):
            key = self._label_key(str(item.get("label", "")))
            if key in scores:
                scores[key] = float(item.get("score", 0.0))

        return {
            "entailment": clip_score(scores["entailment"], 0.0, 1.0),
            "contradiction": clip_score(scores["contradiction"], 0.0, 1.0),
        }

    def pair_scores(self, a: str, b: str) -> Dict[str, object]:
        s1 = self.semantic_score(a, b)
        s2 = self.nli_stance_scores(a, b)
        s3 = self.lexical_score(a, b)
        s4 = length_similarity(a, b)
        return {
            "score1": round_score(clip_score(s1)),
            "score2": {
                "entailment": round_score(s2["entailment"]),
                "contradiction": round_score(s2["contradiction"]),
            },
            "score3": round_score(clip_score(s3, 0.0, 1.0)),
            "score4": round_score(clip_score(s4, 0.0, 1.0)),
        }


def iter_input_csvs(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.glob("*.csv")):
        if p.is_file():
            yield p


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(csv_path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_evaluator(args) -> SimilarityEvaluator:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required. Install with: "
            "pip install sentence-transformers"
        ) from e

    try:
        from transformers import pipeline
    except ImportError as e:
        raise ImportError("transformers is required. Install with: pip install transformers") from e

    embed_device = autodetect_device(args.device)
    hf_device = resolve_hf_device(args.hf_device, args.hf_device_id)

    embedder = SentenceTransformer(args.embedding_model, device=embed_device)
    nli_pipe = pipeline(
        "text-classification",
        model=args.nli_model,
        tokenizer=args.nli_model,
        top_k=None,
        device=hf_device,
    )

    return SimilarityEvaluator(
        embedder=embedder,
        nli_pipe=nli_pipe,
        lexical_mode=args.lexical_mode,
        char_ngram_range=(args.char_ngram_min, args.char_ngram_max),
        jaccard_n=args.jaccard_n,
    )


def process_one_csv(
    csv_path: Path, out_path: Path, evaluator: SimilarityEvaluator, args: argparse.Namespace
) -> None:
    rows = read_rows(csv_path)
    if not rows:
        write_rows(out_path, [], ["iter", "trial", "original_ans", "llm_cond1", "llm_cond2", "llm_cond3"])
        return

    required = {"original_ans", "llm_cond1", "llm_cond2", "llm_cond3"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

    # Backward compatibility:
    # - If old tables only have `index`, map it to `trial`.
    # - If `iter` is absent, fill with default iteration id.
    for row in rows:
        if "trial" not in row or not str(row.get("trial", "")).strip():
            row["trial"] = str(row.get("index", "")).strip()
        if not str(row.get("trial", "")).strip():
            raise ValueError(
                f"{csv_path} row missing both `trial` and `index`; cannot build stable row id"
            )
        if "iter" not in row or not str(row.get("iter", "")).strip():
            row["iter"] = str(args.iter_default)

    for row in rows:
        original = normalize_cell_text(row.get("original_ans", ""))
        for idx in (1, 2, 3):
            cond_col = f"llm_cond{idx}"
            out_col = f"similarity_cond{idx}"
            candidate = normalize_cell_text(row.get(cond_col, ""))
            row[out_col] = json.dumps(
                evaluator.pair_scores(original, candidate),
                ensure_ascii=False,
                separators=(",", ":"),
            )

    # Keep row columns, but force `iter` / `trial` to the front for downstream joins.
    existing_fields = list(rows[0].keys())
    fieldnames = []
    for base in ("iter", "trial"):
        if base in existing_fields and base not in fieldnames:
            fieldnames.append(base)
    for col in existing_fields:
        if col not in fieldnames:
            fieldnames.append(col)
    for idx in (1, 2, 3):
        col = f"similarity_cond{idx}"
        if col not in fieldnames:
            fieldnames.append(col)

    write_rows(out_path, rows, fieldnames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add similarity_cond1/2/3 columns to llm table CSVs."
    )
    parser.add_argument("--input-dir", default="llm_tables_out")
    parser.add_argument("--output-dir", default="llm_tables_similarity_out")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model for semantic similarity",
    )
    parser.add_argument(
        "--nli-model",
        default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        help="HF NLI model (text-classification pipeline) for stance consistency",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="SentenceTransformer device: auto (cuda/mps/cpu) or explicit value",
    )
    parser.add_argument(
        "--hf-device",
        default="auto",
        help="Transformers pipeline device: auto/cpu/cuda/mps. Ignored if --hf-device-id is set.",
    )
    parser.add_argument(
        "--hf-device-id",
        type=int,
        default=None,
        help="Optional transformers pipeline device id (-1=CPU, 0=CUDA:0). Overrides --hf-device when set.",
    )
    parser.add_argument(
        "--lexical-mode",
        choices=["tfidf", "jaccard"],
        default="tfidf",
        help="score3 computation method",
    )
    parser.add_argument("--char-ngram-min", type=int, default=3)
    parser.add_argument("--char-ngram-max", type=int, default=5)
    parser.add_argument("--jaccard-n", type=int, default=3)
    parser.add_argument(
        "--iter-default",
        type=int,
        default=1,
        help="Fallback iter value when input rows do not contain an `iter` column",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    evaluator = build_evaluator(args)
    csv_paths = list(iter_input_csvs(input_dir))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    for csv_path in csv_paths:
        out_path = output_dir / csv_path.name
        print(f"Processing {csv_path} -> {out_path}")
        process_one_csv(csv_path, out_path, evaluator, args)

    print(f"Done. Wrote {len(csv_paths)} files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
