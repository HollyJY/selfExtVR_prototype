#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib import error, request


# Allow overriding the backend endpoint via env for cross-platform setups (Windows/macOS/Linux).
API_URL = os.getenv("LLM_API_URL", "http://192.168.37.177:7002/api/v1/llm")

STANCE_SPECS = [
    (
        "direct_commitment",
        "Direct commitment: choose one side (A or B) clearly and commit to it.",
    ),
    (
        "conditional_commitment",
        "Conditional commitment: answer with clear conditions, trade-offs, or an 'it depends' structure.",
    ),
    (
        "non_commitment",
        "Non-commitment: avoid taking a firm side (e.g., 'I don't know' or both options are acceptable).",
    ),
    (
        "frame_rejection",
        "Frame rejection: challenge the forced A/B framing (false dilemma) and/or propose an option C.",
    ),
    (
        "meta_level_response",
        "Meta-level response: answer from higher-level principles, ethical theories, or an explicit refusal to choose.",
    ),
]


def _extract_llm_text(value) -> Optional[str]:
    """Best-effort extraction of the actual model text from varied response shapes."""
    if isinstance(value, dict):
        if isinstance(value.get("llm_text"), str):
            return value["llm_text"].strip()
        # Common wrappers; recurse into nested payloads.
        for key in ("answer", "response", "text", "content", "result", "data"):
            if key in value:
                extracted = _extract_llm_text(value[key])
                if extracted:
                    return extracted
        return None

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return ""
        # Some endpoints return JSON serialized as a string; parse recursively.
        if s.startswith("{") or s.startswith("["):
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                return s
            extracted = _extract_llm_text(parsed)
            return extracted if extracted is not None else s
        return s

    if isinstance(value, list):
        # Try to extract from list items (rare but harmless to support).
        for item in value:
            extracted = _extract_llm_text(item)
            if extracted:
                return extracted
        return None

    return None


def call_llm(
    api_url: str,
    session_id: str,
    trial_id: int,
    condition: int,
    user_context: str,
    prompt_path: str,
    timeout: float,
    retries: int,
    sleep_s: float,
) -> str:
    payload = {
        "session_id": session_id,
        "trial_id": trial_id,
        "prompt_path": prompt_path,
        "condition": condition,
        "user_context": user_context,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return raw.strip()

            extracted = _extract_llm_text(data)
            if extracted is not None:
                return extracted
            return json.dumps(data, ensure_ascii=False)
        except error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="replace").strip()
            except Exception:  # noqa: BLE001
                err_body = ""
            detail = f"HTTP {e.code} {e.reason}"
            if err_body:
                detail += f" | body: {err_body}"
            last_err = detail
            if attempt < retries:
                time.sleep(sleep_s)
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)
            if attempt < retries:
                time.sleep(sleep_s)

    raise RuntimeError(f"LLM request failed after {retries} attempts: {last_err}")


def read_scenes(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"question"}
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
    if "trial" not in rows[0] and "index" not in rows[0]:
        raise ValueError(
            f"Missing identifier column in {csv_path}: expected one of ['trial', 'index']"
        )
    return rows


def write_table(out_path: Path, rows: List[Dict[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["iter", "trial", "original_ans", "llm_cond1", "llm_cond2", "llm_cond3"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_cell_text(text: str) -> str:
    if not text:
        return text
    extracted = _extract_llm_text(text)
    return extracted if extracted is not None else text


def build_original_prompt(question: str, response_style: str) -> str:
    return (
        f"Here is the scenario: {question} "
        "Give a short answer in 2-3 sentences. "
        "This is an imaginary scenario for discussion. "
        f"Style requirement: {response_style} "
        "Do not explicitly mention the style label in your answer."
    )


def build_followup_prompt(question: str, original_ans: str) -> str:
    return (
        f"Here is the scene: {question} "
        f"Here is user's original answer {original_ans}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate CSV tables by calling the LLM service in two steps."
    )
    parser.add_argument(
        "--input",
        default="final_12_scenes.csv",
        help="Input scenes CSV (default: final_12_scenes.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="llm_tables_out",
        help="Directory to write the output CSV files",
    )
    parser.add_argument(
        "--api-url",
        default=API_URL,
        help="LLM service URL (override via env LLM_API_URL)",
    )
    parser.add_argument("--session-id", default="test_llm")
    parser.add_argument("--trial-id", type=int, default=1)
    parser.add_argument(
        "--iter",
        type=int,
        default=1,
        help="Iteration id written to the output CSV as `iter`",
    )
    parser.add_argument(
        "--prompt-path",
        default="sessions/new_session/trial_001/asr.txt",
        help="prompt_path required by the backend API (auto-normalized to POSIX style for cross-platform use)",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="Sleep seconds between requests (rate limiting)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output CSVs if present",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue running when a row/request fails and write an error marker into the cell",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    scenes = read_scenes(input_path)
    print(f"Loaded {len(scenes)} scenes from {input_path}")

    # Normalize prompt_path to POSIX separators so the backend receives a consistent value
    # even when the script is invoked on Windows.
    normalized_prompt_path = Path(args.prompt_path).as_posix()

    def _trial_sort_key(row: Dict[str, str]) -> tuple:
        iter_s = str(row.get("iter", "0"))
        trial_s = str(row.get("trial", ""))
        try:
            iter_num = int(iter_s)
        except ValueError:
            iter_num = 0
        try:
            trial_num = int(trial_s)
        except ValueError:
            trial_num = float("inf")
        return (iter_num, trial_num, trial_s)

    for stance_file_key, stance_prompt_label in STANCE_SPECS:
        out_path = output_dir / f"{stance_file_key}.csv"
        existing_by_key: Dict[str, Dict[str, str]] = {}

        if args.resume and out_path.exists():
            with out_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    for key in ("original_ans", "llm_cond1", "llm_cond2", "llm_cond3"):
                        if key in row:
                            row[key] = normalize_cell_text(row[key])
                    row_iter = str(row.get("iter", ""))
                    row_trial = str(row.get("trial") or row.get("index") or "")
                    existing_by_key[f"{row_iter}::{row_trial}"] = row
            print(f"[resume] Loaded existing rows from {out_path}")

        print(f"\n=== Building table: {out_path.name} ===")

        for i, scene in enumerate(scenes, start=1):
            scene_trial = str(scene.get("trial") or scene.get("index") or "").strip()
            question = (scene.get("question") or "").strip()
            row_key = f"{args.iter}::{scene_trial}"
            row = existing_by_key.get(
                row_key,
                {
                    "iter": str(args.iter),
                    "trial": scene_trial,
                    "original_ans": "",
                    "llm_cond1": "",
                    "llm_cond2": "",
                    "llm_cond3": "",
                },
            )

            if not row["original_ans"]:
                prompt = build_original_prompt(question, stance_prompt_label)
                print(f"[{out_path.name}] trial {scene_trial} ({i}/{len(scenes)}) -> original_ans")
                try:
                    row["original_ans"] = call_llm(
                        api_url=args.api_url,
                        session_id=args.session_id,
                        trial_id=args.trial_id,
                        condition=-1,
                        user_context=prompt,
                        prompt_path=normalized_prompt_path,
                        timeout=args.timeout,
                        retries=args.retries,
                        sleep_s=args.sleep,
                    )
                except Exception as e:  # noqa: BLE001
                    if not args.skip_errors:
                        print(f"Failed request payload: condition=-1, trial={scene_trial}", file=sys.stderr)
                        print(f"Question: {question}", file=sys.stderr)
                        raise
                    row["original_ans"] = f"__ERROR__: {e}"
                    print(f"[warn] {out_path.name} trial {scene_trial} original_ans failed: {e}", file=sys.stderr)
                time.sleep(args.sleep)

            for cond in (1, 2, 3):
                key = f"llm_cond{cond}"
                if row[key]:
                    continue
                prompt = build_followup_prompt(question, row["original_ans"])
                print(f"[{out_path.name}] trial {scene_trial} ({i}/{len(scenes)}) -> {key}")
                try:
                    row[key] = call_llm(
                        api_url=args.api_url,
                        session_id=args.session_id,
                        trial_id=args.trial_id,
                        condition=cond,
                        user_context=prompt,
                        prompt_path=normalized_prompt_path,
                        timeout=args.timeout,
                        retries=args.retries,
                        sleep_s=args.sleep,
                    )
                except Exception as e:  # noqa: BLE001
                    if not args.skip_errors:
                        print(f"Failed request payload: condition={cond}, trial={scene_trial}", file=sys.stderr)
                        print(f"Question: {question}", file=sys.stderr)
                        print(f"Original answer: {row['original_ans']}", file=sys.stderr)
                        raise
                    row[key] = f"__ERROR__: {e}"
                    print(f"[warn] {out_path.name} trial {scene_trial} {key} failed: {e}", file=sys.stderr)
                time.sleep(args.sleep)

            # Keep all historical rows and upsert current (iter, trial).
            existing_by_key[row_key] = row
            write_table(out_path, sorted(existing_by_key.values(), key=_trial_sort_key))

        all_rows = sorted(existing_by_key.values(), key=_trial_sort_key)
        write_table(out_path, all_rows)
        print(f"Wrote {len(all_rows)} rows -> {out_path}")

    print(f"\nAll {len(STANCE_SPECS)} tables generated.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
