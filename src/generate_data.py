"""Generate synthetic (symptom, structured response) training pairs.

Uses GPT-4o-mini as a teacher model to produce schema-valid pre-visit prep
responses for ~85 seed symptoms. Each seed spawns several patient-voice
variations, and every output is validated against OUTPUT_SCHEMA before
being kept.

Reads OPENAI_API_KEY from the local Zona .env so the user does not need to
export it manually. Writes data/processed/train.jsonl and val.jsonl.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from jsonschema import ValidationError, validate
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import OUTPUT_SCHEMA, SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parent.parent
SEEDS_PATH = ROOT / "data" / "seed_symptoms.txt"
OUT_DIR = ROOT / "data" / "processed"

TEACHER_MODEL = "gpt-4o-mini"
VARIATIONS_PER_SEED = 6
VAL_FRACTION = 0.1
SEED = 42

VARIATION_PROMPT = """Rewrite the following patient-reported symptom in {n} different but realistic ways. Vary the tone, specificity, and which details the patient emphasizes. Keep every version 1 sentence. Output a JSON array of strings only, no preamble.

Symptom: "{symptom}"
"""


def load_env():
    candidates = [
        Path.home() / "zona-superadmin-server" / ".env",
        Path("/Users/k/zona-superadmin-server/.env"),
    ]
    for path in candidates:
        if path.exists():
            load_dotenv(path)
            break
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY not found. Set it in the environment or in "
            "~/zona-superadmin-server/.env"
        )


def generate_variations(client: OpenAI, symptom: str, n: int) -> list[str]:
    resp = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[
            {"role": "system", "content": "You rewrite patient symptom descriptions. Output valid JSON only."},
            {"role": "user", "content": VARIATION_PROMPT.format(n=n, symptom=symptom)},
        ],
        temperature=0.9,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    # Model may wrap in an object; extract first list value.
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [s.strip() for s in parsed if isinstance(s, str)]
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return [s.strip() for s in v if isinstance(s, str)]
    except json.JSONDecodeError:
        pass
    return []


def generate_response(client: OpenAI, symptom: str) -> dict | None:
    resp = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": symptom},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
        validate(instance=data, schema=OUTPUT_SCHEMA)
        return data
    except (json.JSONDecodeError, ValidationError):
        return None


def main():
    load_env()
    client = OpenAI()

    seeds = [
        line.strip()
        for line in SEEDS_PATH.read_text().splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(seeds)} seed symptoms.")

    pairs = []
    skipped = 0

    for seed in tqdm(seeds, desc="seeds"):
        # Include the original seed + generated variations
        candidates = [seed]
        variations = generate_variations(client, seed, VARIATIONS_PER_SEED)
        candidates.extend(variations)

        for symptom in candidates:
            response = generate_response(client, symptom)
            if response is None:
                skipped += 1
                continue
            pairs.append({
                "input": symptom,
                "output": json.dumps(response, ensure_ascii=False),
            })

    print(f"Kept {len(pairs)} valid pairs, skipped {skipped} invalid.")

    import random
    random.Random(SEED).shuffle(pairs)
    cut = int(len(pairs) * (1 - VAL_FRACTION))
    train, val = pairs[:cut], pairs[cut:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train.jsonl", train), ("val.jsonl", val)]:
        path = OUT_DIR / name
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
