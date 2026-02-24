"""
GenAI Restaurant Search using Qwen (local inference via transformers)

Queries a Qwen model for the top 5 restaurants at a given zip code.
This simulates a GenAI-style search where the model answers directly
from its training knowledge — for comparison against Google Search results.

The model is downloaded from HuggingFace on first run and cached locally.

Usage:
    python genai_search.py --zip 10001
    python genai_search.py --zip 06013 --json
    python genai_search.py --zip 90210 --model Qwen/Qwen2.5-1.5B-Instruct

Requirements:
    pip install transformers torch
"""

import argparse
import json
import sys
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = """\
You are a helpful local search assistant. When given a US zip code, you return
the top 5 restaurants in that area based on your knowledge. Always respond with
valid JSON — no markdown fences, no extra commentary.

The JSON must follow this exact schema:
{
  "zip_code": "<zip>",
  "restaurants": [
    {
      "rank": 1,
      "name": "<restaurant name>",
      "cuisine": "<cuisine type>",
      "address": "<street address if known, else null>",
      "notable_for": "<one sentence describing what makes it notable>"
    }
  ]
}

Return exactly 5 restaurants. If you are uncertain about specific details,
use your best knowledge and mark uncertain fields with null.
"""


def build_user_prompt(zip_code: str) -> str:
    return (
        f"What are the top 5 restaurants near zip code {zip_code} in the United States? "
        "Return only the JSON object described in the system instructions."
    )


def query_qwen(zip_code: str, model_id: str) -> dict:
    """Run the Qwen model locally and return parsed JSON."""
    print(f"Loading {model_id}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(zip_code)},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    print("Generating...", file=sys.stderr)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Strip markdown code fences if the model adds them despite instructions
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


def print_results(data: dict) -> None:
    zip_code = data.get("zip_code", "unknown")
    restaurants = data.get("restaurants", [])

    print(f"\nTop 5 restaurants near zip code {zip_code} (GenAI Search via Qwen)\n")
    print("-" * 60)
    for r in restaurants:
        print(f"#{r['rank']}  {r['name']}")
        print(f"    Cuisine : {r['cuisine']}")
        if r.get("address"):
            print(f"    Address : {r['address']}")
        print(f"    Notable : {r['notable_for']}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use a Qwen model to find top 5 restaurants by zip code."
    )
    parser.add_argument(
        "--zip", required=True, metavar="ZIP_CODE",
        help="US zip code to search (e.g. 10001)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL}). "
             "Other options: Qwen/Qwen2.5-1.5B-Instruct (more capable)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print raw JSON output instead of formatted text"
    )
    args = parser.parse_args()

    if not re.fullmatch(r"\d{5}", args.zip):
        print(f"Error: '{args.zip}' is not a valid 5-digit US zip code.", file=sys.stderr)
        sys.exit(1)

    data = query_qwen(zip_code=args.zip, model_id=args.model)

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print_results(data)


if __name__ == "__main__":
    main()
