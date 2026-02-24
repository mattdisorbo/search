"""
GenAI Restaurant Search using Qwen (via Hugging Face Inference API)

Queries a Qwen model for the top 5 restaurants at a given zip code.
This simulates a GenAI-style search where the model answers directly
from its training knowledge — for comparison against Google Search results.

Usage:
    python genai_search.py --zip 10001
    python genai_search.py --zip 90210 --model Qwen/Qwen2.5-72B-Instruct

Requirements:
    pip install huggingface_hub
    Set HF_TOKEN env var (optional, but raises rate limits):
        export HF_TOKEN=hf_your_token_here
"""

import argparse
import json
import os
import sys
import re
from huggingface_hub import InferenceClient


DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"

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


def query_qwen(zip_code: str, model: str, hf_token: str | None) -> dict:
    """Call the Qwen model via HF Inference API and return parsed JSON."""
    client = InferenceClient(model=model, token=hf_token)

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(zip_code)},
        ],
        max_tokens=1024,
        temperature=0.3,  # low temp for more consistent, factual answers
    )

    raw = response.choices[0].message.content.strip()

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
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print raw JSON output instead of formatted text"
    )
    args = parser.parse_args()

    if not re.fullmatch(r"\d{5}", args.zip):
        print(f"Error: '{args.zip}' is not a valid 5-digit US zip code.", file=sys.stderr)
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print(
            "Note: HF_TOKEN not set. Using unauthenticated access (rate-limited).\n"
            "Set HF_TOKEN to a free Hugging Face token to raise limits.\n",
            file=sys.stderr,
        )

    print(f"Querying {args.model} for restaurants near {args.zip}...", file=sys.stderr)

    data = query_qwen(zip_code=args.zip, model=args.model, hf_token=hf_token)

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print_results(data)


if __name__ == "__main__":
    main()
