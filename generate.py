"""
generate.py — Generazione varianti hero title via Anthropic LLM.

Genera 60 varianti (3 categorie × 20) a partire da un titolo hero originale.
"""

import json
import os
import time
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-20250514"
MAX_RETRIES = 3

SYSTEM_PROMPT = (
    "You are a conversion copywriter expert. "
    "Respond ONLY with a valid JSON array of strings. "
    "No markdown, no explanations, no extra text.\n"
    'Example format: ["title 1", "title 2", "title 3"]'
)

CATEGORY_PROMPTS = {
    "similar": (
        'Original hero title: "{title}"\n\n'
        "Generate exactly 20 hero section titles that are SIMILAR to the original.\n"
        "Rules:\n"
        "- Same overall tone and message\n"
        "- Same approximate length (±20% words)\n"
        "- Small variations: synonyms, word order, minor reframing\n"
        "- Must feel like A/B test variants of the same copy\n"
        "Return a JSON array of exactly 20 strings."
    ),
    "alternative": (
        'Original hero title: "{title}"\n\n'
        "Generate exactly 20 hero section titles that are ALTERNATIVE to the original.\n"
        "Rules:\n"
        "- Different angle: try benefit-led, problem-led, question format, "
        "social proof hook, curiosity gap, \"how\" framing\n"
        "- Same product/service implied\n"
        "- Different emotional register or logical structure\n"
        "- Each title should feel meaningfully different from the others\n"
        "Return a JSON array of exactly 20 strings."
    ),
    "exaggerated": (
        'Original hero title: "{title}"\n\n'
        "Generate exactly 20 hero section titles that are EXAGGERATED versions.\n"
        "Rules:\n"
        "- Hyperbolic, bold, direct-response marketing style\n"
        "- Big promises, strong verbs, urgency, superlatives\n"
        "- Think: late-night infomercial meets Silicon Valley pitch deck\n"
        "- Push the language to its extreme while keeping it about the same product/service\n"
        "Return a JSON array of exactly 20 strings."
    ),
}


def _call_llm(client: anthropic.Anthropic, category: str, title: str) -> list[str]:
    """Chiama l'API Anthropic per generare 20 varianti di una categoria."""
    user_prompt = CATEGORY_PROMPTS[category].format(title=title)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_text = response.content[0].text.strip()

            # Pulizia: rimuovi eventuale markdown fencing
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1]
                if raw_text.endswith("```"):
                    raw_text = raw_text[: raw_text.rfind("```")]
                raw_text = raw_text.strip()

            titles = json.loads(raw_text)

            if not isinstance(titles, list) or len(titles) != 20:
                raise ValueError(
                    f"Expected list of 20 strings, got {type(titles).__name__} "
                    f"with {len(titles) if isinstance(titles, list) else '?'} items"
                )

            if not all(isinstance(t, str) for t in titles):
                raise ValueError("Not all items are strings")

            return titles

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"  ⚠ {category} attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(1)
            else:
                raise RuntimeError(
                    f"Failed to generate {category} variants after {MAX_RETRIES} attempts"
                ) from e

    return []  # unreachable


def generate_variants(
    title: str, api_key: Optional[str] = None
) -> dict:
    """
    Genera 60 varianti (20 per categoria) del titolo hero fornito.

    Returns:
        dict con chiavi: input_title, similar, alternative, exaggerated
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY non trovata. Impostala nel file .env o passala come argomento."
        )
    client = anthropic.Anthropic(api_key=key)

    result = {"input_title": title}

    for category in ["similar", "alternative", "exaggerated"]:
        print(f"  → Generating {category} variants...")
        result[category] = _call_llm(client, category, title)
        print(f"    ✓ {len(result[category])} variants generated")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    variants = generate_variants(args.title)
    out_path = "output/variants.json"
    os.makedirs("output", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(variants, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")
