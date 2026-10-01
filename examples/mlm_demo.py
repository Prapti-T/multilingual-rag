"""Optional: what a multilingual masked language model does with Devanagari vs Romanized text.

    pip install -e ".[mlm]" && python examples/mlm_demo.py

Prints how the tokenizer fragments the same sentence written three ways (tokens per word is a
direct proxy for cost and for how much context a chunk uses), then asks the MLM to fill a masked
word. This is a demonstration, not part of the retrieval pipeline.
"""
import sys

from transformers import AutoTokenizer, pipeline

MODEL = "xlm-roberta-base"
SENTENCES = {
    "english": "Kathmandu is the capital of Nepal.",
    "devanagari": "काठमाडौं नेपालको राजधानी हो।",
    "romanized": "Kathmandu Nepal ko rajdhani ho.",
}

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

tok = AutoTokenizer.from_pretrained(MODEL)
for name, s in SENTENCES.items():
    pieces = tok.tokenize(s)
    print(f"{name:11s} {len(pieces):3d} tokens / {len(s.split())} words "
          f"= {len(pieces) / len(s.split()):.2f} tokens per word")
    print("            ", pieces)

fill = pipeline("fill-mask", model=MODEL)
masked = "काठमाडौं नेपालको <mask> हो।"
print(f"\n{masked}")
for p in fill(masked, top_k=5):
    print(f"  {p['score']:.3f}  {p['token_str']}")
