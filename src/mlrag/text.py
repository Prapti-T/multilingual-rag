"""Unicode-aware text helpers for Nepali (Devanagari), Romanized Nepali and English.

Two details here matter more than they look:

* Python's ``\\w`` does not match Devanagari vowel signs and the virama (they are
  combining marks, not "alphanumeric"), so ``re.findall(r"\\w+", ...)`` shreds Nepali
  words into fragments. ``tokenize_words`` walks Unicode categories instead.
* Hrasva/dirgha (short/long vowel) spelling variation is common in real Nepali text.
  ``fold_length=True`` collapses it. It is applied only to the lexical (sparse) analyzer,
  never to text that is shown to a user.
"""
from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH = {ord(c): None for c in "​‌‍⁠﻿"}
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# long -> short vowel signs and independent vowels; chandrabindu -> anusvara
_LENGTH_FOLD = str.maketrans({"ी": "ि", "ू": "ु", "ई": "इ", "ऊ": "उ", "ँ": "ं"})

_SENT_END = re.compile(r"(?<=[।॥.!?])\s+")
_INLINE_WS = re.compile(r"[ \t ]+")


def normalize(text: str) -> str:
    """NFC, drop zero-width characters, collapse inline whitespace. Newlines are kept."""
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    text = _INLINE_WS.sub(" ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def _is_word_char(ch: str) -> bool:
    # Letters, combining marks (Devanagari matras, virama, nukta) and numbers.
    return unicodedata.category(ch)[0] in "LMN"


def tokenize_words(text: str, *, fold_length: bool = False) -> list[str]:
    """Lowercased word tokens. Devanagari digits become ASCII digits."""
    text = normalize(text).casefold().translate(_DEV_DIGITS)
    if fold_length:
        text = text.translate(_LENGTH_FOLD)
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if _is_word_char(ch):
            buf.append(ch)
        elif buf:
            tokens.append("".join(buf))
            buf = []
    if buf:
        tokens.append("".join(buf))
    return tokens


def split_sentences(text: str) -> list[str]:
    """Split on the Devanagari danda (।, ॥) as well as . ! ?"""
    return [s.strip() for s in _SENT_END.split(text) if s.strip()]


def script_of(text: str) -> str:
    """'deva' | 'latin' | 'mixed' | 'none', by share of letters."""
    deva = latin = 0
    for ch in text:
        if "ऀ" <= ch <= "ॿ":
            if unicodedata.category(ch)[0] == "L":
                deva += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1
    total = deva + latin
    if total == 0:
        return "none"
    if deva / total >= 0.8:
        return "deva"
    if latin / total >= 0.8:
        return "latin"
    return "mixed"


def canonical_for_match(text: str) -> str:
    """Normalization used when checking that a gold span appears in a chunk."""
    return " ".join(normalize(text).casefold().split())
