import re

from mlrag.text import canonical_for_match, normalize, script_of, split_sentences, tokenize_words


def test_naive_word_regex_shreds_devanagari():
    # This is why tokenize_words walks Unicode categories instead of using \w+.
    assert len(re.findall(r"\w+", "नेपालको")) > 1


def test_devanagari_words_stay_whole():
    assert tokenize_words("काठमाडौं नेपालको राजधानी हो।") == ["काठमाडौं", "नेपालको", "राजधानी", "हो"]


def test_length_folding_is_optional_and_collapses_variants():
    assert tokenize_words("राजधानी") != tokenize_words("राजधानि")
    assert tokenize_words("राजधानी", fold_length=True) == tokenize_words("राजधानि", fold_length=True)
    assert tokenize_words("ईश्वर", fold_length=True) == tokenize_words("इश्वर", fold_length=True)
    assert tokenize_words("हुँ", fold_length=True) == tokenize_words("हुं", fold_length=True)


def test_devanagari_digits_become_ascii():
    assert "2024" in tokenize_words("सन् २०२४ मा")


def test_zero_width_characters_are_removed():
    assert normalize("क‍ष") == "कष"
    assert tokenize_words("राजधानी‌") == ["राजधानी"]


def test_latin_tokens_are_lowercased():
    assert tokenize_words("Kathmandu IS the Capital!") == ["kathmandu", "is", "the", "capital"]


def test_sentence_split_handles_danda_and_latin_punctuation():
    s = split_sentences("यो पहिलो हो। यो दोस्रो हो। Is this third? Yes.")
    assert s == ["यो पहिलो हो।", "यो दोस्रो हो।", "Is this third?", "Yes."]


def test_script_detection():
    assert script_of("नेपालको राजधानी") == "deva"
    assert script_of("Nepal ko rajdhani") == "latin"
    assert script_of("Nepal को राजधानी") == "mixed"
    assert script_of("123 ...") == "none"


def test_canonical_for_match_ignores_case_whitespace_and_composition():
    assert canonical_for_match("  Hello   WORLD ") == "hello world"
    assert canonical_for_match("é") == canonical_for_match("é")
