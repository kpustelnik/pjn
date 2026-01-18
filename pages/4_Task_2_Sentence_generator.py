"""Task 2: Syntax Generation - sentence generator."""

import streamlit as st
import morfeusz2
import os
import re
import pickle
from uuid import uuid4
from pathlib import Path
import yaml
import sys
from collections import Counter
import spacy

st.set_page_config(page_title="Task 2: Syntax generator", layout="wide")
st.title("Task 2: Syntax generation")

# Load config
sys.path.insert(0, str(Path(__file__).parent.parent))
config_path = Path(__file__).parent.parent / "config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Check if corpus is loaded
if not st.session_state.get('corpus_loaded', False):
    st.warning("Please load and process a corpus first in the Load Data page.")
    st.stop()

st.markdown("""
### Task Objective
Build a generator for simple Polish sentences based on SVO (Subject-Verb-Object) structure, designed for people practicing Polish language.

### Features
This generator creates grammatically correct sentences with:
- Subject-Verb agreement (person, number, gender)
- Verb-Object case agreement (accusative, genitive)
- Support for pronouns and noun phrases
- Various tenses (present, past, future)
- Different moods (indicative, conditional)
""")

st.markdown("Build the sentence as a tree - add parts of sentence as nodes")

@st.cache_resource
def get_morfeusz():
    return morfeusz2.Morfeusz(generate=True)

morf: morfeusz2.Morfeusz = get_morfeusz()

# === FUNCTIONS FOR BIGRAMS ===
# Check for cached data
cache_dir = Path(config['corpus']['cache_dir'])
BIGRAM_CACHE_PATH = cache_dir / "bigram_index.pkl"
WORDS_CACHE_PATH = cache_dir / "corpus_words.pkl"

@st.cache_data
def build_bigram_index(_tokens_df) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    """
    Builds bigram index (forms and lemmas) for fast lookup.
    Caches result to pickle file for persistence between sessions.
    """

    # Check if cache exists
    if os.path.exists(BIGRAM_CACHE_PATH):
        try:
            with open(BIGRAM_CACHE_PATH, 'rb') as f:
                cached = pickle.load(f)
                # Check if cache is fresh (compare DataFrame size)
                if cached.get('df_len') == len(_tokens_df):
                    return cached['token_idx'], cached['lemma_idx']
        except:
            pass

    if _tokens_df is None or len(_tokens_df) == 0:
        return {}, {}

    # Retrieve all tokens and lemmas
    tokens = _tokens_df['token'].str.lower().tolist() if 'token' in _tokens_df.columns else []
    lemmas = _tokens_df['lemma'].str.lower().tolist() if 'lemma' in _tokens_df.columns else []

    # Count connections of tokens and lemmas
    token_bigrams = Counter()
    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i+1])
        token_bigrams[bigram] += 1

    lemma_bigrams = Counter()
    for i in range(len(lemmas) - 1):
        bigram = (lemmas[i], lemmas[i+1])
        lemma_bigrams[bigram] += 1

    # Change the counters to dicts
    token_idx = dict(token_bigrams)
    lemma_idx = dict(lemma_bigrams)

    # Save to cache file
    try:
        os.makedirs(os.path.dirname(BIGRAM_CACHE_PATH), exist_ok=True)
        with open(BIGRAM_CACHE_PATH, 'wb') as f:
            pickle.dump({
                'df_len': len(_tokens_df),
                'token_idx': token_idx,
                'lemma_idx': lemma_idx
            }, f)
    except:
        pass

    return token_idx, lemma_idx

# Selector for spaCy model
with st.sidebar:
    st.markdown("---")
    st.subheader("spaCy Lemmatizer")
    spacy_models = ["pl_core_news_sm", "pl_core_news_md", "pl_core_news_lg"]
    selected_spacy_model = st.selectbox(
        "spaCy model for lemmatization:",
        spacy_models,
        index=2,
        key="spacy_model_select"
    )
    st.caption("If the selected model is not installed, lemmatization will fallback to Morfeusz.")

@st.cache_resource
def get_nlp(model_name: str):
    """Load spaCy model, cached."""
    try:
        nlp = spacy.load(model_name)
        return nlp
    except Exception:
        return None

def get_lemma(word: str) -> str:
    """Get word lemma via spaCy first, fallback to Morfeusz."""
    model_name = st.session_state.get("spacy_model_select", "pl_core_news_sm")
    try:
        nlp = get_nlp(model_name)
        if nlp:
            doc = nlp(word)
            if doc and doc[0].lemma_:
                return doc[0].lemma_
    except Exception:
        pass
    # Fallback to Morfeusz
    try:
        analyses = morf.analyse(word)
        if analyses:
            return analyses[0][2][1]
    except Exception:
        pass
    return word.lower()

def get_bigram_counts(w1: str, w2: str, l1: str, l2: str) -> tuple[int, int]:
    """Returns (count_forms, count_lemmas) for bigram."""
    token_idx, lemma_idx = st.session_state.bigram_cache

    token_count = token_idx.get((w1.lower(), w2.lower()), 0)
    lemma_count = lemma_idx.get((l1.lower(), l2.lower()), 0)
    return token_count, lemma_count

def get_bigram_color(cnt_form: int, cnt_lemma: int) -> tuple[str, str]:
    """Returns status description with icons based on frequency."""
    if cnt_form > 0:
        if cnt_form >= 10:
            return '🔵', f"Frequent ({cnt_form}x exact)"
        elif cnt_form >= 5:
            return '🟢', f"Common ({cnt_form}x exact)"
        elif cnt_form >= 2:
            return '🟡', f"Occurs ({cnt_form}x exact)"
        else:
            return '🔴', f"Rare ({cnt_form}x exact)"
    elif cnt_lemma > 0:
        return '⚠️', f"Lemmas only: {cnt_lemma}x"
    else:
        return '❌', "Not in corpus"


def parse_tag(tag: str) -> list[list[str]]:
    """Parses a morphological tag into its components."""
    parts: list[str] = tag.split(":")
    expanded_parts: list[list[str]] = []
    for part in parts:
        expanded_parts.append(part.split("."))
    return expanded_parts

# === MORPHOLOGICAL FUNCTIONS ===
def get_noun_gender(lemma):
    try:
        for _, _, interp in morf.analyse(lemma):
            tag: str = interp[2]
            parsed_tag = parse_tag(tag)
            if 'subst' in parsed_tag[0]: # Rzeczownik
                return parsed_tag[3][0] if len(parsed_tag) >= 4 and parsed_tag[3] else "m3"
    except:
        pass
    return "m3"


def get_noun_form(lemma, case, number="sg"):
    try:
        for form, _, tag, *_ in morf.generate(lemma):
            parsed_tag = parse_tag(tag)
            if 'subst' not in parsed_tag[0] or number not in parsed_tag[1]:
                continue
            if case not in parsed_tag[2]:
                continue
            return form
    except:
        pass
    return lemma


def get_adj_form(lemma, case, gender, number="sg", degree="pos"):
    if not lemma:
        return ""
    try:
        for form, _, tag, *_ in morf.generate(lemma):
            parsed_tag = parse_tag(tag)
            if "adj" not in parsed_tag[0]:
                continue

            if len(parsed_tag) < 5:
                continue
            if number not in parsed_tag[1] or degree not in parsed_tag[4]:
                continue
            if case not in parsed_tag[2]:
                continue
            g_options = parsed_tag[3]
            # In plural: m1 = virile, m2/m3/f/n = non-virile
            if number == "pl":
                if gender == "m1" and "m1" in g_options:
                    return form
                # m2, m3 use non-virile forms (together with f, n)
                if gender in ["m2", "m3"] and any(x in g_options for x in ["m2", "m3", "n"]):
                    if "m1" not in g_options or any(x in g_options for x in ["m2", "m3", "f", "n"]):
                        return form
                if gender == "f" and "f" in g_options:
                    return form
                if gender == "n" and "n" in g_options:
                    return form
            else:
                # Singular - all masculine genders are similar
                if gender.startswith("m") and any(x in g_options for x in ["m1", "m2", "m3"]):
                    return form
                if gender == "f" and "f" in g_options:
                    return form
                if gender == "n" and "n" in g_options:
                    return form
    except:
        pass
    return lemma


def get_verb_aspects(lemma: str) -> set[str]:
    try:
        forms = morf.generate(lemma)
        aspects = set()
        for _, _, tag, *_ in forms:
            parsed_tag = parse_tag(tag)
            if any('imperf' in x for x in parsed_tag):
                aspects.add('imperf')
            if any('perf' in x for x in parsed_tag):
                aspects.add('perf')
        return aspects if aspects else {'imperf'}
    except:
        return {'imperf'}


def has_adj_comparison(lemma: str) -> bool:
    """
    Checks if adjective has comparative/superlative forms (com/sup).
    Adjectives like 'sam', 'martwy', 'drewniany' do not have grading.
    """
    if not lemma:
        return False
    try:
        forms = morf.generate(lemma)
        for _, _, tag, *_ in forms:
            parsed_tag = parse_tag(tag)
            if 'adj' in parsed_tag[0]:
                if any(x in y for x in ['com', 'sup'] for y in parsed_tag):
                    return True
    except:
        pass
    return False


def _get_gender_tags(gender: str, num: str) -> list[str]:
    if num == "pl":
        return ["m1"] if gender == "m1" else ["m2", "m3", "f", "n"]
    return ["m1", "m2", "m3"] if gender.startswith("m") else (["f"] if gender == "f" else ["n"])


def _match_case_in_tag(tag, case, case_position=2):
    """Checks if tag contains given case in appropriate field.

    Tag structure: class:number:case:gender:...
    e.g. ppas:pl:nom.voc:m1:imperf:aff
    case can be complex: nom.acc.voc
    """
    parsed_tag = parse_tag(tag)
    if len(parsed_tag) <= case_position:
        return False
    return case in parsed_tag[case_position]


def _match_gender_in_tag(tag, g_tags):
    """Checks if tag contains any of the given genders in appropriate field."""
    parsed_tag = parse_tag(tag)
    if len(parsed_tag) < 4:
        return False
    # For ppas gender is in part 3 (0-indexed): ppas:sg:nom.voc:f:imperf:aff
    gender_part = parsed_tag[3]
    return any(g in gender_part for g in g_tags)


def get_verb_form(lemma, person, number, tense, mood, gender="m1", aspect="niedokonany", voice="czynna"):
    num = "sg" if number == "sing" else "pl"
    pri = {"1": "pri", "2": "sec", "3": "ter"}[person]
    is_perf = aspect == "dokonany"

    try:
        forms = morf.generate(lemma)
        byc_forms = morf.generate("być")
        g_tags = _get_gender_tags(gender, num)

        # Passive voice
        if voice == "bierna":
            ppas = None
            for form, _, tag, *_ in forms:
                parsed_tag = parse_tag(tag)
                if 'ppas' in parsed_tag[0] and num in parsed_tag[1] and 'nom' in parsed_tag[2] and 'aff' in parsed_tag[5]:
                    # Could technically add a negation here?
                    if _match_gender_in_tag(tag, g_tags):
                        ppas = form
                        break
            if not ppas:
                return None

            # Conditional passive: byłbym/byłbyś/byłby + ppas
            if mood == "cond":
                # Find praet form of "być" (był/była/było/byli/były)
                byc_praet = None
                for bf, _, bt, *_ in byc_forms:
                    parsed_tag = parse_tag(bt)
                    if 'praet' in parsed_tag[0] and num in parsed_tag[1]:
                        if len(parsed_tag) > 2 and any(g in parsed_tag[2] for g in g_tags):
                            byc_praet = bf
                            break
                if not byc_praet:
                    return None

                if person == '3':
                    return f"{byc_praet}by {ppas}"
                # For 1st and 2nd person: byłbym, byłbyś, bylibyśmy, bylibyście
                for ag, _, at, *_ in byc_forms:
                    parsed_tag = parse_tag(at)
                    if 'aglt' in parsed_tag[0] and num in parsed_tag[1] and pri in parsed_tag[2] and 'nwok' in parsed_tag[4]:
                        return f"{byc_praet}by{ag} {ppas}"
                return f"{byc_praet}by {ppas}"

            if tense == "praes":
                for bf, _, bt, *_ in byc_forms:
                    parsed_tag = parse_tag(bt)
                    if 'fin' in parsed_tag[0] and num in parsed_tag[1] and pri in parsed_tag[2]:
                        return f"{bf} {ppas}"
            elif tense == "fut":
                for bf, _, bt, *_ in byc_forms:
                    parsed_tag = parse_tag(bt)
                    if 'bedzie' in parsed_tag[0] and num in parsed_tag[1] and pri in parsed_tag[2]:
                        return f"{bf} {ppas}"
            elif tense == "praet":
                for bf, _, bt, *_ in byc_forms:
                    parsed_tag = parse_tag(bt)
                    if 'praet' in parsed_tag[0] and num in parsed_tag[1]:
                        if len(parsed_tag) > 2 and any(g in parsed_tag[2] for g in g_tags):
                            if person == '3':
                                return f"{bf} {ppas}"
                            for ag, _, at, *_ in byc_forms:
                                nparsed_tag = parse_tag(at)
                                if "aglt" in nparsed_tag[0] and num in nparsed_tag[1] and pri in nparsed_tag[2] and "wok" in nparsed_tag[4]:
                                    return f"{bf}{ag} {ppas}"
            return ppas

        # Imperative mood
        if mood == "impt":
            if person == "1" and num == "sg":
                return None
            if person == "3":
                return None
            for form, _, tag, *_ in forms:
                parsed_tag = parse_tag(tag)
                if 'impt' in parsed_tag[0] and num in parsed_tag[1] and pri in parsed_tag[2]:
                    return form
            return None

        # Conditional mood
        if mood == "cond":
            praet = None
            for form, _, tag, *_ in forms:
                parsed_tag = parse_tag(tag)
                if 'praet' in parsed_tag[0] and num in parsed_tag[1]:
                    if len(parsed_tag) > 2 and any(g in parsed_tag[2] for g in g_tags):
                        praet = form
                        break
            if not praet:
                return None
            if person == "3":
                return praet + "by"
            for ag, _, at, *_ in byc_forms:
                parsed_tag = parse_tag(at)
                if 'aglt' in parsed_tag[0] and num in parsed_tag[1] and pri in parsed_tag[2] and 'nwok' in parsed_tag[4]:
                    return praet + "by" + ag
            return praet + "by"

        # Future tense
        if tense == "fut":
            if is_perf:
                for form, _, tag, *_ in forms:
                    parsed_tag = parse_tag(tag)
                    if 'fin' in parsed_tag[0] and num in parsed_tag[1] and pri in parsed_tag[2]:
                        return form
            else:
                for bf, _, bt, *_ in byc_forms:
                    parsed_tag = parse_tag(bt)
                    if 'bedzie' in parsed_tag[0] and num in parsed_tag[1] and pri in parsed_tag[2]:
                        for pf, _, pt, *_ in forms:
                            nparsed_tag = parse_tag(pt)
                            if 'praet' in nparsed_tag[0] and num in nparsed_tag[1]:
                                if len(nparsed_tag) > 2 and any(g in nparsed_tag[2] for g in g_tags):
                                    return f"{bf} {pf}"
            return lemma

        # Present tense
        if tense == "praes":
            if is_perf:
                return None
            for form, _, tag, *_ in forms:
                parsed_tag = parse_tag(tag)
                if 'fin' in parsed_tag[0] and num in parsed_tag[1] and pri in parsed_tag[2]:
                    return form

        # Past tense
        if tense == "praet":
            praet = None
            for form, _, tag, *_ in forms:
                parsed_tag = parse_tag(tag)
                if 'praet' in parsed_tag[0] and num in parsed_tag[1]:
                    if len(parsed_tag) > 2 and any(g in parsed_tag[2] for g in g_tags):
                        praet = form
                        break
            if praet:
                if person == "3":
                    return praet
                for ag, _, at, *_ in byc_forms:
                    parsed_tag = parse_tag(at)
                    if "aglt" in parsed_tag[0] and num in parsed_tag[1] and pri in parsed_tag[2] and "wok" in parsed_tag[4]:
                        return praet + ag
                return praet

        for form, _, tag, *_ in forms:
            parsed_tag = parse_tag(tag)
            if 'fin' in parsed_tag[0] and num in parsed_tag[1] and pri in parsed_tag[2]:
                return form
    except:
        pass
    return lemma


# === PARTICIPLE GENERATION ===

def get_participle(verb_lemma, participle_type, gender="m1", number="sg", case="nom"):
    """
    Generates participle from verb.

    participle_type:
        - 'pact' - active adjectival (reading)
        - 'ppas' - passive adjectival (read)
        - 'pcon' - contemporary adverbial (reading)
        - 'pant' - anterior adverbial (having read)
    """
    try:
        forms = morf.generate(verb_lemma)
        if not forms:
            return None

        # Adverbial participles - uninflected
        if participle_type == 'pcon' or participle_type == 'pant':
            for form, _, tag, *_ in forms:
                parsed_tag = parse_tag(tag)
                if participle_type in parsed_tag[0]:
                    return form
            return None

        # Adjectival participles - inflected
        num: str = "sg" if number == "sg" else "pl"
        g_tags: list[str] = {
            "m1": ["m1"], "m2": ["m2"], "m3": ["m3"],
            "f": ["f"], "n": ["n"]
        }.get(gender, ["m1", "m2", "m3"])

        for form, _, tag, *_ in forms:
            parsed_tag = parse_tag(tag)
            if participle_type not in parsed_tag[0]:
                continue
            if num not in parsed_tag[1]:
                continue
            if not _match_case_in_tag(tag, case):
                continue
            if 'aff' not in parsed_tag[5]:
                continue

            # Check gender - for pact/ppas structure: type:number:case:gender:aspect:negation
            if any(g in parsed_tag[3] for g in g_tags):
                return form
        return None
    except:
        return None


# === CORPUS LOADING ===
def validate_word(lemma: str, pos_type: str) -> bool:
    try:
        forms = morf.generate(lemma)
        if pos_type == "verb":
            return any("fin" in parse_tag(f[2])[0] for f in forms)
        if pos_type == "noun":
            return any("subst" in parse_tag(f[2])[0] for f in forms)
        if pos_type == "adv":
            return any("adv" in parse_tag(f[2])[0] for f in forms)
        if pos_type == "det":
            analyses = morf.analyse(lemma)
            for a in analyses:
                parsed_tag = parse_tag(a[2][2])
                if 'adj' in parsed_tag[0] and any(case in parsed_tag[2] for case in ['nom', 'acc', 'gen']):
                    return True
            return False
        if pos_type == "adj":
            # Check if it is a real adjective (not cardinal numeral)
            # Cardinal numerals (one, two...) are tagged as adj, but also have num
            analyses = morf.analyse(lemma)
            has_num: bool = False
            has_adj: bool = False
            for a in analyses:
                parsed_tag = parse_tag(a[2][2])
                if "adj" in parsed_tag[0]:
                    has_adj = True
                if "num" in parsed_tag[0]:
                    has_num = True
            # Reject if main interpretation is a numeral
            if has_num:
                return False
            return has_adj
    except:
        pass
    return False


def get_corpus_words():
    if 'tokens_df' not in st.session_state or st.session_state.tokens_df is None:
        return None, None, None, None, None

    df = st.session_state.tokens_df
    nouns = [n for n in df[df['pos'] == 'NOUN']['lemma'].value_counts().index if validate_word(n, "noun")]
    verbs = [v for v in df[df['pos'] == 'VERB']['lemma'].value_counts().index if validate_word(v, "verb")]
    adjs = [a for a in df[df['pos'] == 'ADJ']['lemma'].value_counts().index if validate_word(a, "adj")]
    advs = [a for a in df[df['pos'] == 'ADV']['lemma'].value_counts().index if validate_word(a, "adv")]
    dets = [a for a in df[df['pos'].isin(['DET', 'PRON'])]['lemma'].value_counts().index if validate_word(a, "det")]
    
    return nouns, verbs, adjs, advs, dets


if 'tree_corpus' not in st.session_state:
    st.session_state.tree_corpus = None
if st.session_state.tree_corpus is None and 'tokens_df' in st.session_state:
    # Try loading from cache
    if os.path.exists(WORDS_CACHE_PATH):
        try:
            with open(WORDS_CACHE_PATH, "rb") as f:
                st.session_state.tree_corpus = pickle.load(f)
        except:
            pass
    # If not cached, build from tokens_df
    if st.session_state.tree_corpus is None:
        with st.spinner("Loading the words from text corpus"):
            st.session_state.tree_corpus = get_corpus_words()
            with open(WORDS_CACHE_PATH, "wb") as f:
                pickle.dump(st.session_state.tree_corpus, f)

corpus = st.session_state.tree_corpus
assert corpus is not None
_nouns, _verbs, _adjs, _advs, _dets = corpus
assert _nouns is not None and _verbs is not None and _adjs is not None and _advs is not None and _dets is not None
all_nouns: list[str] = _nouns
all_verbs: list[str] = _verbs
all_adjs: list[str] = _adjs
all_advs: list[str] = _advs
all_dets: list[str] = _dets

with st.sidebar:
    st.header("Settings")

    n_nouns = st.slider("Number of nouns:", 20, len(all_nouns), min(500, len(all_nouns)), 10)
    n_verbs = st.slider("Number of verbs:", 20, len(all_verbs), min(500, len(all_verbs)), 10)
    n_adjs = st.slider("Number of adjectives:", 10, len(all_adjs), min(500, len(all_adjs)), 5)
    st.caption(f"Available: {len(all_verbs)} verbs, {len(all_nouns)} nouns, {len(all_adjs)} adjs.")

    st.markdown("---")
    st.subheader("Bigram Cache")
    # Bigram cache status
    if os.path.exists(BIGRAM_CACHE_PATH):
        cache_size = os.path.getsize(BIGRAM_CACHE_PATH) / (1024 * 1024)  # MB
        st.success(f"Cache saved ({cache_size:.1f} MB)")
    else:
        st.warning("No cache - click 'Check frequency' to build")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Rebuild", help="Rebuild bigram index"):
            st.session_state.pop('bigram_cache', None)
            if os.path.exists(BIGRAM_CACHE_PATH):
                os.remove(BIGRAM_CACHE_PATH)
            # Clear Streamlit cache
            build_bigram_index.clear()
            st.rerun()

# Trim word lists
NOUNS = all_nouns[:n_nouns]
VERBS = all_verbs[:n_verbs]
ADJS = all_adjs[:n_adjs]
ZAIMKI = ["ja", "ty", "on", "ona", "ono", "my", "wy", "oni", "one"]
PRZYPADKI = ["nom", "gen", "dat", "acc", "inst", "loc"]
PRZYPADKI_NAZWY = {
    "nom": "nominative",
    "gen": "genitive",
    "dat": "dative", 
    "acc": "accusative", 
    "inst": "instrumental", 
    "loc": "locative",
    "voc": "vocative"
}

# === PARTS OF SPEECH ===

# Indefinite pronouns (for objects) - dynamically generated
ZAIMKI_DOPELNIENIE = {
    "nom": ["ktoś", "coś", "nikt", "nic"],
    "gen": ["kogoś", "czegoś", "nikogo", "niczego"],
    "dat": ["komuś", "czemuś", "nikomu", "niczemu"],
    "acc": ["kogoś", "coś", "nikogo", "nic"],
    "inst": ["kimś", "czymś", "nikim", "niczym"],
    "loc": ["kimś", "czymś", "nikim", "niczym"],
}

# Demonstrative determiners - dynamically generated  
OKRESLNIKI_WSKAZUJACE = ['ten', 'tamten', 'taki', 'ów'] # all_dets

# Possessive determiners (closed class - these are fixed)
OKRESLNIKI_DZIERZAWCZE = ["mój", "twój", "jego", "jej", "nasz", "wasz", "ich"]

# Cardinal numerals (inflected)
LICZEBNIKI_GLOWNE = ["jeden", "dwa", "trzy", "cztery", "pięć", "sześć", "siedem", "osiem", "dziewięć", "dziesięć", "jedenaście", "dwanaście", "dwadzieścia", "sto"]
# Ordinal numerals (inflected like adjectives)
LICZEBNIKI_PORZADKOWE = ["pierwszy", "drugi", "trzeci", "czwarty", "piąty", "szósty", "siódmy", "ósmy", "dziewiąty", "dziesiąty"]

# Adverbs - dynamically generated from corpus
PRZYSLOWKI = all_advs
# Fallback if empty
if not PRZYSLOWKI:
    PRZYSLOWKI = ["szybko", "wolno", "dobrze", "źle", "cicho", "głośno", "wysoko", "nisko", "daleko", "blisko", "łatwo", "trudno", "często", "rzadko", "bardzo", "mało"]

# Modal and phase verbs - dynamically filtered from corpus
CZASOWNIKI_MODALNE = ["móc", "musieć", "chcieć", "potrafić", "umieć", "mieć", "wolno", "trzeba", "należy", "warto", "wypada", "powinien"]
CZASOWNIKI_FAZOWE = ["zacząć", "zaczynać", "kończyć", "skończyć", "przestać", "przestawać", "kontynuować", "rozpocząć", "rozpoczynać"]
CZASOWNIKI_ZLOZENIA = CZASOWNIKI_MODALNE + CZASOWNIKI_FAZOWE

# Prepositions requiring specific cases (preposition order)
PRZYIMKI_PRZYPADKI = {
    "gen": ["", "z", "od", "do", "dla", "bez", "u", "spod", "znad", "koło", "obok", "wzdłuż", "podczas"],
    "dat": ["", "ku", "dzięki", "wbrew", "przeciw"],
    "acc": ["", "na", "w", "za", "przez", "po", "o", "nad", "pod", "przed", "między"],
    "inst": ["", "z", "za", "nad", "pod", "przed", "między"],
    "loc": ["", "w", "na", "o", "przy", "po"],
}

# Prepositions for objects (by case)
PRZYIMKI_DOPELNIENIE = {
    'gen': ['', 'z', 'od', 'do', 'dla', 'bez'],
    'dat': ['', 'ku'],
    'acc': ['', 'na', 'za', 'przez', 'o'],
    'inst': ['', 'z', 'za', 'nad', 'pod', 'przed'],
    'loc': ['', 'o', 'w', 'na', 'przy'],
}

# Adverbial types with their typical constructions (case + prepositions)
OKOLICZNIK_TYPY = {
    "miejsca_gdzie": {
        "label": "place (where?)",
        "case": "loc",
        "preps": ["w", "na", "przy", "obok", "koło"],
    },
    "miejsca_skad": {
        "label": "place (from where?)",
        "case": "gen",
        "preps": ["z", "od", "spod", "znad"],
    },
    "miejsca_dokad": {
        "label": "place (to where?)",
        "case": "gen",  # do + gen
        "preps": ["do"],
        # na/w + acc
        "alt": {"case": "acc", "preps": ["na", "w", "pod", "nad", "za"]},
    },
    "czasu_kiedy": {
        "label": "time (when?)",
        "case": "loc",
        "preps": ["w", "po", "przy"],
        "alt": {"case": "acc", "preps": ["w", "na", "o"]},
    },
    "czasu_jak_dlugo": {
        "label": "time (how long?)",
        "case": "acc",
        "preps": ["przez", ""],
    },
    "sposobu": {
        "label": "manner (how?)",
        "type": "adverb",  # adverb - szybko, wolno, dobrze
        "examples": ["szybko", "wolno", "dobrze", "źle", "cicho", "głośno", "łatwo", "trudno"],
    },
    "sposobu_przyimkowy": {
        "label": "manner prepositional (with joy)",
        "case": "inst",
        "preps": ["z", ""],
        "alt": {"case": "gen", "preps": ["bez"]},
    },
    "celu": {
        "label": "purpose (what for?)",
        "case": "gen",
        "preps": ["dla"],
        "alt": {"case": "acc", "preps": ["na", "po"]},
    },
    "przyczyny": {
        "label": "cause (why?)",
        "case": "gen",
        "preps": ["z"],
        "alt": {"case": "acc", "preps": ["przez"]},
    },
    "stopnia": {
        "label": "degree (how much?)",
        "type": "adverb",  # uses adverbs instead of nouns
        "examples": ["bardzo", "całkiem", "zupełnie", "trochę", "dużo", "mało"],
    },
    "warunku": {
        "label": "condition (under what condition?)",
        "case": "gen",
        "preps": ["bez"],
        "alt": {"case": "loc", "preps": ["przy", "w"]},
        "note": "full condition realized by subordinate clause (if...)",
    },
    "przyzwolenia": {
        "label": "concession (despite what?)",
        "case": "acc",
        "preps": ["mimo"],
        "alt": {"case": "dat", "preps": ["wbrew"]},
    },
    "imiesłowowy_współczesny": {
        "label": "contemporary participle (walking)",
        "type": "participle",
        "participle": "pcon",  # czytając, idąc
    },
    "imiesłowowy_uprzedni": {
        "label": "anterior participle (having read)",
        "type": "participle",
        "participle": "pant",  # przeczytawszy, zrobiwszy
    },
}

OSOBA_MAP = {
    "ja": ("1", "sing", "m1"), "ty": ("2", "sing", "m1"), "on": ("3", "sing", "m1"),
    "ona": ("3", "sing", "f"), "ono": ("3", "sing", "n"), "my": ("1", "plur", "m1"),
    "wy": ("2", "plur", "m1"), "oni": ("3", "plur", "m1"), "one": ("3", "plur", "f")
}

# === WALENTY - VERB VALENCY ===
@st.cache_data
def load_walenty_verb_cases():
    """Load verb valency from Walenty dictionary (IPI PAN)."""
    walenty_path = Path(config['semantics']['walenty_path'])

    if not os.path.exists(walenty_path):
        return {}

    verb_cases = {}

    try:
        with open(walenty_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('%') or not line.strip():
                    continue

                parts = line.split(':')
                if len(parts) < 6:
                    continue

                lemma = parts[0].strip()
                opinion = parts[1].strip()

                if opinion not in ['pewny', 'wątpliwy']:
                    continue

                if lemma not in verb_cases:
                    verb_cases[lemma] = { 'gen': 0, 'dat': 0, 'acc': 0, 'inst': 0, 'loc': 0 }

                schema = ':'.join(parts[5:]).strip()

                if re.search(r'subj\{lex\(', schema):
                    continue

                arguments = schema.split(' + ')

                # Very simple Walenty parsing (there might be room for improvement)
                for arg in arguments:
                    arg = arg.strip()
                    # Skip subject and obj (this is already main accusative)
                    if arg.startswith('subj'):
                        continue
                    if arg.startswith(('{lex(', '{cp(', '{ncp(', '{or(', '{nonch', '{xp(', '{advp(')):
                        continue

                    # controllee{np(inst)} - action result
                    if 'controllee{np(inst)}' in arg or 'controllee{adjp(inst)}' in arg:
                        # priorytet dla konstrukcji wynikowej
                        verb_cases[lemma]['inst'] += 2

                    # obj - main object (accusative)
                    if arg.startswith('obj') and 'np(str)' in arg:
                        verb_cases[lemma]['acc'] += 1

                    # Prepositional and case arguments
                    if 'np(str)' in arg and ';ncp(' not in arg and not arg.startswith('obj'):
                        verb_cases[lemma]['acc'] += 1
                    if 'np(gen)' in arg and ';ncp(' not in arg:
                        verb_cases[lemma]['gen'] += 1
                    if 'np(dat)' in arg and ';ncp(' not in arg:
                        verb_cases[lemma]['dat'] += 1
                    if 'np(inst)' in arg and ';ncp(' not in arg:
                        verb_cases[lemma]['inst'] += 1
                    if 'prepnp(o,loc)' in arg:
                        verb_cases[lemma]['loc'] += 1
                    # prepnp(z,gen) - "uczynić Z króla" (make OF king)
                    if 'prepnp(z,gen)' in arg:
                        verb_cases[lemma]['gen'] += 1
    except:
        pass

    return verb_cases


if 'verb_cases_cache' not in st.session_state:
    st.session_state.verb_cases_cache = None
if st.session_state.verb_cases_cache is None:
    with st.spinner("Loading Walenty dictionary..."):
        st.session_state.verb_cases_cache = load_walenty_verb_cases()

def get_verb_cases(verb):
    """Returns list of cases for a verb based on Walenty."""

    verb_data = st.session_state.verb_cases_cache.get(verb, {})
    case_info = [
        ('acc', "accusative"),
        ('gen', "genitive"),
        ('dat', "dative"),
        ('inst', "instrumental"),
        ('loc', "locative"),
    ]

    # Sort by frequency in Walenty
    sorted_cases = sorted(case_info, key=lambda x: verb_data.get(x[0], 0), reverse=True)

    result = []
    for case_code, case_name in sorted_cases:
        count = verb_data.get(case_code, 0)
        if count > 0:
            result.append((case_code, f"{case_name} [{count}x]"))

    # If no data - default to accusative
    if not result:
        result.append(('acc', "accusative (default)"))

    return result

# === TREE STRUCTURE ===
def new_node(node_type, **kwargs):
    """Tworzy nowy węzeł drzewa."""
    return {
        'id': str(uuid4())[:8],
        'type': node_type,
        'children': [],
        **kwargs
    }


def init_tree():
    """Initializes sentence tree."""
    return {
        'podmiot': new_node('podmiot', word_type='zaimek', word='ja', number='sing'),
        'orzeczenie': new_node('orzeczenie', word=VERBS[0], tense='praes', mood='ind', voice='czynna', negation=False),
    }


if 'syntax_tree' not in st.session_state:
    st.session_state.syntax_tree = init_tree()

tree = st.session_state.syntax_tree

# If verb is not in current VERBS - reset
if tree['orzeczenie']['word'] not in VERBS:
    tree['orzeczenie']['word'] = VERBS[0]

# === NODE RENDERING ===
def get_okreslnik_form(lemma, case, gender, number="sg"):
    """Inflects modifier (demonstrative/possessive) by cases."""
    # Uninflected: jego, jej, ich
    if lemma in ["jego", "jej", "ich"]:
        return lemma
    # Inflected - use Morfeusz
    try:
        for form, _, tag, *_ in morf.generate(lemma):
            parsed_tag = parse_tag(tag)
            if not any(number in x for x in parsed_tag):
                continue
            if not _match_case_in_tag(tag, case):
                continue
            if len(parsed_tag) < 4:
                continue
            g_options = parsed_tag[3]

            # In plural: m1 = virile, m2/m3/f/n = non-virile
            # We must match gender exactly, not accept any masculine
            if number == "pl":
                # m1 (virile) requires exact match
                if gender == "m1" and "m1" in g_options:
                    return form
                # m2, m3, f, n (non-virile) - can be together in one tag
                if gender in ["m2", "m3"] and any(x in g_options for x in ["m2", "m3", "n"]):
                    # Ensure it is NOT virile form (only m1)
                    if "m1" not in g_options or any(x in g_options for x in ["m2", "m3", "f", "n"]):
                        return form
                if gender == "f" and "f" in g_options:
                    return form
                if gender == "n" and "n" in g_options:
                    return form
            else:
                # In singular - old logic
                if gender.startswith("m") and any(x in g_options for x in ["m1", "m2", "m3"]):
                    return form
                if gender == "f" and "f" in g_options:
                    return form
                if gender == "n" and "n" in g_options:
                    return form
    except:
        pass
    return lemma


def render_okreslnik(node, parent_id, exclude_possessive=False):
    """Renders a modifier (adjective, demonstrative, possessive, participle).

    Args:
        node: Modifier node
        parent_id: Parent node ID
        exclude_possessive: True to hide possessive option (e.g., in passive voice)
    """
    c1, c2, c3, c4 = st.columns([2, 3, 2, 1])

    with c1:
        # Rodzaje okoliczników
        okr_typy = ["przymiotnik", "wskazujący", "dzierżawczy", "rzeczowny", "liczebnik", "przyimkowy", "imiesłów_czynny", "imiesłów_bierny"]

        # Exclude possessive in passive voice (agent "by X" doesn't have possessive pronoun)
        if exclude_possessive:
            okr_typy = [t for t in okr_typy if t != "dzierżawczy"]

        okr_labels: dict[str, str] = {
            "przymiotnik": "adjective (what kind?)",
            "wskazujący": "demonstrative (which?)",
            "dzierżawczy": "possessive (whose?)",
            "rzeczowny": "noun adjunct (father's house)",
            "liczebnik": "numeral (how many?)",
            "przyimkowy": "prepositional (in hat)",
            "imiesłów_czynny": "active participle (reading)",
            "imiesłów_bierny": "passive participle (read)",
        }
        current_typ = node.get('okr_typ', 'przymiotnik')
        if current_typ not in okr_typy:
            current_typ = 'przymiotnik'
        node['okr_typ'] = st.selectbox("Type:", okr_typy,
                                       format_func=lambda x: okr_labels.get(x, x) or x,
                                       index=okr_typy.index(current_typ), key=f"okr_typ_{node['id']}")

    with c2:
        if node['okr_typ'] == 'przymiotnik':
            options = [""] + ADJS
            current = node.get('word', '')
            idx = (ADJS.index(current) + 1) if current in ADJS else 0
            node['word'] = st.selectbox("Word:", options, index=idx, key=f"okr_word_{node['id']}")
        elif node['okr_typ'] == 'wskazujący':
            options = OKRESLNIKI_WSKAZUJACE
            current = node.get('word', 'ten')
            idx = options.index(current) if current in options else 0
            node['word'] = st.selectbox("Word:", options, index=idx, key=f"okr_word_{node['id']}")
        elif node['okr_typ'] == 'dzierżawczy':
            options = OKRESLNIKI_DZIERZAWCZE
            current = node.get('word', 'mój')
            idx = options.index(current) if current in options else 0
            node['word'] = st.selectbox("Word:", options, index=idx, key=f"okr_word_{node['id']}")
        elif node['okr_typ'] == 'rzeczowny':
            # Noun adjunct - noun in genitive (dom ojca)
            noun_opts = NOUNS if NOUNS else ["rzecz"]
            current = node.get('word', noun_opts[0])
            idx = noun_opts.index(current) if current in noun_opts else 0
            node['word'] = st.selectbox("Noun:", noun_opts, index=idx, key=f"okr_word_{node['id']}")
            node['case'] = 'gen'  # noun adjunct always genitive
        elif node['okr_typ'] == 'liczebnik':
            # Numeral adjunct
            num_opts = LICZEBNIKI_GLOWNE + LICZEBNIKI_PORZADKOWE
            current = node.get('word', 'jeden')
            idx = num_opts.index(current) if current in num_opts else 0
            node['word'] = st.selectbox("Numeral:", num_opts, index=idx, key=f"okr_word_{node['id']}")
        elif node['okr_typ'] == 'przyimkowy':
            # Prepositional adjunct (człowiek w kapeluszu)
            noun_opts = NOUNS if NOUNS else ["rzecz"]
            current = node.get('word', noun_opts[0])
            idx = noun_opts.index(current) if current in noun_opts else 0
            node['word'] = st.selectbox("Noun:", noun_opts, index=idx, key=f"okr_word_{node['id']}")
        elif node['okr_typ'] in ['imiesłów_czynny', 'imiesłów_bierny']:
            # Adjectival participle - verb selection
            verb_opts = VERBS if VERBS else ["robić"]
            current = node.get('verb_lemma', verb_opts[0])
            idx = verb_opts.index(current) if current in verb_opts else 0
            node['verb_lemma'] = st.selectbox("Verb:", verb_opts, index=idx, key=f"okr_verb_{node['id']}")
            node['word'] = node['verb_lemma']  # for compatibility

            # Check aspect and warn user
            if node['verb_lemma']:
                aspects = get_verb_aspects(node['verb_lemma'])
                if node['okr_typ'] == 'imiesłów_czynny' and 'imperf' not in aspects:
                    st.warning(f"Perfective verb '{node['verb_lemma']}' has no active participle!")
                elif node['okr_typ'] == 'imiesłów_bierny':
                    # Most verbs have ppas, but some intransitive ones don't
                    participle = get_participle(node['verb_lemma'], 'ppas', 'm1', 'sg', 'nom')
                    if not participle:
                        st.warning(f"Verb '{node['verb_lemma']}' has no passive participle!")

    with c3:
        if node['okr_typ'] == 'przymiotnik' and node.get('word'):
            # Check if adjective has comparison degrees
            if has_adj_comparison(node['word']):
                node['degree'] = st.selectbox("Degree:", ["pos", "com", "sup"],
                                              format_func=lambda x: {"pos": "positive", "com": "comparative", "sup": "superlative"}[x],
                                              key=f"deg_{node['id']}")
            else:
                node['degree'] = 'pos'
                st.caption("No grading is available")
        elif node['okr_typ'] == 'przyimkowy':
            # Select preposition and case for prepositional adjunct
            case_opts = ["gen", "loc", "inst"]
            current_case = node.get('case', 'loc')
            if current_case not in case_opts:
                current_case = 'loc'
            node['case'] = st.selectbox("Case:", case_opts,
                                        format_func=lambda x: {"gen": "Gen.", "loc": "Loc.", "inst": "Inst."}[x],
                                        index=case_opts.index(current_case),
                                        key=f"okr_case_{node['id']}")
        else:
            node['degree'] = 'pos'
            if node['okr_typ'] == 'wskazujący':
                st.caption("near/far" if node.get('word') in ['ten', 'tamten'] else "")

    with c4:
        # For prepositional adjunct - select preposition
        if node['okr_typ'] == 'przyimkowy':
            preps = PRZYIMKI_PRZYPADKI.get(node.get('case', 'loc'), ["w"])
            preps = [p for p in preps if p]  # remove empty
            if not preps:
                preps = ["w"]
            current_prep = node.get('preposition', preps[0])
            if current_prep not in preps:
                current_prep = preps[0]
            node['preposition'] = st.selectbox("Prep:", preps,
                                               index=preps.index(current_prep),
                                               key=f"okr_prep_{node['id']}")
        else:
            if st.button("X", key=f"del_{node['id']}"):
                return None

    # Separate remove button for prepositional (because c4 is occupied)
    if node['okr_typ'] == 'przyimkowy':
        if st.button("Remove adjunct", key=f"del_przydawka_{node['id']}"):
            return None

    return node


def render_dopelnienie(node, verb_lemma, voice="czynna"):
    """Renders object with optional preposition."""

    # Check removal AT START (before rendering anything)
    delete_key = f"del_dop_{node['id']}"
    if st.session_state.get(delete_key):
        return None

    with st.container():
        # Header with remove button
        col_header, col_del = st.columns([6, 1])
        with col_header:
            st.markdown(f"#### Object")
        with col_del:
            if st.button("Remove", key=delete_key, help="Remove object"):
                st.rerun()

        # Passive voice forces "przez + accusative"
        if voice == "bierna":
            st.info("Passive voice: object = agent (przez + accusative)")
            node['case'] = 'acc'
            node['preposition'] = 'przez'

        # Get cases from Walenty
        verb_cases = get_verb_cases(verb_lemma)
        case_codes = [c[0] for c in verb_cases]
        case_labels = {c[0]: c[1] for c in verb_cases}

        # Object type: noun or indefinite pronoun
        dop_typ = node.get('dop_typ', 'rzeczownik')
        node['dop_typ'] = st.radio("Type:", ["rzeczownik", "zaimek nieokreślony"],
                                   index=0 if dop_typ == 'rzeczownik' else 1, horizontal=True, key=f"dop_typ_{node['id']}")

        if node['dop_typ'] == 'zaimek nieokreślony':
            # Indefinite pronoun - case, preposition and pronoun
            c1, c2, c3 = st.columns(3)
            with c1:
                current_case = node.get('case', 'acc')
                if voice != "bierna":
                    if current_case not in case_codes:
                        current_case = case_codes[0]
                    idx = case_codes.index(current_case)
                    node['case'] = st.selectbox("Case:", case_codes,
                                                format_func=lambda x: case_labels.get(x, PRZYPADKI_NAZWY.get(x, x)),
                                                index=idx, key=f"dop_case_{node['id']}")
                else:
                    node['case'] = 'acc'
            with c2:
                # Preposition for indefinite pronoun (e.g. "o kimś", "za czymś")
                if voice != "bierna":
                    available_preps = PRZYIMKI_DOPELNIENIE.get(node['case'], [""])
                    current_prep = node.get('preposition', '')
                    if current_prep not in available_preps:
                        current_prep = available_preps[0]
                    idx = available_preps.index(current_prep)
                    node['preposition'] = st.selectbox("Preposition:", available_preps,
                                                       format_func=lambda x: "(none)" if x == "" else x, index=idx, key=f"dop_prep_{node['id']}")
                else:
                    node['preposition'] = 'przez'
                    st.caption("przez (forced)")
            with c3:
                zaimki_list = ZAIMKI_DOPELNIENIE.get(node['case'], ["coś"])
                current_zaimek = node.get('zaimek', zaimki_list[0])
                if current_zaimek not in zaimki_list:
                    current_zaimek = zaimki_list[0]
                idx = zaimki_list.index(current_zaimek)
                node['zaimek'] = st.selectbox("Pronoun:", zaimki_list, index=idx, key=f"dop_zaimek_{node['id']}")
            node['number'] = 'sg'
        else:
            # Noun - full options
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                idx = NOUNS.index(node.get('word', NOUNS[0])) if node.get('word', NOUNS[0]) in NOUNS else 0
                node['word'] = st.selectbox("Noun:", NOUNS, index=idx, key=f"dop_word_{node['id']}")
            with c2:
                # Use cases from Walenty (unless passive voice)
                if voice != "bierna":
                    current_case = node.get('case', 'acc')
                    if current_case not in case_codes:
                        current_case = case_codes[0]
                    idx = case_codes.index(current_case)
                    node['case'] = st.selectbox("Case:", case_codes,
                                                format_func=lambda x: case_labels.get(x, PRZYPADKI_NAZWY.get(x, x)),
                                                index=idx, key=f"dop_case_{node['id']}")
            with c3:
                # Prepositions filtered by case (unless passive voice)
                if voice != "bierna":
                    available_preps = PRZYIMKI_DOPELNIENIE.get(node['case'], [""])
                    current_prep = node.get('preposition', '')
                    if current_prep not in available_preps:
                        current_prep = available_preps[0]
                    idx = available_preps.index(current_prep)
                    node['preposition'] = st.selectbox("Preposition:", available_preps,
                                                       format_func=lambda x: "(none)" if x == "" else x, index=idx, key=f"dop_prep_{node['id']}")
                else:
                    st.caption("przez (forced)")
            with c4:
                node['number'] = st.selectbox("Number:", ["sg", "pl"],
                                              format_func=lambda x: "sg." if x == "sg" else "pl.", key=f"dop_num_{node['id']}")

        # Modifiers for object (noun only)
        if node['dop_typ'] == 'rzeczownik':
            # Cannot use possessive in passive voice (agent "by X")
            is_passive = (voice == "bierna")
            if node.get('children'):
                st.markdown("*Modifiers:*")
                render_children_inline(node, is_passive_object=is_passive)

            # Button to add modifier
            if st.button("Add Modifier", key=f"add_okr_dop_{node['id']}"):
                if 'children' not in node:
                    node['children'] = []
                node['children'].append(new_node('okreslnik', word='', okr_typ='przymiotnik', degree='pos'))
                st.rerun()

    return node


def render_okolicznik(node):
    """Renders adverbial with semantic type."""
    with st.container():
        # Select adverbial type
        typ_keys = list(OKOLICZNIK_TYPY.keys())
        typ_labels = {k: v['label'] for k, v in OKOLICZNIK_TYPY.items()}

        current_typ = node.get('okolicznik_typ', 'miejsca_gdzie')
        if current_typ not in typ_keys:
            current_typ = 'miejsca_gdzie'

        col_header, col_del = st.columns([4, 1])
        with col_header:
            node['okolicznik_typ'] = st.selectbox("Adverbial Type:", typ_keys,
                                                  format_func=lambda x: typ_labels[x],
                                                  index=typ_keys.index(current_typ), key=f"okol_typ_{node['id']}")
        with col_del:
            st.write("")
            if st.button("Remove", key=f"del_okol_{node['id']}"):
                return None

        # Get config for selected type
        typ_config = OKOLICZNIK_TYPY[node['okolicznik_typ']]

        # Degree/measure adverbial - uses adverbs instead of nouns
        if typ_config.get('type') == 'adverb':
            col1, col2 = st.columns([3, 1])
            with col1:
                # List of adverbs from examples + general list
                adv_options = typ_config.get('examples', []) + [a for a in PRZYSLOWKI if a not in typ_config.get('examples', [])]
                current_adv = node.get('adverb', adv_options[0] if adv_options else '')
                idx = adv_options.index(current_adv) if current_adv in adv_options else 0
                node['adverb'] = st.selectbox("Adverb:", adv_options, index=idx, key=f"okol_adv_{node['id']}")
            with col2:
                # Adverb degree
                node['degree'] = st.selectbox("Degree:", ["pos", "com", "sup"],
                                              format_func=lambda x: {"pos": "positive", "com": "comparative", "sup": "superlative"}[x],
                                              key=f"okol_deg_{node['id']}")
            return node

        # Participle adverbial - adverbial participle (reading, having read)
        if typ_config.get('type') == 'participle':
            participle_type = typ_config.get('participle', 'pcon')
            col1, col2 = st.columns([3, 1])
            with col1:
                verb_opts = VERBS if VERBS else ["robić"]
                current_verb = node.get('verb_lemma', verb_opts[0])
                idx = verb_opts.index(current_verb) if current_verb in verb_opts else 0
                node['verb_lemma'] = st.selectbox("Verb:", verb_opts, index=idx, key=f"okol_verb_{node['id']}")
            with col2:
                st.caption(f"{'contemporary' if participle_type == 'pcon' else 'anterior'}")
                # Preview form
                participle_form = get_participle(node['verb_lemma'], participle_type)
                if participle_form:
                    st.success(f"-> {participle_form}")
                else:
                    st.warning("no form")
            node['participle_type'] = participle_type
            return node

        # Collect all available combinations (case, prep) for this type
        options = []
        for prep in typ_config['preps']:
            options.append((typ_config['case'], prep))
        if 'alt' in typ_config:
            for prep in typ_config['alt']['preps']:
                options.append((typ_config['alt']['case'], prep))

        # Select construction (which determines case)
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            idx = NOUNS.index(node.get('word', NOUNS[0])) if node.get('word', NOUNS[0]) in NOUNS else 0
            node['word'] = st.selectbox("Noun:", NOUNS, index=idx, key=f"okol_word_{node['id']}")
        with c2:
            # Show prepositions with case info
            def format_option(opt):
                case, prep = opt
                case_short = {"gen": "Gen.", "dat": "Dat.", "acc": "Acc.", "inst": "Inst.", "loc": "Loc."}[case]
                if prep:
                    return f"{prep} + {case_short}"
                return f"(none) + {case_short}"

            current_opt = (node.get('case', typ_config['case']), node.get('preposition', typ_config['preps'][0]))
            if current_opt not in options:
                current_opt = options[0]
            idx = options.index(current_opt)

            selected = st.selectbox("Construction:", options, format_func=format_option, index=idx, key=f"okol_konstr_{node['id']}")
            if selected is not None:
                node['case'] = selected[0]
                node['preposition'] = selected[1]
        with c3:
            node['number'] = st.selectbox("Number:", ["sg", "pl"],
                                          format_func=lambda x: "sg." if x == "sg" else "pl.",
                                          key=f"okol_num_{node['id']}")

        # Modifiers
        if node.get('children'):
            st.markdown("*Modifiers:*")
            render_children_inline(node)

        if st.button("Add Modifier", key=f"add_okr_okol_{node['id']}"):
            if 'children' not in node:
                node['children'] = []
            node['children'].append(new_node('okreslnik', word='', okr_typ='przymiotnik', degree='pos'))
            st.rerun()

    return node


def render_children_inline(parent_node, is_passive_object=False):
    """Renders node children (modifiers) inline.

    Args:
        parent_node: Parent node
        is_passive_object: True if this is object in passive voice (excludes possessive)
    """
    children_to_keep = []
    for child in parent_node.get('children', []):
        if child['type'] in ['przydawka', 'okreslnik']:
            result = render_okreslnik(child, parent_node['id'], exclude_possessive=is_passive_object)
            if result is not None:
                children_to_keep.append(result)
    parent_node['children'] = children_to_keep

# === MAIN INTERFACE ===


col_left, col_right = st.columns(2)

# SUBJECT
with col_left:
    st.subheader("Subject")

    subj = tree['podmiot']
    subj['word_type'] = st.radio("Type:", ["zaimek", "rzeczownik"], horizontal=True, key="subj_type")

    if subj['word_type'] == "zaimek":
        subj['word'] = st.selectbox("Pronoun:", ZAIMKI, key="subj_word")
        # Gender selection for 1st/2nd person (ja, ty, my, wy) - affects verb inflection in past/conditional
        if subj['word'] in ["ja", "ty"]:
            gender_opts = ["m1", "f"]
            current_gender = subj.get('gender_override', 'm1')
            if current_gender not in gender_opts:
                current_gender = 'm1'
            subj['gender_override'] = st.radio("Gender:", gender_opts,
                                               format_func=lambda x: "masculine" if x == "m1" else "feminine",
                                               index=gender_opts.index(current_gender), horizontal=True, key="subj_gender")
        elif subj['word'] in ["my", "wy"]:
            gender_opts = ["m1", "f"]  # m1 = masculine personal, f = other
            current_gender = subj.get('gender_override', 'm1')
            if current_gender not in gender_opts:
                current_gender = 'm1'
            subj['gender_override'] = st.radio("Gender:", gender_opts,
                                               format_func=lambda x: "masculine personal" if x == "m1" else "other",
                                               index=gender_opts.index(current_gender), horizontal=True, key="subj_gender")
        else:
            # on/ona/ono/oni/one have fixed gender
            subj.pop('gender_override', None)
        # Personal pronouns don't have modifiers
        subj['children'] = []
    else:
        idx = NOUNS.index(subj.get('word', NOUNS[0])) if subj.get('word', NOUNS[0]) in NOUNS else 0
        subj['word'] = st.selectbox("Noun:", NOUNS, index=idx, key="subj_noun")
        subj['number'] = st.radio("Number:", ["sing", "plur"],
                                  format_func=lambda x: "sg." if x == "sing" else "pl.",
                                  horizontal=True,
                                  key="subj_num")

        # Subject modifiers - only for noun
        with st.expander("Subject Modifiers", expanded=len(subj.get('children', [])) > 0):
            render_children_inline(subj)
            if st.button("Add Modifier", key="add_subj_okr"):
                if 'children' not in subj:
                    subj['children'] = []
                subj['children'].append(new_node('okreslnik', word='', okr_typ='przymiotnik', degree='pos'))
                st.rerun()

# PREDICATE
with col_right:
    st.subheader("Predicate")
    verb = tree['orzeczenie']

    # Predicate type: simple vs compound
    verb['compound'] = st.checkbox("Compound predicate (modal/phasal + infinitive)",
                                   value=verb.get('compound', False), key="verb_compound")

    if verb['compound']:
        # Compound predicate: modal/phasal verb + infinitive
        c_modal, c_inf = st.columns(2)
        with c_modal:
            modal_opts = CZASOWNIKI_ZLOZENIA
            current_modal = verb.get('modal_verb', 'móc')
            if current_modal not in modal_opts:
                current_modal = 'móc'
            verb['modal_verb'] = st.selectbox("Modal/Phasal Verb:", modal_opts,
                                              index=modal_opts.index(current_modal), key="verb_modal")
        with c_inf:
            # Second verb (can be phasal or main)
            c_verbs = CZASOWNIKI_ZLOZENIA + (VERBS if VERBS else [])
            # Remove duplicates preserving order
            c_verbs = list(dict.fromkeys(c_verbs))
            current_inf = verb.get('word', 'robić')
            idx = c_verbs.index(current_inf) if current_inf in c_verbs else 0
            verb['word'] = st.selectbox("Second verb:", c_verbs, index=idx, key="verb_second")

        # For compound predicate: aspect and tense from modal verb
        verb['aspect'] = "niedokonany"  # modals are imperfective
        
        # Show preview
        st.caption(f"-> *{verb['modal_verb']}* + **{verb['word']}** (infinitive)")
        
        # Compound predicate limitations
        if verb.get('voice') == 'bierna':
            st.warning("Passive voice is not supported for compound predicates.")
    else:
        idx = VERBS.index(verb.get('word', VERBS[0])) if verb.get('word', VERBS[0]) in VERBS else 0
        verb['word'] = st.selectbox("Verb:", VERBS, index=idx, key="verb_word")

    aspects = get_verb_aspects(verb.get('modal_verb', verb['word']) if verb.get('compound') else verb['word'])
    if not verb.get('compound'):
        if 'perf' in aspects and 'imperf' in aspects:
            verb['aspect'] = "dwuaspektowy"
            st.success("Aspect: **bi-aspectual**")
        elif 'perf' in aspects and 'imperf' not in aspects:
            verb['aspect'] = "dokonany"
            st.info("Aspect: **perfective**")
        else:
            verb['aspect'] = "niedokonany"
            st.info("Aspect: **imperfective**")
        st.caption("Change lemma for different aspect")

    c1, c2 = st.columns(2)
    with c1:
        # Perfective: past, future simple (no present!)
        # Imperfective: present, past, future compound
        if verb['aspect'] == "dokonany":
            tense_opts = ["przeszły", "przyszły prosty"]
            st.caption("No present tense (perfective aspect)")
        else:
            tense_opts = ["teraźniejszy", "przeszły", "przyszły złożony"]
        verb['tense'] = st.selectbox("Tense:", tense_opts, key="verb_tense")
        verb['mood'] = st.selectbox("Mood:", ["orzekający", "przypuszczający", "rozkazujący"], key="verb_mood")
    with c2:
        verb['voice'] = st.radio("Voice:", ["czynna", "bierna"], horizontal=True, key="verb_voice")
        c2a, c2b = st.columns(2)
        with c2a:
            verb['negation'] = st.checkbox("Negation", key="verb_neg")
        with c2b:
            verb['question'] = st.checkbox("Question", key="verb_question")

    # Imperative validation
    if verb['mood'] == "rozkazujący":
        # Imperative works in all aspects but tense is irrelevant
        if verb['question']:
            st.warning("Imperative cannot be a question. Question option will be ignored.")
        st.info("Imperative mood ignores tense selection.")

# === PREDICATE ELEMENTS (objects, adverbials) ===

st.markdown("---")
st.subheader("Sentence Expansion")
st.caption("Object cases from Walenty Dictionary (IPI PAN)")

# Render existing objects and adverbials
verb_lemma = tree['orzeczenie']['word']
verb_voice = tree['orzeczenie'].get('voice', 'czynna')
children_to_keep = []
for child in tree['orzeczenie'].get('children', []):
    with st.container():
        st.markdown("---")
        if child['type'] == 'dopelnienie':
            result = render_dopelnienie(child, verb_lemma, verb_voice)
        elif child['type'] == 'okolicznik':
            result = render_okolicznik(child)
        else:
            result = child
        if result is not None:
            children_to_keep.append(result)

tree['orzeczenie']['children'] = children_to_keep

# Add buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Add Object"):
        tree['orzeczenie']['children'].append(new_node(
            'dopelnienie', word=NOUNS[0], case='acc', preposition='', number='sg'))
        st.rerun()
with col2:
    if st.button("Add Adverbial"):
        tree['orzeczenie']['children'].append(new_node(
            'okolicznik', word=NOUNS[0], case='loc', preposition='w', number='sg'))
        st.rerun()

# === SENTENCE GENERATION ===

st.markdown("---")
def generate_okreslnik_form(node, parent_case, parent_gender, parent_number):
    """Generates modifier form agreeing with parent."""
    word = node.get('word', '')
    if not word:
        return ""

    okr_typ = node.get('okr_typ', 'przymiotnik')

    if okr_typ == 'przymiotnik':
        return get_adj_form(word, parent_case, parent_gender, parent_number, node.get('degree', 'pos'))
    elif okr_typ in ['wskazujący', 'dzierżawczy']:
        # Demonstrative or possessive
        return get_okreslnik_form(word, parent_case, parent_gender, parent_number)
    elif okr_typ == 'rzeczowny':
        # Noun adjunct - always genitive
        return get_noun_form(word, 'gen', parent_number)
    elif okr_typ == 'liczebnik':
        # Numeral adjunct
        return get_numeral_form(word, parent_case, parent_gender, parent_number)
    elif okr_typ == 'przyimkowy':
        # Prepositional adjunct
        prep = node.get('preposition', 'w')
        case = node.get('case', 'loc')
        noun_form = get_noun_form(word, case, parent_number)
        return f"{prep} {noun_form}"
    elif okr_typ == 'imiesłów_czynny':
        # Active participle
        # NOTE: Only imperfective verbs
        verb_lemma = node.get('verb_lemma', word)
        participle = get_participle(verb_lemma, 'pact', parent_gender, parent_number, parent_case)
        if participle:
            return participle
        else:
            return f"[MISSING: {verb_lemma}->pact]"
    elif okr_typ == 'imiesłów_bierny':
        # Passive participle
        verb_lemma = node.get('verb_lemma', word)
        participle = get_participle(verb_lemma, 'ppas', parent_gender, parent_number, parent_case)
        if participle:
            return participle
        else:
            return f"[MISSING: {verb_lemma}->ppas]"

    return word


def get_numeral_form(lemma, case, gender, number='sg'):
    """Inflects numeral by cases.

    Numeral tag structure:
    - num:pl:nom.acc:m2.m3.f.n:congr (cardinal)
    - numcol:pl:nom.acc.voc:n:ncol (collective)
    """
    if not morf:
        return lemma

    try:
        forms = morf.generate(lemma)
        for form, _, tag, *_ in forms:
            parsed_tag = parse_tag(tag)
            # Match case correctly via split
            if not _match_case_in_tag(tag, case):
                continue
            # Match number
            if number is not None and not any(number in x for x in parsed_tag):
                continue
            # Numerals have specific inflection
            if 'num' in parsed_tag[0] or 'adj' in parsed_tag[0]:  # ordinal numerals are like adjectives
                if len(parsed_tag) >= 4:
                    g_options = parsed_tag[3]
                    # In plural: m1 = masculine personal, m2/m3/f/n = non-masculine personal
                    if number == 'pl':
                        if gender == "m1" and "m1" in g_options:
                            return form
                        if gender in ["m2", "m3"] and any(x in g_options for x in ["m2", "m3", "n"]):
                            if "m1" not in g_options or any(x in g_options for x in ["m2", "m3", "f", "n"]):
                                return form
                        if gender == "f" and "f" in g_options:
                            return form
                        if gender == "n" and "n" in g_options:
                            return form
                    else:
                        if gender.startswith("m") and any(x in g_options for x in ["m1", "m2", "m3"]):
                            return form
                        if gender == "f" and "f" in g_options:
                            return form
                        if gender == "n" and "n" in g_options:
                            return form
    except:
        pass
    return lemma


def get_adverb_form(adverb, degree='pos'):
    """Gets adverb form in appropriate degree."""
    if not morf:
        return adverb

    try:
        analyses = morf.analyse(adverb)
        for interp in analyses:
            lemma = interp[2][1]
            tag = interp[2][2]
            parsed_tag = parse_tag(tag)

            if 'adv' in parsed_tag[0]:
                # Look for form in appropriate degree
                forms = morf.generate(lemma)
                for form in forms:
                    form_tag = form[2]
                    fparsed_tag = parse_tag(form_tag)
                    if 'adv' in fparsed_tag[0] and degree in fparsed_tag[1]:
                        return form[0]
    except:
        pass

    return adverb


def generate_phrase(node, parent_case=None, parent_gender=None, parent_number=None, negation=False):
    """Generates phrase from tree node."""

    if node['type'] in ['przydawka', 'okreslnik']:
        if not node.get('word'):
            return ""
        return generate_okreslnik_form(node, parent_case or 'nom', parent_gender or 'm1', parent_number or 'sg')

    elif node['type'] == 'okolicznik':
        # Participial adverbial (czytając, przeczytawszy)
        if node.get('participle_type'):
            verb_lemma = node.get('verb_lemma', '')
            participle_type = node.get('participle_type')
            participle = get_participle(verb_lemma, participle_type)
            if participle:
                return participle
            else:
                return f"[BRAK: {verb_lemma}->{participle_type}]"

        # Adverbial of degree/measure
        if node.get('adverb'):
            degree = node.get('degree', 'pos')
            return get_adverb_form(node['adverb'], degree)

        # Noun adverbial (place, time, etc.)
        case = node.get('case', 'acc')
        number = node.get('number', 'sg')
        gender = get_noun_gender(node['word'])

        # Collect modifiers
        okr_parts = []
        for child in node.get('children', []):
            if child['type'] in ['przydawka', 'okreslnik']:
                okr = generate_phrase(child, case, gender, number, negation)
                if okr:
                    okr_parts.append(okr)

        noun_form = get_noun_form(node['word'], case, number)
        phrase = " ".join(okr_parts + [noun_form])

        if node.get('preposition'):
            phrase = node['preposition'] + " " + phrase

        return phrase

    elif node['type'] == 'dopelnienie':
        # Indefinite pronoun - already in correct form
        if node.get('dop_typ') == 'zaimek nieokreślony':
            zaimek = node.get('zaimek', 'coś')
            if node.get('preposition'):
                return node['preposition'] + " " + zaimek
            return zaimek

        # Noun
        case = node.get('case', 'acc')
        number = node.get('number', 'sg')

        # Negation forces genitive for accusative (if not passive voice)
        if negation and case == 'acc' and not node.get('preposition'):
            case = 'gen'

        gender = get_noun_gender(node['word'])

        # Collect modifiers
        okr_parts = []
        for child in node.get('children', []):
            if child['type'] in ['przydawka', 'okreslnik']:
                okr = generate_phrase(child, case, gender, number, negation)
                if okr:
                    okr_parts.append(okr)

        noun_form = get_noun_form(node['word'], case, number)
        phrase = " ".join(okr_parts + [noun_form])

        if node.get('preposition'):
            phrase = node['preposition'] + " " + phrase

        return phrase

    return ""


# Generate subject
subj = tree['podmiot']
verb = tree['orzeczenie']
mood_map = {"orzekający": "ind", "przypuszczający": "cond", "rozkazujący": "impt"}
is_imperative = mood_map.get(verb['mood']) == 'impt'

if subj['word_type'] == 'zaimek':
    person, number, p_gender = OSOBA_MAP[subj['word']]
    # Use overridden gender for ja/ty/my/wy (if set)
    if subj.get('gender_override'):
        p_gender = subj['gender_override']
    podmiot_form = subj['word']
    # Imperative mood - hide pronoun (rozkaż! not: ty rozkaż!)
    if is_imperative:
        podmiot_form = ""
else:
    p_gender = get_noun_gender(subj['word'])
    number = subj.get('number', 'sing')
    num_code = "sg" if number == "sing" else "pl"

    # Imperative mood - subject in VOCATIVE, verb in 2nd person
    if is_imperative:
        person = "2"  # imperative mood = 2nd person
        subj_case = "voc"  # vocative
    else:
        person = "3"
        subj_case = "nom"

    # Subject modifiers
    okr_parts = []
    for child in subj.get('children', []):
        if child['type'] in ['przydawka', 'okreslnik'] and child.get('word'):
            okr = generate_okreslnik_form(child, subj_case, p_gender, num_code)
            if okr:
                okr_parts.append(okr)

    noun_form = get_noun_form(subj['word'], subj_case, num_code)
    podmiot_form = " ".join(okr_parts + [noun_form])

# Generate predicate
tense_map = {
    "teraźniejszy": "praes",
    "przeszły": "praet",
    "przyszły prosty": "fut",  # perfective
    "przyszły złożony": "fut"  # imperfective
}

# Compound predicate: modal/phasal + infinitive
if verb.get('compound') and verb.get('modal_verb'):
    # Inflect modal/phasal verb
    modal_form = get_verb_form(
        verb['modal_verb'], person, number,
        tense_map.get(verb['tense'], 'praes'),
        mood_map.get(verb['mood'], 'ind'),
        p_gender, 'niedokonany', 'czynna'  # modals are imperfective and active
    )
    
    # Modal (inflected) + main verb (infinitive): mogę robić, chcę pisać
    verb_form = f"{modal_form} {verb['word']}" if modal_form else verb['word']
else:
    verb_form = get_verb_form(
        verb['word'], person, number,
        tense_map.get(verb['tense'], 'praes'),
        mood_map.get(verb['mood'], 'ind'),
        p_gender, verb.get('aspect', 'niedokonany'), verb['voice']
    )

if verb_form is None:
    mood = verb.get('mood', 'orzekający')
    voice = verb.get('voice', 'czynna')

    if mood == "rozkazujący":
        st.error(f"Imperative mood does not exist for this person ({person} pers. {'pl.' if number == 'plur' else 'sg.'})")
        st.info("Imperative exists only for: 2nd pers. sg., 1st pers. pl., 2nd pers. pl.")
    elif voice == "bierna":
        st.error(f"Passive voice unavailable for verb '{verb['word']}'")
        st.info("Not all verbs have passive participle (ppas). Try another verb.")
    else:
        st.error("This verb form does not exist for selected parameters")
    st.stop()

if verb['negation']:
    verb_form = "nie " + verb_form

# Generate objects and adverbials
is_negation = verb['negation']
extras = []
for child in tree['orzeczenie'].get('children', []):
    phrase = generate_phrase(child, negation=is_negation)
    if phrase:
        extras.append(phrase)

# Assemble sentence
is_question = verb.get('question', False)

if is_imperative:
    # Imperative mood: "Ręko, miej głowę!" or "Miej głowę!" (without subject)
    if podmiot_form and podmiot_form.strip():
        # Subject in vocative + comma + verb
        parts = [podmiot_form.capitalize() + ",", verb_form] + extras
    else:
        parts = [verb_form.capitalize()] + extras
    zdanie = " ".join(parts) + "!"
else:
    if is_question:
        # In question inverted order: Czy + subject + predicate...
        parts = ["Czy", podmiot_form, verb_form] + extras
        zdanie = " ".join(parts) + "?"
    else:
        parts = [podmiot_form.capitalize(), verb_form] + extras
        zdanie = " ".join(parts) + "."

st.markdown("### Generated Sentence:")
st.success(f"## {zdanie}")

# Tree visualization
with st.expander("Syntax Tree Structure", expanded=False):
    def tree_to_text(node, indent=0):
        prefix = "  " * indent + ("|-- " if indent > 0 else "")
        lines = []

        if node['type'] == 'podmiot':
            lines.append(f"SUBJECT: {node['word']}")
        elif node['type'] == 'orzeczenie':
            lines.append(f"PREDICATE: {node['word']} ({verb['tense']}, {verb['mood']}, {verb['voice']})")
        elif node['type'] == 'dopelnienie':
            prep = f"{node.get('preposition', '')} + " if node.get('preposition') else ""
            lines.append(f"{prefix}OBJECT: {prep}{node['word']} ({PRZYPADKI_NAZWY.get(node.get('case', 'acc'), '')})")
        elif node['type'] == 'okolicznik':
            prep = f"{node.get('preposition', '')} + " if node.get('preposition') else ""
            typ_label = OKOLICZNIK_TYPY.get(node.get('okolicznik_typ', 'miejsca_gdzie'), {}).get('label', '?')
            lines.append(f"{prefix}ADVERBIAL ({typ_label}): {prep}{node['word']} ({PRZYPADKI_NAZWY.get(node.get('case', ''), '')})")
        elif node['type'] in ['przydawka', 'okreslnik']:
            if node.get('word'):
                okr_typ = node.get('okr_typ', 'przymiotnik')
                if okr_typ == 'przymiotnik':
                    deg = {"pos": "", "com": " (comp.)", "sup": " (super.)"}.get(node.get('degree', 'pos'), '')
                    lines.append(f"{prefix}MODIFIER: {node['word']}{deg}")
                elif okr_typ == 'wskazujący':
                    lines.append(f"{prefix}DEMONSTRATIVE: {node['word']}")
                else:
                    lines.append(f"{prefix}POSSESSIVE: {node['word']}")

        for i, child in enumerate(node.get('children', [])):
            child_lines = tree_to_text(child, indent + 1)
            lines.extend(child_lines)

        return lines

    tree_text = []
    tree_text.append("SENTENCE")
    tree_text.extend(tree_to_text(tree['podmiot'], 1))
    tree_text.extend(tree_to_text(tree['orzeczenie'], 1))

    st.code("\n".join(tree_text))

# Grammatical Analysis
with st.expander("Grammatical Analysis", expanded=False):
    liczba_podmiotu = "plural" if number == "plur" else "singular"
    strona_label = verb['voice']
    aspekt_label = verb.get('aspect', 'niedokonany')

    # Build table
    table_rows = []
    table_rows.append("| Element | Value | Grammatical Features |")
    table_rows.append("|---------|-------|----------------------|")
    table_rows.append(f"| **Subject** | {podmiot_form} | {liczba_podmiotu}, person {person}, gender {p_gender} |")
    table_rows.append(f"| **Predicate** | {verb_form} | {verb['tense']}, {verb['mood']}, {aspekt_label}, {strona_label} |")

    # Objects
    for child in tree['orzeczenie'].get('children', []):
        if child['type'] == 'dopelnienie':
            if child.get('dop_typ') == 'zaimek nieokreślony':
                dop_value = child.get('zaimek', 'coś')
                dop_info = "indefinite pronoun"
            else:
                dop_value = child.get('word', '?')
                dop_gender = get_noun_gender(child['word'])
                dop_number = "pl." if child.get('number', 'sg') == 'pl' else "sg."
                dop_info = f"{dop_number}, gender {dop_gender}"
            prep_info = f"{child.get('preposition', '')} + " if child.get('preposition') else ""
            case_name = PRZYPADKI_NAZWY.get(child.get('case', 'acc'), child.get('case', 'acc'))
            table_rows.append(f"| **Object** | {dop_value} | {prep_info}{case_name}, {dop_info} |")
        elif child['type'] == 'okolicznik':
            okol_typ = OKOLICZNIK_TYPY.get(child.get('okolicznik_typ', 'miejsca_gdzie'), {}).get('label', '?')
            if child.get('adverb'):
                table_rows.append(f"| **Adverbial** | {child['adverb']} | {okol_typ}, adverb |")
            else:
                prep_info = f"{child.get('preposition', '')} + " if child.get('preposition') else ""
                case_name = PRZYPADKI_NAZWY.get(child.get('case', ''), '')
                table_rows.append(f"| **Adverbial** | {child.get('word', '?')} | {okol_typ}, {prep_info}{case_name} |")

    st.markdown("\n".join(table_rows))

# Check corpus stats
if st.button("Check collocation frequency in corpus", use_container_width=True):
    words = zdanie.replace("?", "").replace(".", "").replace("!", "").split()

    # Init cache
    if 'bigram_cache' not in st.session_state:
        # Try loading from file
        with st.spinner("Building bigram index (one-time, saving to cache)..."):
            token_idx, lemma_idx = build_bigram_index(st.session_state.tokens_df)
            st.session_state.bigram_cache = (token_idx, lemma_idx)

    st.markdown("### Corpus Collocation Assessment")
    st.caption("Frequent | Common | Occurs | Rare | Lemmas only | Missing")

    if len(words) >= 2 and 'bigram_cache' in st.session_state:
        bigram_data = []
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            l1, l2 = get_lemma(w1), get_lemma(w2)
            cnt_form, cnt_lemma = get_bigram_counts(w1, w2, l1, l2)
            emoji, desc = get_bigram_color(cnt_form, cnt_lemma)
            bigram_data.append((f"{w1} {w2}", l1, l2, emoji, desc, cnt_form, cnt_lemma))

        if bigram_data:
            # Build table
            table_rows = []
            table_rows.append("| Collocation | Lemmas | Forms | Lemmas | Status |")
            table_rows.append("|-------------|--------|-------|--------|--------|")
            for bigram, l1, l2, emoji, desc, cnt_f, cnt_l in bigram_data:
                table_rows.append(f"| **{bigram}** | {l1} + {l2} | {cnt_f}x | {cnt_l}x | {emoji} {desc} |")
            st.markdown("\n".join(table_rows))
    elif 'bigram_cache' not in st.session_state:
        st.warning(
            "First load the corpus in **Load Data** to check bigram frequencies.")

st.markdown("---")

# Reset
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("Reset Tree"):
        st.session_state.syntax_tree = init_tree()
        st.rerun()
