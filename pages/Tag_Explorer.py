"""
Strona eksploracji tagów Morfeusza2 dla wszystkich wyrazów w korpusie.
"""
import streamlit as st
import pandas as pd
import yaml
from pathlib import Path
import sys
import morfeusz2

st.set_page_config(page_title="Tag Explorer", layout="wide")
st.title("Morfeusz2 Tag Explorer")

# Load config
sys.path.insert(0, str(Path(__file__).parent.parent))
config_path = Path(__file__).parent.parent / "config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Check if corpus is loaded
if not st.session_state.get('corpus_loaded', False):
    st.warning("Please load and process a corpus first in the Load Data page.")
    st.stop()

# Inicjalizacja Morfeusza2
morf = morfeusz2.Morfeusz()

# Nazwy pozycji w tagu dla różnych części mowy (NKJP tagset)
# https://git.nlp.ipipan.waw.pl/SGJP/Morfeusz/blob/master/input/morfeusz-sgjp.tagset
# https://korpusomat.readthedocs.io/_/downloads/pl/latest/pdf/
TAG_POSITIONS = {
    "subst": ["klasa", "liczba", "przypadek", "rodzaj"],
    "depr": ["klasa", "liczba", "przypadek", "rodzaj"],
    "adj": ["klasa", "liczba", "przypadek", "rodzaj", "stopień"],
    "adja": ["klasa"],
    "adjp": ["klasa"],
    "adjc": ["klasa"],
    "adv": ["klasa", "stopień"],
    "num": ["klasa", "liczba", "przypadek", "rodzaj", "akomodacyjność"],
    "numcol": ["klasa", "liczba", "przypadek", "rodzaj", "akomodacyjność"],
    "ppron12": ["klasa", "liczba", "przypadek", "rodzaj", "osoba", "akcentowość"],
    "ppron3": ["klasa", "liczba", "przypadek", "rodzaj", "osoba", "akcentowość", "poprzyimkowość"],
    "siebie": ["klasa", "przypadek"],
    "fin": ["klasa", "liczba", "osoba", "aspekt"],
    "bedzie": ["klasa", "liczba", "osoba", "aspekt"],
    "aglt": ["klasa", "liczba", "osoba", "aspekt", "wokaliczność"],
    "praet": ["klasa", "liczba", "rodzaj", "aspekt", "aglutynacyjność"],
    "impt": ["klasa", "liczba", "osoba", "aspekt"],
    "imps": ["klasa", "aspekt"],
    "inf": ["klasa", "aspekt"],
    "pcon": ["klasa", "aspekt"],
    "pant": ["klasa", "aspekt"],
    "ger": ["klasa", "liczba", "przypadek", "rodzaj", "aspekt", "negacja"],
    "pact": ["klasa", "liczba", "przypadek", "rodzaj", "aspekt", "negacja"],
    "ppas": ["klasa", "liczba", "przypadek", "rodzaj", "aspekt", "negacja"],
    "winien": ["klasa", "liczba", "rodzaj", "aspekt"],
    "pred": ["klasa"],
    "prep": ["klasa", "przypadek", "wokaliczność"],
    "conj": ["klasa"],
    "comp": ["klasa"],
    "qub": ["klasa", "wokaliczność"],
    "brev": ["klasa", "interpunkcja"],
    "burk": ["klasa"],
    "interj": ["klasa"],
    "interp": ["klasa"],
    "xxx": ["klasa"],
    "ign": ["klasa"],
}

# Cache w sesji - analiza tagów
if 'word_tags' not in st.session_state:
    with st.spinner("Analysing Morfeusz2 tags for all words in corpus..."):
        all_words = set(st.session_state.tokens_df['token'].str.lower().unique())
        
        # Analizuj tagi dla wszystkich słów
        word_tags = []  # lista (słowo, lemat, tag_surowy, pos, expanded_parts)
        pos_options: dict[str, list[set]] = {}  # pos -> pozycja -> set(opcje)
        
        for word in all_words:
            try:
                for interp in morf.analyse(word):
                    lemma = interp[2][1]
                    tag = interp[2][2]
                    parts = tag.split(":")
                    pos = parts[0] if parts else "?"
                    
                    # Rozbij każdą część na opcje (po kropce)
                    expanded_parts = []
                    for part in parts:
                        expanded_parts.append(part.split("."))
                    
                    word_tags.append((word, lemma, tag, pos, expanded_parts))
                    
                    # Zbierz opcje dla każdej pozycji
                    if pos not in pos_options:
                        pos_options[pos] = []
                    
                    while len(parts) > len(pos_options[pos]):
                        pos_options[pos].append(set())
                    for i, expanded_part in enumerate(expanded_parts):
                        pos_options[pos][i].update(expanded_part)
            except:
                pass
        
        st.session_state['word_tags'] = word_tags
        st.session_state['pos_options'] = pos_options

word_tags = st.session_state['word_tags']
pos_options = st.session_state['pos_options']

# UI: wybór części mowy
st.markdown("### 1. Wybierz część mowy")
all_pos = sorted(pos_options.keys())
selected_pos = st.selectbox("Część mowy:", all_pos, key="pos_select")

# Pokaż opcje dla wybranej części mowy
st.markdown("### 2. Pick criteria (checkboxes)")
st.caption("Select options that must be met. No selection = accept all in this column.")

# Pobierz nazwy pozycji dla tej POS
default_names = ["klasa"] + [f"poz_{i}" for i in range(1, 10)]
pos_names = TAG_POSITIONS.get(selected_pos, default_names) if selected_pos else default_names

# Utwórz kolumny dla każdej pozycji w tagu
if selected_pos is not None:
    pos_option_words = pos_options[selected_pos]
    cols = st.columns(len(pos_option_words))
    selected_criteria: dict[int, set[str]] = {}  # pozycja -> set(wybrane opcje)

    for col_idx, options in enumerate(pos_option_words):
        pos_name = pos_names[col_idx] if col_idx < len(pos_names) else f"poz_{col_idx}"
        
        with cols[col_idx]:
            st.markdown(f"**{pos_name}**")
            selected = set()
            for opt in options:
                if st.checkbox(opt, key=f"opt_{selected_pos}_{col_idx}_{opt}"):
                    selected.add(opt)
            selected_criteria[col_idx] = selected

    # Filtruj wyrazy według kryteriów
    st.markdown("### 3. Examples of words")

    matching = []
    for word, lemma, tag, pos, expanded_parts in word_tags:
        if pos != selected_pos:
            continue
        
        # Sprawdź czy pasuje do wszystkich kryteriów
        matches = True
        for pos_idx, required in selected_criteria.items():
            if not required:  # Brak zaznaczenia = akceptuj wszystko
                continue
            if pos_idx >= len(expanded_parts):
                matches = False
                break
            # Sprawdź czy którakolwiek opcja z tagu pasuje do wybranych
            if not (set(expanded_parts[pos_idx]) & required):
                matches = False
                break
        
        if matches:
            matching.append((word, lemma, tag))

    # Pokaż wyniki
    if matching:
        st.success(f"Found **{len(matching)}** matching forms")
        
        # Usuń duplikaty (słowo, tag)
        unique_matches = list(set((w, l, t) for w, l, t in matching))
        unique_matches.sort(key=lambda x: (x[0], x[2]))
        
        # Wyświetl jako tabelę
        df = pd.DataFrame(unique_matches[:200], columns=["Forma", "Lemat", "Tag"])
        st.dataframe(df, use_container_width=True, height=400)
        
        if len(unique_matches) > 200:
            st.info(f"Showing 200 of {len(unique_matches)} results")
    else:
        st.warning("No words matching the criteria were found")

