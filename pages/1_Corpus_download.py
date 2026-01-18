import sys
import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from dataclasses import dataclass
from pathlib import Path
import yaml
import glob
import pickle

st.set_page_config(page_title="Corpus download", page_icon="📚")
st.title("Corpus download")
st.markdown("""
This module allows for the acquisition and analysis of literary texts from **Wolne Lektury**. 
It includes functionality for scraping book metadata, downloading text content, and performing initial statistical analysis on the corpus.
""")

# Load config
sys.path.insert(0, str(Path(__file__).parent.parent))
config_path = Path(__file__).parent.parent / "config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

BOOKS_PICKLE_PATH = os.path.join(config['corpus']['pickle_path'])
BOOKS_SOURCE_DIR = os.path.join(config['corpus']['data_dir'])
WOLNE_LEKTURY_BASE_URL = config['corpus']['wolne_lektury_url']

# --- Data Classes ---
@dataclass
class BookFragment:
    html_url: str
    txt_url: str


@dataclass
class Book:
    name: str
    fragments: list[BookFragment]

    def __hash__(self):
        return hash(self.name)


# Hack to allow unpickling of classes defined in __main__ in the original script
if "Book" not in sys.modules['__main__'].__dict__:
    setattr(sys.modules['__main__'], 'Book', Book)
if "BookFragment" not in sys.modules['__main__'].__dict__:
    setattr(sys.modules['__main__'], 'BookFragment', BookFragment)

# --- Helper Functions ---

def download_text(url: str, use_txt: bool = False) -> str | None:
    """Downloads text from a given URL."""\
    # eg. download_text('https://wolnelektury.pl/katalog/lektura/prawdziwy-opis-wypadku-z-p-waldemarem.html')
    try:
        res: requests.Response = requests.get(url)
        if res.status_code == 404:
            return None

        if use_txt:  # Parse the text
            txt: str = res.text
            # Should remove the header (ends with double `\n`) and the footer (starts with `-----`)
            # NOTE: It's not trivial to remove the header due to whitespace characters probably?...
            footer_start: int = txt.rfind('-----')
            return txt[:footer_start]
        else:  # Parse HTML
            soup = BeautifulSoup(res.text, 'html.parser')
            text_div = soup.find(id='book-text')
            if text_div:
                text: str = " ".join(
                    [ele.text for ele in text_div.find_all(class_=["paragraph", "stanza"])])
                return text
            return None
    except Exception as e:
        st.error(f"Error downloading book content {url}: {e}")
        return None


def scrape_books(only_use_school_books: bool):
    """Scrapes book URLs from Wolne Lektury."""
    use_pagination: bool = True
    all_url: str = f'{WOLNE_LEKTURY_BASE_URL}/katalog/lektury/'
    if only_use_school_books:
        all_url = f'{WOLNE_LEKTURY_BASE_URL}/info/lektury-szkolne/'
        use_pagination = False

    page_num: int = 1
    book_urls: set[Book] = set()

    status_text = st.empty()

    while True:
        status_text.text(f"Fetching page {page_num}...")
        current_url: str = f'{all_url}?page={page_num}' if use_pagination else all_url
        res: requests.Response = requests.get(current_url)
        soup = BeautifulSoup(res.text, 'html.parser')

        if res.status_code == 404:  # Page does not exist
            break

        links = soup.find_all('a')
        for ele in links:
            href = ele.attrs.get('href')
            if href is None or '/katalog/lektura' not in href:
                continue

            # Fetch the main book page (usually w/o extension)
            book_url: str = f"{WOLNE_LEKTURY_BASE_URL}{href}"
            book_txt_result = re.search(
                r"\/katalog\/lektura\/([\w-]+)\/", book_url)
            if book_txt_result is None:
                continue
            if any(book.name == book_txt_result[1] for book in book_urls):
                continue  # Already fetched

            book_page_res: requests.Response = requests.get(book_url)
            book_soup = BeautifulSoup(book_page_res.text, 'html.parser')

            # Find the article section and retrieve the html book page
            article = book_soup.find(class_='l-article')
            if article is None: continue
            read_online_link = article.find(class_='l-button--media--full')
            if read_online_link is None: continue

            # Create the empty book object
            cur_book: Book = Book(book_txt_result[1], [])

            # Prepare the first fragment url
            cur_href_attr = read_online_link.attrs.get('href')
            if cur_href_attr is None or isinstance(cur_href_attr, list):
                continue
            cur_href: str = cur_href_attr
            cur_url: str = f"{WOLNE_LEKTURY_BASE_URL}{cur_href}"

            while cur_url is not None:  # Keep fetching the current fragment and find potential next one
                cur_txt_result = re.search(
                    r"\/katalog\/lektura\/([\w-]+).html", cur_url)
                if cur_txt_result is None:
                    break

                cur_book.fragments.append(BookFragment(
                    # Add the current fragment
                    cur_url, f"{WOLNE_LEKTURY_BASE_URL}/media/book/txt/{cur_txt_result[1]}.txt"))

                # Find the next fragment url
                cur_res: requests.Response = requests.get(
                    cur_url)  # Fetch the current fragment source
                if cur_res.status_code == 404:
                    break
                cur_soup = BeautifulSoup(cur_res.text, 'html.parser')

                # Find the next fragment link
                next_link = cur_soup.find('a', class_='text_next-book')
                if next_link is not None:
                    next_href = next_link.attrs.get('href')
                    if next_href is None or isinstance(next_href, list):
                        break
                    # Repeat with updated url
                    cur_url = f"{WOLNE_LEKTURY_BASE_URL}{str(next_href)}"
                else:
                    break  # Finish the search of fragments

            if len(cur_book.fragments) > 0:
                book_urls.add(cur_book)
                st.write(f"Found: {cur_book.name} ({len(cur_book.fragments)} fragments)")

        page_num += 1
        if not use_pagination:
            break

    status_text.empty()
    return book_urls

# --- Main UI ---
st.header("1. Data Acquisition")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Scraping book metadata")
    only_use_school_books = st.checkbox("Only use school books", value=False)

    if st.button("Start Scraping"):
        with st.spinner("Scraping books..."):
            if not os.path.exists(BOOKS_PICKLE_PATH):
                book_urls = scrape_books(only_use_school_books)

                # Save the books urls
                with open(BOOKS_PICKLE_PATH, 'wb') as f:
                    pickle.dump(book_urls, f)
                st.success(
                    f"Scraped {len(book_urls)} books. Saved to {BOOKS_PICKLE_PATH}")
            else:
                st.warning(
                    f"Books metadata already exists at {BOOKS_PICKLE_PATH}. Please delete it to re-scrape.")

with col2:
    st.subheader("Downloading book content")
    if os.path.exists(BOOKS_PICKLE_PATH):
        with open(BOOKS_PICKLE_PATH, 'rb') as f:
            book_urls = pickle.load(f)
        st.info(f"Found {len(book_urls)} books in metadata.")

        if st.button("Download Text Content"):
            os.makedirs(BOOKS_SOURCE_DIR, exist_ok=True)
            progress_bar = st.progress(0)
            status_text = st.empty()

            total_books = len(book_urls)
            for idx, book in enumerate(book_urls):
                status_text.text(
                    f"Processing {book.name} ({idx+1}/{total_books})")
                progress_bar.progress((idx + 1) / total_books)

                if os.path.exists(os.path.join(BOOKS_SOURCE_DIR, f"{book.name}.txt")):
                    continue

                all_text: str = ''
                all_success: bool = True
                for i, fragment in enumerate(book.fragments):
                    text: str | None = download_text(
                        fragment.txt_url, use_txt=True)
                    if text is None:
                        text = download_text(fragment.html_url)
                    if text is not None:
                        all_text += text + ' '
                    else:
                        all_success = False
                        st.write(f'Failed to fetch text for {book.name} fragment {i}')
                        break

                if all_success:
                    st.write(f"Successfully fetched {book.name} fragments {len(book.fragments)}")
                    try:
                        with open(os.path.join(BOOKS_SOURCE_DIR, f"{book.name}.txt"), 'w', encoding=config['corpus']['encoding'], newline='') as f:
                            f.write(all_text.strip())
                    except Exception as e:
                        st.error(f"Error writing file for {book.name}: {e}")

            status_text.empty()
            st.success("Download complete!")
    else:
        st.warning("No book metadata found. Please scrape the list first.")

st.header("2. Simple analysis")

if st.button("Run analysis"):
    with st.spinner("Analyzing text files..."):
        # Load all text files
        txt_files = glob.glob(os.path.join(BOOKS_SOURCE_DIR, "*.txt"))

        if not txt_files:
            st.error("No text files found. Please download content first.")
        else:
            full_text: str = ""

            progress_bar = st.progress(0)
            for i, file_path in enumerate(txt_files):
                with open(file_path, 'r', encoding=config['corpus']['encoding'], errors='ignore') as f:
                    full_text += f.read().strip() + " "
                if i % 10 == 0:
                    progress_bar.progress((i + 1) / len(txt_files))
            progress_bar.progress(1.0)

            # Preprocessing
            s = re.sub(r'[^\w\s]|\d', ' ', full_text)
            words = [w.lower() for w in s.split() if w]

            st.write(f"Total words processed: {len(words)}")

            # Frequency analysis
            word_freq: dict[str, int] = {}
            for w in words:
                word_freq[w] = word_freq.get(w, 0) + 1

            vals = sorted(word_freq.items(),
                          key=lambda x: x[1], reverse=True)
            df = pd.DataFrame(vals, columns=['word', 'count'])

            st.subheader("Top Words")
            st.dataframe(df.head(20))

            # Comparison with Wiktionary data
            # https://pl.wiktionary.org/wiki/Indeks:Polski_-_Najpopularniejsze_s%C5%82owa_1-2000
            wiki_data = {
                "w": 78576, "z": 57267, "być": 48949, "na": 45738, "i": 44709, "do": 35686,
                "nie": 23257, "który": 18821, "lub": 18533, "to": 18340, "o": 16116, "mieć": 15641,
                "się": 15001, "coś": 12678, "ten": 10236, "dotyczyć": 9548, "że": 9243, "od": 8581,
                "on": 8285, "co": 8250, "po": 8184, "przez": 7861, "miasto": 7398, "ktoś": 7195,
                "język": 7189, "a": 7072, "jak": 7036, "za": 6884, "osoba": 6840, "cecha": 6701,
                "ja": 6537, "jeden": 6375, "rok": 6253, "swój": 6051, "dla": 5946, "bardzo": 5894,
                "taki": 5765, "związany": 5621, "móc": 5535, "mój": 5316, "człowiek": 5314,
                "część": 5032, "kobieta": 4875, "ona": 4745, "dwa": 4482, "ze": 4421, "inny": 4266
            }

            w_top10 = set([k for k in wiki_data.keys()][:20])
            m_top10 = set([v[0] for v in vals][:20])

            st.subheader("Comparison of top 20 common words with Wiktionary")

            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Wiktionary Top 20:**")
                st.write(w_top10)
            with col_b:
                st.write("**Corpus Top 20:**")
                st.write(m_top10)

            diff_wikt_to_corp = w_top10.difference(m_top10)
            st.write(f"**Difference (in Wiktionary but not in Corpus):** {len(diff_wikt_to_corp)}")
            st.write(diff_wikt_to_corp)

            diff_corp_to_wiki = m_top10.difference(w_top10)
            st.write(f"**Difference (in Corpus but not in Wiktionary):** {len(diff_corp_to_wiki)}")
            st.write(diff_corp_to_wiki)
