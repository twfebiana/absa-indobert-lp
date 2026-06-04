import re
import pandas as pd
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory



# LOAD KAMUS NORMALISASI

def load_slang_dict(path="utils/colloquial-indonesian-lexicon.csv"):
    df = pd.read_csv(path)
    return dict(zip(df["slang"], df["formal"]))

SLANG_DICT = load_slang_dict()


# STOPWORD

factory = StopWordRemoverFactory()
stopword = set(factory.get_stop_words())

negasi = {
    'tidak', 'bukan', 'belum', 'jangan'
}

stopword = stopword.difference(negasi)


# STEMMER

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()


# CLEANING

def cleaning(text):
    text = str(text)

    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)

    text = re.sub(
        r'\b([a-zA-Z]+)(?:2|²|\^2)(?=\b)',
        r'\1-\1',
        text
    )

    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)

    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    return text


# CASE FOLDING

def case_folding(text):
    return text.lower()


# TOKENISASI

def tokenisasi(text):
    return word_tokenize(text)


# NORMALISASI

def normalisasi(tokens):
    return [
        SLANG_DICT[token]
        if token in SLANG_DICT
        else token
        for token in tokens
    ]


# STOPWORD REMOVAL

def stopword_removal(tokens):
    return [
        token for token in tokens
        if token not in stopword
    ]


# STEMMING

def stemming(tokens):
    return [
        stemmer.stem(token)
        for token in tokens
    ]


# FULL PREPROCESS

def preprocess_text(text):

    hasil = {}

    hasil["original"] = text

    clean = cleaning(text)
    hasil["cleaning"] = clean

    cf = case_folding(clean)
    hasil["case_folding"] = cf

    tok = tokenisasi(cf)
    hasil["tokenisasi"] = tok

    norm = normalisasi(tok)
    hasil["normalisasi"] = norm

    sw = stopword_removal(norm)
    hasil["stopword_removal"] = sw

    stem = stemming(sw)
    hasil["stemming"] = stem

    final_text = " ".join(stem)
    hasil["final_text"] = final_text

    return hasil