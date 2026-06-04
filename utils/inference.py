import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from utils.preprocessing import preprocess_text

MODEL_PATH = "twfebiana/absa-indobert-lp"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None

with open("utils/label_maps.json", "r") as f:
    maps = json.load(f)

id2label = {
    int(k): v
    for k, v in maps["id2label"].items()
}

binary_map = maps["binary_map"]


# LOAD MODEL

def load_model():

    global tokenizer
    global model

    if tokenizer is None or model is None:

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH
        )

        model.to(DEVICE)
        model.eval()


# MAP ASPEK

def map_aspects(binary):

    keys = [
        "kpms_pos",
        "kpms_neg",
        "fi_pos",
        "fi_neg",
        "wt_pos",
        "wt_neg",
        "bl_pos",
        "bl_neg"
    ]

    result = {}

    for k, b in zip(keys, binary):
        result[k] = int(b)

    return result


# PREDIKSI

def predict_single(text):

    load_model()

    prep = preprocess_text(text)

    final_text = prep["final_text"]

    enc = tokenizer(
        final_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    enc = {
        k: v.to(DEVICE)
        for k, v in enc.items()
    }

    with torch.no_grad():

        logits = model(**enc).logits

        pred_id = int(
            torch.argmax(logits, dim=1)
            .cpu()
            .numpy()[0]
        )

    pred_class = id2label[pred_id]

    binary = binary_map[pred_class]

    aspects = map_aspects(binary)

    return {
        "preprocessing": prep,
        "class": pred_class,
        "binary": binary,
        "aspects": aspects
    }