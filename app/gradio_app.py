"""Gradio app: Kinyarwanda news topic classifier (fine-tuned AfriBERTa on KINNEWS)."""
import os
import gradio as gr
from transformers import pipeline

MODEL_ID = os.environ.get("MODEL_ID", "Seziberagabriel/kinyarwanda-news-classifier")
clf = pipeline("text-classification", model=MODEL_ID, top_k=None)


def predict(text):
    if not text or not text.strip():
        return {}, "Please enter some Kinyarwanda text."
    # The training text was lowercased, so we lowercase input to match.
    res = clf(text.lower(), truncation=True, max_length=256)
    if res and isinstance(res[0], list):
        res = res[0]
    scores = {r["label"]: float(r["score"]) for r in res}
    top = max(scores, key=scores.get)
    note = f"Predicted topic: **{top}** ({scores[top]:.0%} confidence)."
    if scores[top] < 0.5:
        note += " Confidence is low: the text may be ambiguous, short, or outside the news domain."
    if len(text.split()) < 8:
        note += " Very short inputs are less reliable; the model was trained on full news articles."
    return scores, note


examples = [
    ["amavubi u anganyije zambia ahita asezererwa marushanwa. ikipe y’igihugu amavubi y’abatarengeje imyaka isezerewe zambia irushanwa ryo guhatanira itike kujya gikombe afurika."],
    ["niki wakora mbere yogutandukana nuwo mwashakanye. icyo ugomba kumenya mbere yogutandukana n’uwo mwashakanye nubwo bigoye gufata icyemezo."],
    ["pdi ntiyanyuzwe kuba umukandida wayo atemerewe kujya sena. nyuma y’uko urukiko rw’ikirenga rwanzuye ishyaka riravuga ritanyuzwe."],
]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=8, label="Andika inkuru mu Kinyarwanda (Kinyarwanda news text)"),
    outputs=[gr.Label(num_top_classes=5, label="Predicted topic"), gr.Markdown()],
    title="Kinyarwanda News Topic Classifier",
    description=(
        "Fine-tuned AfriBERTa (castorini/afriberta_base) trained on the KINNEWS corpus "
        "(Niyongabo et al., 2020). 14 topics: politics, sport, economy, health, entertainment, "
        "history, technology, tourism, culture, fashion, religion, environment, education, relationship. "
        "Works best on news-style text; weakest on history and environment (few training examples)."
    ),
    examples=examples,
)

if __name__ == "__main__":
    demo.launch()
