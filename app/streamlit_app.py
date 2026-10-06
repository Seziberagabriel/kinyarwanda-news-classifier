"""Streamlit app: Kinyarwanda news topic classifier (fine-tuned AfriBERTa on KINNEWS)."""
import pandas as pd
import streamlit as st
from transformers import pipeline

MODEL_ID = "Seziberagabriel/kinyarwanda-news-classifier"

st.set_page_config(page_title="Kinyarwanda News Classifier", page_icon="📰")


@st.cache_resource(show_spinner="Loading model (first run takes a minute)...")
def load_model():
    return pipeline("text-classification", model=MODEL_ID, top_k=None)


EXAMPLES = {
    "Sport": "amavubi u anganyije zambia ahita asezererwa marushanwa. ikipe y’igihugu amavubi y’abatarengeje imyaka isezerewe zambia irushanwa ryo guhatanira itike kujya gikombe afurika.",
    "Relationship": "niki wakora mbere yogutandukana nuwo mwashakanye. icyo ugomba kumenya mbere yogutandukana n’uwo mwashakanye nubwo bigoye gufata icyemezo.",
    "Politics": "pdi ntiyanyuzwe kuba umukandida wayo atemerewe kujya sena. nyuma y’uko urukiko rw’ikirenga rwanzuye ishyaka riravuga ritanyuzwe.",
}

st.title("📰 Kinyarwanda News Topic Classifier")
st.write(
    "Fine-tuned **AfriBERTa** trained on the **KINNEWS** corpus (Niyongabo et al., 2020). "
    "Paste Kinyarwanda news text and the model predicts one of 14 topics: politics, sport, economy, "
    "health, entertainment, history, technology, tourism, culture, fashion, religion, environment, "
    "education, relationship."
)

choice = st.selectbox("Try an example (optional)", ["(write your own)"] + list(EXAMPLES))
default = EXAMPLES.get(choice, "")
text = st.text_area("Andika inkuru mu Kinyarwanda", value=default, height=200)

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        clf = load_model()
        # Training text was lowercased, so lowercase input to match.
        res = clf(text.lower(), truncation=True, max_length=256)
        if res and isinstance(res[0], list):
            res = res[0]
        df = pd.DataFrame(res).sort_values("score", ascending=False)
        top = df.iloc[0]
        st.success(f"Predicted topic: **{top['label']}** ({top['score']:.0%} confidence)")
        if top["score"] < 0.5:
            st.info("Confidence is low: the text may be ambiguous, short, or outside the news domain.")
        if len(text.split()) < 8:
            st.info("Very short inputs are less reliable; the model was trained on full news articles.")
        st.bar_chart(df.head(5).set_index("label")["score"])

st.caption("Limitations: news domain only; long articles are truncated to 256 tokens; "
           "weakest on rare topics such as history and environment.")
