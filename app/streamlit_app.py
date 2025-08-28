import streamlit as st
import pandas as pd
from src.data_pipeline import load_reviews_csv
from src.rules_engine import load_policy, predict_batch

st.set_page_config(page_title="Trustworthy Location Reviews", layout="wide")
st.title("Filtering the Noise — Review Moderation (Starter)")

cfg = load_policy("configs/policy.yaml")

st.sidebar.header("Input")
mode = st.sidebar.radio("Mode", ["Single review", "Batch CSV"], index=0)

if mode == "Single review":
    biz_name = st.text_input("Business Name", "Sunrise Coffee")
    biz_cats = st.text_input("Business Categories", "Cafe, Coffee")
    biz_desc = st.text_area("Business Description", "Small-batch roastery and cafe near Riverwalk.")
    review_text = st.text_area("Review Text", "Lovely flat white and chill vibes. Beans are freshly roasted on-site.")
    if st.button("Predict"):
        df = pd.DataFrame([{
            "biz_name": biz_name,
            "biz_cats": biz_cats,
            "biz_desc": biz_desc,
            "review_text": review_text
        }])
        out = predict_batch(df, cfg)
        st.json(out[0])
else:
    st.write("Upload a CSV with columns: biz_name,biz_cats,biz_desc,review_text (optional: labels).")
    file = st.file_uploader("reviews.csv", type=["csv"])
    if st.button("Run on sample") or file is not None:
        if file is None:
            path = "data/samples/reviews_sample.csv"
            df = load_reviews_csv(path)
        else:
            df = pd.read_csv(file)
        out = predict_batch(df, cfg)
        res = pd.concat([df, pd.DataFrame(out)], axis=1)
        st.dataframe(res)
        st.download_button("Download Results", res.to_csv(index=False), file_name="predictions.csv")
