import streamlit as st
import requests

BACKEND = "http://127.0.0.1:5000/predict"
st.title("Authentica:Fake News Detector ")

text = st.text_area("Enter text:", height=200)
model = st.selectbox("Model:", ["lr", "dt", "gb", "rf"])

if st.button("Predict"):
    if not text.strip():
        st.error("Please enter some text.")
    else:
        try:
            r = requests.post(BACKEND, json={"text": text, "model": model})
            data = r.json()
            st.write("### Result")
            st.write(f"**Label:** {data.get('label')}")
            # st.write(f"**Raw:** {data.get('raw_prediction')}")
            st.write(f"**Confidence:** {data.get('confidence')}")
            if data.get("probabilities") is not None:
                st.write("**Probabilities:**", data["probabilities"])
        except Exception as e:
            st.error(f"Error: {e}")