import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Bazel Build Time Predictor")
st.title("Bazel Build CPU-Time Predictor")
st.write("Pick the kind of change in a commit and get the predicted CPU time of the build.")

PREFIXES = ["bazelci/", "examples/", "scripts/", "site/", "src/conditions",
            "src/java_tools", "src/main", "src/test", "src/tools",
            "third_party/", "tools/"]
TYPES = ["JAVA", "C/C++", "Starlark", "python", "HTML/CSS/JS"]


@st.cache_resource
def load_models():
    return (joblib.load("model/best_no_cross.joblib"),
            joblib.load("model/best_cross.joblib"))


m_no, m_cr = load_models()

prefix = st.selectbox("Most common changed file prefix", PREFIXES)
ftype = st.selectbox("Most common changed file type", TYPES)

if st.button("Predict CPU time"):
    row = pd.DataFrame([{
        "prefix": prefix,
        "file_type": ftype,
        "cross": prefix + "_" + ftype,
    }])
    p_no = m_no.predict(row)[0]
    p_cr = m_cr.predict(row)[0]

    col1, col2 = st.columns(2)
    col1.metric("Without feature cross", f"{p_no:,.0f} ms")
    col2.metric("With feature cross", f"{p_cr:,.0f} ms")
    st.caption("Predictions come from Ridge regression models trained on commit data.")