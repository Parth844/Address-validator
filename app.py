import streamlit as st
from src.predict import predict

# Page config
st.set_page_config(
    page_title="Address Validator",
    page_icon="📍",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
}
.valid {
    background-color: #0f5132;
    color: #d1e7dd;
}
.invalid {
    background-color: #842029;
    color: #f8d7da;
}
.score {
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">📦 Address Validation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Validate Indian addresses using ML 🚀</div>', unsafe_allow_html=True)

# Input
address = st.text_area("📍 Enter Address", height=120)

col1, col2 = st.columns(2)

# Example buttons
with col1:
    if st.button("✅ Try Valid Example"):
        address = "123 MG Road, Bangalore, Karnataka 560001"

with col2:
    if st.button("❌ Try Invalid Example"):
        address = "asdfgh jklmnop"

# Validate button
if st.button("🔍 Validate Address"):

    if not address.strip():
        st.warning("Please enter an address")
    else:
        result = predict(address)

        score = result["valid_score"]
        prediction = result["prediction"]

        # Progress bar
        st.progress(float(score))

        # Result UI
        if prediction == "VALID":
            st.markdown(f"""
            <div class="result-box valid">
                <h3>✅ VALID ADDRESS</h3>
                <p class="score">Confidence: {score*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box invalid">
                <h3>❌ INVALID ADDRESS</h3>
                <p class="score">Confidence: {score*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

        # Expandable debug info
        with st.expander("🔎 See Details"):
            st.write(result)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using XGBoost + Streamlit")