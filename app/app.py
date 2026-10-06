from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf

# ============================================================
# Smart Waste Classifier - Polished Streamlit Application
# Model: MobileNetV2 Transfer Learning
# Test Accuracy: 94.75%
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mobilenetv2_waste.keras"
IMG_SIZE = (224, 224)

# ------------------------- PAGE ------------------------------

st.set_page_config(
    page_title="EcoSort AI | Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- CSS -------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f6fbf7 0%, #eef8f1 50%, #f8fbf9 100%);
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    .hero {
        padding: 28px 34px;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f6b43, #19a463);
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 12px 30px rgba(15,107,67,.18);
    }

    .hero h1 {
        font-size: 42px;
        margin: 0;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .hero p {
        font-size: 17px;
        margin: 8px 0 0;
        opacity: .92;
    }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 18px;
        border: 1px solid #dcebe1;
        box-shadow: 0 7px 22px rgba(0,0,0,.05);
        text-align: center;
        min-height: 125px;
    }

    .metric-title {
        color: #648071;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #0f6b43;
        font-size: 29px;
        font-weight: 800;
        margin-top: 8px;
    }

    .upload-card {
        background: white;
        padding: 24px;
        border-radius: 20px;
        border: 1px solid #dcebe1;
        box-shadow: 0 7px 22px rgba(0,0,0,.05);
    }

    .result-good {
        background: linear-gradient(135deg, #e8f8ee, #f7fff9);
        border: 2px solid #69bd89;
        border-radius: 20px;
        padding: 24px;
        text-align: center;
    }

    .result-bad {
        background: linear-gradient(135deg, #fff3e6, #fffaf5);
        border: 2px solid #e8a55f;
        border-radius: 20px;
        padding: 24px;
        text-align: center;
    }

    .result-label {
        font-size: 31px;
        font-weight: 800;
        margin: 5px 0 8px;
    }

    .result-confidence {
        font-size: 17px;
        color: #5f7067;
    }

    .section-title {
        font-size: 25px;
        font-weight: 800;
        color: #174b35;
        margin: 18px 0 12px;
    }

    .type-card {
        background: white;
        border: 1px solid #dcebe1;
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 5px 16px rgba(0,0,0,.04);
    }

    .type-card h4 {
        color: #174b35;
        margin: 4px 0;
    }

    .footer {
        text-align: center;
        color: #718077;
        padding: 25px 0 8px;
        font-size: 13px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        background: #138a55;
        color: white;
        font-weight: 700;
        padding: 12px;
    }

    div.stButton > button:hover {
        background: #0f6b43;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------- MODEL ------------------------------

if not MODEL_PATH.exists():
    st.error(
        "MobileNetV2 model not found. Please make sure "
        "`models/mobilenetv2_waste.keras` exists."
    )
    st.stop()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

# ------------------------- SIDEBAR ----------------------------

with st.sidebar:
    st.markdown("## ♻️ EcoSort AI")
    st.caption("Smart image-based waste classification")

    st.divider()

    st.markdown("### 🧠 Model")
    st.write("**MobileNetV2**")
    st.write("Transfer Learning")

    st.markdown("### 📊 Performance")
    st.metric("Test Accuracy", "94.75%")
    st.metric("F1-Score", "95.63%")

    st.divider()

    st.markdown("### 🗂️ Dataset")
    st.write("TrashNet")
    st.write("2,390 usable images")
    st.write("5 waste categories")

    st.divider()

    st.caption(
        "Degradable: Paper, Cardboard\n\n"
        "Non-Degradable: Plastic, Glass, Metal"
    )

# ------------------------- HERO --------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>♻️ EcoSort AI</h1>
        <p>
            Intelligent waste classification using CNN-based deep learning
            and MobileNetV2 transfer learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------- METRICS -----------------------------

c1, c2, c3, c4 = st.columns(4)

metrics = [
    ("🎯", "Test Accuracy", "94.75%"),
    ("📈", "F1-Score", "95.63%"),
    ("🖼️", "Test Images", "362"),
    ("🧠", "Model", "MobileNetV2"),
]

for col, (icon, title, value) in zip([c1, c2, c3, c4], metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{icon} {title}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# ------------------------- MAIN AREA ---------------------------

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="section-title">📷 Upload Waste Image</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="upload-card">', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a clear image of a waste item",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG",
    )

    if uploaded:
        image = tf.keras.utils.load_img(
            uploaded,
            target_size=IMG_SIZE
        )
        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True,
        )

        classify = st.button("🔍 Classify Waste")

        if classify:
            with st.spinner("Analyzing image with MobileNetV2..."):
                arr = tf.keras.utils.img_to_array(image)
                arr = np.expand_dims(arr, axis=0)

                arr = tf.keras.applications.mobilenet_v2.preprocess_input(
                    arr
                )

                probability = float(
                    model.predict(arr, verbose=0)[0][0]
                )

            if probability >= 0.5:
                label = "Non-Degradable"
                confidence = probability
                icon = "🚫"
                result_class = "result-bad"
                message = (
                    "This image is classified as non-degradable waste. "
                    "Examples include plastic, glass and metal."
                )
            else:
                label = "Degradable"
                confidence = 1 - probability
                icon = "🌱"
                result_class = "result-good"
                message = (
                    "This image is classified as degradable waste. "
                    "Examples include paper and cardboard."
                )

            confidence_pct = confidence * 100

            st.markdown(
                f"""
                <div class="{result_class}">
                    <div style="font-size:38px">{icon}</div>
                    <div class="result-label">{label}</div>
                    <div class="result-confidence">
                        Confidence: <b>{confidence_pct:.2f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(
                min(int(round(confidence_pct)), 100)
            )

            if confidence_pct >= 90:
                st.success("High-confidence prediction.")
            elif confidence_pct >= 70:
                st.info("Moderate-to-high confidence prediction.")
            else:
                st.warning(
                    "Lower-confidence prediction. Try a clearer image "
                    "with a single waste item and less background clutter."
                )

            st.write(message)

    else:
        st.info(
            "Upload a waste image to get a prediction and confidence score."
        )

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">🧩 Supported Waste Types</div>',
                unsafe_allow_html=True)

    a, b = st.columns(2)

    with a:
        st.markdown(
            """
            <div class="type-card">
                <div style="font-size:30px">🌱</div>
                <h4>Degradable</h4>
                <p>Paper<br>Cardboard</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            """
            <div class="type-card">
                <div style="font-size:30px">♻️</div>
                <h4>Non-Degradable</h4>
                <p>Plastic<br>Glass<br>Metal</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">⚙️ How It Works</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        **1. Upload**  
        Provide an image of the waste item.

        **2. Preprocess**  
        The image is resized to **224 × 224 pixels** and prepared
        using MobileNetV2 preprocessing.

        **3. Classify**  
        The trained MobileNetV2 model analyzes visual features.

        **4. Result**  
        The system returns the predicted category and confidence.
        """
    )

# ------------------------- MODEL DETAILS -----------------------

st.divider()

st.markdown(
    '<div class="section-title">🧠 Model & Project Details</div>',
    unsafe_allow_html=True,
)

d1, d2, d3 = st.columns(3)

with d1:
    st.markdown(
        """
        **Architecture**  
        MobileNetV2 + Transfer Learning

        **Input**  
        224 × 224 RGB image
        """
    )

with d2:
    st.markdown(
        """
        **Training Dataset**  
        TrashNet

        **Usable Images**  
        2,390
        """
    )

with d3:
    st.markdown(
        """
        **Training Strategy**  
        Data augmentation + fine-tuning

        **Test Accuracy**  
        94.75%
        """
    )

# ------------------------- DISCLAIMER --------------------------

st.info(
    "⚠️ This is an academic machine-learning prototype. "
    "Predictions depend on image quality and should not be treated "
    "as an authoritative waste-disposal decision."
)

st.markdown(
    '<div class="footer">'
    'EcoSort AI • Waste Classification using CNN-Based Image Classification'
    '</div>',
    unsafe_allow_html=True,
)
