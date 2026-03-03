import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# 1. PAGE CONFIG & GOOGLE FONTS
# ---------------------------
st.set_page_config(
    page_title="Sniffbnb",
    page_icon="https://cdn-icons-png.flaticon.com/512/69/69524.png",
    layout="wide")

# Injecting modern typography and Glassmorphism CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Hero Styling */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 3rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    /* Glassmorphic Input Card */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1e293b;
        border-radius: 10px 10px 0 0;
        color: white;
        padding: 0 20px;
    }

    /* Styled Result Box */
    .result-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        border-radius: 16px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 2. HERO SECTION
# ---------------------------
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">SNIFFBNB</h1>
        <p>AI-Powered Airbnb Trust & Price Intelligence</p>
        <div class="badge">Trust Before You Book</div>
            """, unsafe_allow_html=True)

# ---------------------------
# 3. DATA & MODELS (With Error Handling)
# ---------------------------
@st.cache_resource
def load_assets():
    try:
        t_model = pickle.load(open("models/trust_model_xgboost.pkl", "rb"))
        p_model = pickle.load(open("models/best_price_model.pkl", "rb"))
        t_cols = pickle.load(open("models/trust_feature_columns.pkl", "rb"))
        p_cols = pickle.load(open("models/price_feature_columns.pkl", "rb"))
        return t_model, p_model, t_cols, p_cols
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the /models folder is present.")
        return None, None, None, None

trust_model, price_model, trust_cols, price_cols = load_assets()

# ---------------------------
# 4. INPUT INTERFACE (Organized Tabs)
# ---------------------------
st.markdown("### Listing Configuration")
tab1, tab2 = st.tabs([" Host & Trust Metrics", "Property Details"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        review_score = st.slider("User Rating Score", 0.0, 5.0, 4.8, help="Average review stars")
        num_reviews = st.number_input("Total Reviews Count", 0, 5000, 45)
        host_age_days = st.slider("Host Seniority (Days)", 0, 5000, 730)
    with c2:
        host_response_rate = st.slider("Host Response (%)", 0, 100, 95)
        price_vs_avg = st.slider("Price Ratio vs. Neighborhood", 0.5, 3.0, 1.0)
        availability_ratio = st.slider("Yearly Availability %", 0.0, 1.0, 0.4)

with tab2:
    c3, c4 = st.columns(2)
    with c3:
        accommodates = st.number_input("Guest Capacity", 1, 16, 2)
        bedrooms = st.number_input("Bedrooms", 1, 10, 1)
        bathrooms = st.slider("Bathrooms", 1.0, 5.0, 1.0)
    with c4:
        latitude = st.number_input("Latitude", value=51.50)
        longitude = st.number_input("Longitude", value=-0.12)
        amenities_count = st.slider("Number of Amenities", 0, 100, 15)
        instant_bookable = st.toggle("Instant Bookable Enabled", value=True)

# ---------------------------
# 5. PREDICTION LOGIC
# ---------------------------
st.markdown("---")
if st.button(" Run AI Analysis", use_container_width=True):
    if trust_model:
        # Prepare Data
        trust_input = pd.DataFrame([{
            "host_age_days": host_age_days, "description_length": 150,
            "amenities_count": amenities_count, "host_response_rate": host_response_rate,
            "number_of_reviews": num_reviews, "review_scores_rating": review_score,
            "price_vs_avg": price_vs_avg, "sentiment_score": 0.5,
            "price_volatility": 0.1, "availability_ratio": availability_ratio,
            "review_to_listing_ratio": 0.2, "host_identity_verified": 1,
            "host_has_profile_pic": 1, "instant_bookable": int(instant_bookable)
        }])[trust_cols]

        price_input = pd.DataFrame([{
            "accommodates": accommodates, "bathrooms": bathrooms, "bedrooms": bedrooms,
            "amenities_count": amenities_count, "room_type_enc": 1, "neighbourhood_enc": 10,
            "latitude": latitude, "longitude": longitude, "review_scores_rating": review_score,
            "availability_ratio": availability_ratio
        }])[price_cols]

        trust_pred = trust_model.predict(trust_input)[0]
        predicted_price = price_model.predict(price_input)[0]


        # ---------------------------
        # 6.RESULTS DISPLAY
        # ---------------------------

        st.markdown("## AI Analysis Results")

        m1, m2 = st.columns(2)

        # Clean trust label handling (works for string or numeric)
        trust_map = {
            0: ("Suspicious", "#ef4444"),
            1: ("Neutral", "#f59e0b"),
            2: ("Trustworthy", "#22c55e"),
            "Suspicious": ("Suspicious", "#ef4444"),
            "Neutral": ("Neutral", "#f59e0b"),
            "Trustworthy": ("Trustworthy", "#22c55e")
            }

        trust_label, trust_color = trust_map.get(trust_pred, ("Unknown", "#9ca3af"))
        with m1:
            st.markdown(f"""
        <div class="result-card">
            <p style="margin:0; color:#94a3b8; font-size:0.9rem;">Trust Assessment</p>
            <h2 style="color:{trust_color}; margin:0; font-weight:700;">{trust_label}</h2>
        </div>""", unsafe_allow_html=True)
            
        with m2:
            st.markdown(f"""
        <div class="result-card">
            <p style="margin:0; color:#94a3b8; font-size:0.9rem;">Fair Market Price</p>
            <h2 style="color:#38bdf8; margin:0; font-weight:700;">
                £{predicted_price:,.2f}
                <span style="font-size:1rem; color:#64748b;"> / night</span>
            </h2>
        </div>""", unsafe_allow_html=True)

        # ---------------------------
        # 7. EXPLAINABILITY
        # ---------------------------

        st.markdown("### Why did the AI decide this?")

        explainer = shap.TreeExplainer(trust_model)
        shap_values = explainer(trust_input)

        class_idx = np.where(trust_model.classes_ == trust_pred)[0][0]

        impact_df = pd.DataFrame({
            "Feature": trust_cols,"Impact": shap_values.values[0, :, class_idx]})

        # Sort by absolute importance
        impact_df["AbsImpact"] = impact_df["Impact"].abs()
        impact_df = impact_df.sort_values("AbsImpact", ascending=False)

        top_factors = impact_df.head(3)

        # Plot
        fig = px.bar(impact_df.head(8).sort_values("Impact"),x="Impact",y="Feature",orientation='h',
                     color="Impact",color_continuous_scale="RdYlGn")

        fig.update_layout(template="plotly_dark",paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',title="Top Factors Influencing Trust Score")

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------
        # 8. HUMAN-FRIENDLY SUMMARY
        # ---------------------------

        reason_1 = top_factors.iloc[0]["Feature"].replace("_", " ")
        reason_2 = top_factors.iloc[1]["Feature"].replace("_", " ")
        reason_3 = top_factors.iloc[2]["Feature"].replace("_", " ")

        if trust_label == "Suspicious":
            summary_text = f"""
            This listing was flagged as **Suspicious** mainly due to:
            • {reason_1}
            • {reason_2}
            • {reason_3}"""
        elif trust_label == "Neutral":
            summary_text = f"""This listing appears **Neutral**. The most influential factors were:
            • {reason_1}
            • {reason_2}
            • {reason_3}"""
        else:
            summary_text = f"""This listing is considered **Trustworthy** based on:
            • {reason_1}
            • {reason_2}
            • {reason_3}"""

        st.markdown(f"""<div class="result-card"
        <p style="color:#94a3b8; font-size:0.9rem;">AI Explanation</p>
                    <p style="font-size:1rem;">{summary_text}</p>
                    </div>""", unsafe_allow_html=True)