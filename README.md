# Sniffbnb  
### AI-Powered Airbnb Trust & Price Intelligence  
**Trust Before You Book**

## Overview

Sniffbnb is an end-to-end Machine Learning application that analyzes Airbnb listings to:

- Predict whether a listing is **Trustworthy, Neutral, or Suspicious**
- Estimate a **fair market price per night**
- Explain predictions using **SHAP (Explainable AI)**
- Provide an interactive web interface via **Streamlit**

This project combines classification, regression, and explainability in a clean, deployment-ready ML pipeline.

## Machine Learning Architecture

### Trust Classification Model
- **Algorithm:** XGBoost Classifier  
- **Output:** Trust Category (Trustworthy / Neutral / Suspicious)  
- **Key Features:**
  - Host age (days)
  - Number of reviews
  - Review rating
  - Price vs neighborhood average
  - Availability ratio
  - Host verification status

---

### Price Prediction Model
- **Algorithm:** XGBoost Regressor  
- **Output:** Fair Market Price per Night  
- **Key Features:**
  - Accommodates
  - Bedrooms & bathrooms
  - Amenities count
  - Location (latitude & longitude)
  - Review rating
  - Availability ratio

## Explainability (SHAP Integration)

Sniffbnb integrates SHAP to provide model transparency.

The system:
- Identifies top features influencing predictions
- Visualizes feature impact using interactive charts
- Generates human-readable reasoning

### Example Explanation:

> "This listing was flagged as Suspicious because the price is significantly above neighborhood average, the host has limited tenure, and the review count is low."

---
##  Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/tanishaasaklani/sniffbnb-ai.git
   cd sniffbnb-ai
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the app:
   ```bash
   streamlit run app.py

## 📬 Contact
Created by **Tanisha Saklani** - MCA AI & DS Student.
Open to feedback!
