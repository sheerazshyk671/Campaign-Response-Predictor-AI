# 📧 Campaign Response Predictor

A machine learning app that predicts whether a customer will respond to a marketing campaign.

## 🎯 What It Does

- Upload a CSV of customer data
- Predicts who will respond to your campaign
- Download results with recommendations (Target ✅ / Skip ❌)

## 🧠 Model

- **Algorithm:** Logistic Regression
- **Accuracy:** ~82%
- **Features used:** Age, Income, Recency, TotalSpend, Children, NumDealsPurchases, NumWebPurchases, NumWebVisitsMonth

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/campaign-response-predictor.git
cd campaign-response-predictor

# Install dependencies
pip install -r requirements.txt

# Train the model (if not already trained)
python train_model.py

# Run the app
streamlit run app.py