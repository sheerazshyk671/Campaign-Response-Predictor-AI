import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.set_page_config(page_title="Campaign Predictor",layout="wide")
st.title("📧 Campaign Response Predictor")
st.write("Upload your customer list and predict who will respond to your marketing campaign.")


@st.cache_resource
def load_model():
    model=pickle.load(open('model.pkl','rb'))
    scaler=pickle.load(open('scaler.pkl','rb'))
    features=pickle.load(open('features.pkl','rb'))
    return model, scaler, features

model,scaler,FEATURES=load_model()


st.sidebar.header("📋 How to Use")
st.sidebar.markdown("""
**Steps:**
1. Download the template CSV
2. Fill it with your customer data
3. Upload it here
4. Click Predict
5. Download results
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Required columns:**")
for col in FEATURES:
    st.sidebar.markdown(f"- `{col}`")


st.header("Step 1: Download Template")
st.write("Your CSV must have these exact columns:")

template=pd.DataFrame(columns=FEATURES)
template.loc[0]=[45, 65000, 15, 1200, 2, 3, 8, 17]
st.dataframe(template,use_container_width=True)

csv_template=template.to_csv(index=False)
st.download_button(
    label="⬇️ Download Template CSV",
    data=csv_template,
    file_name="customer_template.csv",
    mime="text/csv"
)

st.markdown("---")


st.header("Step 2: Upload Your Customer Data")

uploaded_file=st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    
    st.success(f"✅ Loaded {len(df)} customers!")
    st.subheader("Preview of Your Data")
    st.dataframe(df.head(), use_container_width=True)
    
 
    missing=[col for col in FEATURES if col not in df.columns]
    
    if missing:
        st.error(f"❌ Missing columns:{', '.join(missing)}")
        st.info("Please add these columns. Use the template above.")
    else:
        st.success("✅ All required columns found!")
        
        st.markdown("---")
        
      
        st.header("Step 3: Get Predictions")
        
        if st.button("🚀 Predict for All Customers", type="primary", use_container_width=True):
            
            
            X=df[FEATURES].fillna(0)
            X_scaled = scaler.transform(X)
            
            
            predictions=model.predict(X_scaled)
            probabilities=model.predict_proba(X_scaled)[:, 1]
            
            
            df['Response_Probability']=(probabilities * 100).round(1)
            df['Will_Respond']=predictions
            df['Recommendation']=df['Will_Respond'].map({1: '✅ Target', 0: '❌ Skip'})
            
            st.success("✅ Prediction complete!")
            
            
            col1, col2, col3=st.columns(3)
            with col1:
                st.metric("Total Customers",len(df))
            with col2:
                st.metric("Likely to Respond",f"{df['Will_Respond'].sum()} ({df['Will_Respond'].mean():.0%})")
            with col3:
                st.metric("Recommended to Skip",f"{(df['Will_Respond']==0).sum()} ({(df['Will_Respond']==0).mean():.0%})")
            
            
            st.subheader("Results Preview (First 10 Customers)")
            display_cols = FEATURES[:3]+['Response_Probability','Recommendation']
            st.dataframe(df[display_cols].head(10),use_container_width=True)
            
            
            st.markdown("---")
            st.subheader("Step 4: Download Results")
            
            csv_output=df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Predictions (CSV)",
                data=csv_output,
                file_name="campaign_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.info("📁 The downloaded CSV has ALL original data PLUS: Response_Probability, Will_Respond, and Recommendation columns.")