import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Loading data...")


df = pd.read_csv('marketing_campaign.csv', sep=';')

print(f"Data loaded! Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")


df=df.dropna(subset=['Income'])


df['Age']=2024-df['Year_Birth']


spend_cols=['MntWines', 'MntFruits', 'MntMeatProducts', 
              'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df['TotalSpend']=df[spend_cols].sum(axis=1)


df['Children']=df['Kidhome']+df['Teenhome']

print(f"Cleaned data! Shape:{df.shape}")


FEATURES=['Age','Income','Recency','TotalSpend','Children', 
            'NumDealsPurchases','NumWebPurchases','NumWebVisitsMonth']

X=df[FEATURES]
y=df['Response']

print(f"\nFeatures:{FEATURES}")
print(f"X shape:{X.shape}")
print(f"Response rate:{y.mean():.2%}")


X_train,X_test,y_train,y_test =train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


model=LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


y_pred=model.predict(X_test_scaled)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nFeature Importance (Coefficients):")
for feature, coef in zip(FEATURES, model.coef_[0]):
    print(f"  {feature}: {coef:.4f}")


pickle.dump(model,open('model.pkl', 'wb'))
pickle.dump(scaler,open('scaler.pkl', 'wb'))
pickle.dump(FEATURES,open('features.pkl', 'wb'))

