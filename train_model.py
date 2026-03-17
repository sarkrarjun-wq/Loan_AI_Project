import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load generated data
df = pd.read_csv('data/final_hybrid_loan_data.csv')

# Select Features
X = df[['Years_In_Business', 'CIBIL_Score', 'GST_Score', 'DSCR', 'Debt_to_Equity']]
y = df['Loan_Status']

# Encode 'Approved'/'Rejected' to numbers
le = LabelEncoder()
y = le.fit_transform(y)

# Train XGBoost Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = XGBClassifier()
model.fit(X_train, y_train)

# Save the model and encoder
joblib.dump(model, 'loan_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("🧠 AI Model trained and saved as 'loan_model.pkl'!")