import pandas as pd
import numpy as np
import os

# Create data directory
if not os.path.exists('data'):
    os.makedirs('data')

def generate_hybrid_data(n=2000):
    np.random.seed(42)
    data = {
        'Years_In_Business': np.random.randint(1, 20, n),
        'CIBIL_Score': np.random.randint(300, 900, n),
        'Annual_Revenue': np.random.randint(500000, 5000000, n),
        'Loan_Amount': np.random.randint(100000, 2000000, n),
    }
    df = pd.DataFrame(data)

    # Advanced Feature Engineering
    df['GST_Score'] = (df['CIBIL_Score'] / 100) + np.random.normal(0, 0.5, n)
    df['GST_Score'] = df['GST_Score'].clip(1, 10).round(1)
    
    # DSCR = (Profit) / (Debt Obligations)
    df['DSCR'] = (df['Annual_Revenue'] * 0.25) / (df['Loan_Amount'] * 0.15)
    df['DSCR'] = df['DSCR'].round(2)
    
    df['Debt_to_Equity'] = np.random.uniform(0.1, 4.0, n).round(2)

    # AI Target Logic: Approved if financials are strong
    df['Loan_Status'] = np.where(
        (df['CIBIL_Score'] > 650) & (df['DSCR'] > 1.2) & (df['GST_Score'] > 6),
        'Approved', 'Rejected'
    )
    
    df.to_csv('data/final_hybrid_loan_data.csv', index=False)
    print("✅ Dataset 'data/final_hybrid_loan_data.csv' created!")

if __name__ == "__main__":
    generate_hybrid_data()