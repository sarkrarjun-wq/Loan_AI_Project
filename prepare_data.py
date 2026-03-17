import pandas as pd
import numpy as np
import os

# Create data directory if it doesn't exist (safety check)
if not os.path.exists('data'):
    os.makedirs('data')

def merge_datasets():
    print("Loading datasets...")
    
    # Load the real datasets from your 'data' folder
    # Note: Replace these filenames with the exact names of the files you downloaded
    try:
        sba = pd.read_csv('data/SBAnational.csv', low_memory=False)
        kaggle = pd.read_csv('data/loan_approval_dataset.csv')
    except FileNotFoundError:
        print("Error: CSV files not found in the 'data' folder!")
        return

    # 1. Take 'Industry' and 'Business Age' logic from SBA
    # We take a sample to keep it manageable
    sba_sample = sba[['NAICS', 'Term', 'NoEmp', 'CreateJob']].sample(1000).reset_index(drop=True)

    # 2. Take 'Income' and 'CIBIL' logic from Kaggle
    kaggle_sample = kaggle[['income_annum', 'loan_amount', 'cibil_score']].sample(1000).reset_index(drop=True)

    # 3. Combine them side-by-side (Horizontal Concatenation)
    hybrid_df = pd.concat([sba_sample, kaggle_sample], axis=1)

    # 4. Add your "Advanced" Banking Features (Feature Engineering)
    # DSCR = (Income * 0.3) / (Loan Amount / Term)
    hybrid_df['DSCR'] = (hybrid_df['income_annum'] * 0.3) / (hybrid_df['loan_amount'] / 12)
    hybrid_df['DSCR'] = hybrid_df['DSCR'].replace([np.inf, -np.inf], 1.5).fillna(1.2).round(2)

    # GST Compliance Score (Randomly generated based on CIBIL)
    hybrid_df['GST_Score'] = (hybrid_df['cibil_score'] / 100) + np.random.randint(-1, 1, size=len(hybrid_df))
    hybrid_df['GST_Score'] = hybrid_df['GST_Score'].clip(1, 10)

    # 5. Define the Target (Loan Status)
    # Logic: Approved if CIBIL > 600 and DSCR > 1.1
    hybrid_df['Loan_Status'] = np.where(
        (hybrid_df['cibil_score'] > 600) & (hybrid_df['DSCR'] > 1.1), 
        'Approved', 'Rejected'
    )

    # Save the final hybrid dataset
    hybrid_df.to_csv('data/final_hybrid_loan_data.csv', index=False)
    print("Success: 'final_hybrid_loan_data.csv' created in the data folder!")

if __name__ == "__main__":
    merge_datasets()