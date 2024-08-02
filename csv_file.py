import pandas as pd
import numpy as np

# Set the random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 284807

# Generate random data
data = {
    'Time': np.random.randint(0, 172792, size=n_samples),
    'V1': np.random.normal(0, 1, size=n_samples),
    'V2': np.random.normal(0, 1, size=n_samples),
    'V3': np.random.normal(0, 1, size=n_samples),
    'V4': np.random.normal(0, 1, size=n_samples),
    'V5': np.random.normal(0, 1, size=n_samples),
    'V6': np.random.normal(0, 1, size=n_samples),
    'V7': np.random.normal(0, 1, size=n_samples),
    'V8': np.random.normal(0, 1, size=n_samples),
    'V9': np.random.normal(0, 1, size=n_samples),
    'V10': np.random.normal(0, 1, size=n_samples),
    'V11': np.random.normal(0, 1, size=n_samples),
    'V12': np.random.normal(0, 1, size=n_samples),
    'V13': np.random.normal(0, 1, size=n_samples),
    'V14': np.random.normal(0, 1, size=n_samples),
    'V15': np.random.normal(0, 1, size=n_samples),
    'V16': np.random.normal(0, 1, size=n_samples),
    'V17': np.random.normal(0, 1, size=n_samples),
    'V18': np.random.normal(0, 1, size=n_samples),
    'V19': np.random.normal(0, 1, size=n_samples),
    'V20': np.random.normal(0, 1, size=n_samples),
    'V21': np.random.normal(0, 1, size=n_samples),
    'V22': np.random.normal(0, 1, size=n_samples),
    'V23': np.random.normal(0, 1, size=n_samples),
    'V24': np.random.normal(0, 1, size=n_samples),
    'V25': np.random.normal(0, 1, size=n_samples),
    'V26': np.random.normal(0, 1, size=n_samples),
    'V27': np.random.normal(0, 1, size=n_samples),
    'V28': np.random.normal(0, 1, size=n_samples),
    'Amount': np.random.gamma(2, 50, size=n_samples),  # Gamma distribution for Amount
    'Class': np.concatenate((np.zeros(n_samples - 492), np.ones(492)))  # 492 fraud cases
}

# Shuffle the dataset
df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV
df.to_csv('credit.csv', index=False)

print("credit.csv file created successfully!")
