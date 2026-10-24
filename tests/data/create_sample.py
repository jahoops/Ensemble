import pandas as pd
import numpy as np

df = pd.DataFrame({
    'PremiumAmount': np.random.uniform(100, 1000, 100),
    'Age': np.random.randint(18, 80, 100),
    'Coverage': np.random.uniform(10000, 100000, 100)
})
df.to_csv('tests/data/processed/sample_train.csv', index=False)