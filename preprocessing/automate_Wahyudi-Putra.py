import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def handle_outliers(df, columns):
    df_outlier = df.copy()
    for col in columns:
        Q1 = df_outlier[col].quantile(0.25)
        Q3 = df_outlier[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_outlier[col] = df_outlier[col].clip(lower=lower_bound, upper=upper_bound)
    return df_outlier

def run_automation(file_path, output_path='processed_data.csv'):
    
    # 1. Load Data
    df = pd.read_csv(file_path, encoding='latin-1')
    df_clean = df.copy()
    
    # 2. Menangani Missing Values
    cols_to_fix = ['goalsPrevented', 'expectedGoals', 'expectedAssists']
    for col in cols_to_fix:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
    
    # 3. Penanganan Outlier
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'rating' in numeric_cols: 
        numeric_cols.remove('rating')
    df_clean = handle_outliers(df_clean, numeric_cols)
    
    # 4. Drop kolom identitas
    df_clean = df_clean.drop(columns=['player_name'], errors='ignore')
    
    # 5. Feature Selection & Transformer Definition
    categorical_features = ['position', 'team_name']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # 6. Pemisahan Fitur dan Target
    X = df_clean.drop(columns=['rating'])
    y = df_clean['rating']
    
    # 7. Transformasi Data
    X_transformed = preprocessor.fit_transform(X)
    
         
    # Simpan data hasil transformasi ke CSV
    # ubah array ke DataFrame
    X_df = pd.DataFrame(X_transformed.toarray() if hasattr(X_transformed, 'toarray') else X_transformed)
    X_df['target_rating'] = y.values
    X_df.to_csv(output_path, index=False)

       
    print(f"Preprocessing completed. Dataset berhasil disimpan di: {output_path}")
    return X_transformed, y, preprocessor  

if __name__ == "__main__":
    DATA_FILE = 'premier_league_complete_stats_until31thGameDayOnSeason2025-26_raw.csv'
    X, y, transformer = run_automation(DATA_FILE)