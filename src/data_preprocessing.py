import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl


class preprocess:         
    @staticmethod
    def pre_process(df):
        df = df.drop_duplicates()
        df = df.drop(columns=["loyalty_program", "churned", "education_level", "transaction_id", "transaction_date", "payment_method", "store_location", "transaction_hour", "day_of_week", "week_of_year", "month_of_year", "last_purchase_date", "preferred_store", "total_returned_items", "total_returned_value", "total_transactions", "product_rating", "product_review_count", "product_stock", "product_return_rate", "product_color", "product_manufacture_date", "product_expiry_date", "product_shelf_life", "promotion_type", "promotion_channel", "store_zip_code", "distance_to_store", "customer_support_calls"])

        date_cols = []

        # collect date columns
        for i in df.columns:
            if 'date' in i.lower():
                date_cols.append(i)

        for col in date_cols:
            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')
            df[col] = df[col].astype('int64') / 1e9  # convert to float seconds

        numeric_cols = df.select_dtypes(include=['number']).columns
        # Outlier removal
        def whisker(col):
            Q1,Q3 = np.percentile(col,[25,75])
            iqr = Q3 - Q1
            lw = Q1 - (1.5 * iqr)
            uw = Q3 + (1.5 * iqr)
            return lw, uw

        for i in numeric_cols:
            lw,uw = whisker(df[i])
            df[i] = np.where(df[i]<lw, lw, df[i])
            df[i] = np.where(df[i]>uw, uw, df[i])

        number = df.select_dtypes(include=np.number).columns.tolist()

        for col in number:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std

        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        # One-hot encode categorical columns
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=float)

        return df
    
        

    
