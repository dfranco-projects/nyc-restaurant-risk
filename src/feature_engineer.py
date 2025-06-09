from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import joblib

tqdm.pandas()

class FeatureEngineeringPipeline:
    def __init__(self, mode='train'):
        assert mode in ['train', 'unseen'], "Mode must be 'train' or 'unseen'"
        self.mode = mode
        self.onehot_encoders = {}
        self.target_encoders = {}
        self.power_transformer = None
        self.tfidf_vectorizer = None
        self.negative_keywords = [
            'rodent', 'contaminated', 'vermin', 'infestation', 'pest', 
            'mice', 'rats', 'filth', 'sewage', 'unclean', 'hazard', 
            'sanitation', 'cockroach', 'unsanitary'
        ]

    def transform(self, df):
        df = df.copy()
        df = self._apply_temporal_features(df)
        df = self._apply_geographical_features(df)
        df = self._apply_general_encoding(df)
        df = self._apply_text_features(df)
        df = self._apply_restaurant_features(df)
        return df

    def _apply_temporal_features(self, df):
        df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')
        df['inspection_month'] = df['inspection_date'].dt.month
        df['inspection_year'] = df['inspection_date'].dt.year
        df['inspection_season'] = df['inspection_month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        return df

    def _apply_geographical_features(self, df):
        if self.mode == 'train':
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
            borough_encoded = ohe.fit_transform(df[['borough']])
            self.onehot_encoders['borough'] = ohe
        else:
            ohe = self.onehot_encoders.get('borough')
            borough_encoded = ohe.transform(df[['borough']])
        borough_cols = [f'borough_{cat}' for cat in ohe.categories_[0]]
        borough_df = pd.DataFrame(borough_encoded, columns=borough_cols, index=df.index)
        df = pd.concat([df.drop(columns=['borough']), borough_df], axis=1)
        return df

    def _apply_general_encoding(self, df):
        cat_cols = df.select_dtypes(include=['object', 'category']).nunique()
        low_card = cat_cols[cat_cols <= 5].index
        high_card = cat_cols[cat_cols > 5].index

        for col in low_card:
            if self.mode == 'train':
                ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
                enc = ohe.fit_transform(df[[col]])
                self.onehot_encoders[col] = ohe
            else:
                ohe = self.onehot_encoders.get(col)
                enc = ohe.transform(df[[col]])
            ohe_df = pd.DataFrame(enc, columns=[f'{col}_{cat}' for cat in ohe.categories_[0]], index=df.index)
            df = pd.concat([df.drop(columns=[col]), ohe_df], axis=1)

        for col in high_card:
            if self.mode == 'train':
                enc = TargetEncoder()
                df[col] = enc.fit_transform(df[col], df['critical_flag'])
                self.target_encoders[col] = enc
            else:
                enc = self.target_encoders.get(col)
                df[col] = enc.transform(df[col])
        if 'score' in df.columns:
            df['score'] = df['score'].fillna(df['score'].median())
            if self.mode == 'train':
                pt = PowerTransformer(method='box-cox')
                df['score'] = pt.fit_transform(df[['score']])
                self.power_transformer = pt
            else:
                pt = self.power_transformer
                df['score'] = pt.transform(df[['score']])
        return df

    def _apply_text_features(self, df):
        df['desc_length_words'] = df['violation_description'].apply(lambda x: len(x.split()))
        df['desc_length_chars'] = df['violation_description'].apply(lambda x: len(x))

        def keyword_intensity(text):
            return sum(1 for word in text.split() if word in self.negative_keywords)
        
        df['negative_keywords_count'] = df['violation_description'].apply(keyword_intensity)

        if self.mode == 'train':
            tfidf = TfidfVectorizer(max_features=20)
            tfidf_matrix = tfidf.fit_transform(df['violation_description'])
            self.tfidf_vectorizer = tfidf
        else:
            tfidf = self.tfidf_vectorizer
            tfidf_matrix = tfidf.transform(df['violation_description'])

        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{w}' for w in tfidf.get_feature_names_out()], index=df.index)
        df = pd.concat([df, tfidf_df], axis=1)

        return df

    def _apply_restaurant_features(self, df):
        df.sort_values(by=['camis', 'inspection_date'], inplace=True)

        df['avg_score_to_date'] = (
            df.groupby('camis')['score']
            .expanding().mean()
            .reset_index(level=0, drop=True)
        )

        df['inspection_count'] = (
            df.groupby('camis').cumcount() + 1
        )

        def days_since_last(grp):
            return grp['inspection_date'].diff().dt.days.fillna(0)

        df['days_since_last'] = df.groupby('camis').apply(days_since_last).reset_index(level=0, drop=True)

        return df

