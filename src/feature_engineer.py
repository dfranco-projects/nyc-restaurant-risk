from sklearn.preprocessing import MinMaxScaler, PowerTransformer, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk
import re

nltk.download('punkt')
nltk.download('punkt_tab')

class FeatureEngineeringPipeline:
    def __init__(self, df, mode='train', target='critical_flag'):
        '''initialize pipeline with data, mode, and target'''
        assert mode in ['train', 'unseen'], "Mode must be 'train' or 'unseen'"

        self.mode = mode
        self.df = df.copy()
        self.train_df = None
        self.target = target
        self.onehot_encoders = {}
        self.onehot_kept_columns = {}
        self.target_encoders = {}
        self.sentiment_cache = {}
        self.power_transformer = None
        self.tfidf_vectorizer = None
        self.tfidf_selector = None
        self.tfidf_selected_features = None
        self.minmax_scaler = None
        self.minmax_features = []
        self.top_negative_tokens = []
        self.sentiment_analyzer = pipeline(
            'sentiment-analysis',
            model='distilbert/distilbert-base-uncased-finetuned-sst-2-english'
        )

    def set_mode_to_unseen(self, df):
        '''switch to unseen mode and reset df'''
        self.mode = 'unseen'
        self.df = df.copy()
        return self

    def _apply_temporal_features(self):
        '''extract date-related features'''
        self.df['inspection_month'] = self.df['inspection_date'].dt.month
        self.df['inspection_year'] = self.df['inspection_date'].dt.year
        self.df['inspection_season'] = self.df['inspection_month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        return self

    def _apply_geographical_features(self):
        '''placeholder for geo features'''
        return self

    def _encode_target(self):
        '''convert critical_flag to binary'''
        assert self.target == 'critical_flag', "Target must be 'critical_flag at the moment'"
        mapping = {'Critical': 1, 'Not Critical': 0}
        self.df[self.target] = self.df[self.target].map(mapping)
        return self

    def _extract_critical_tokens(self, top_n=30):
        '''extract discriminative tokens linked to critical violations'''

        # get all descriptions labeled as critical and not critical
        critical_texts = self.df[self.df[self.target] == 1]['violation_description']
        not_critical_texts = self.df[self.df[self.target] == 0]['violation_description']

        # vectorize critical texts using bag-of-words (remove english stopwords)
        vec = CountVectorizer(stop_words='english', ngram_range=(1, 1))
        crit_mat = vec.fit_transform(critical_texts)
        crit_freq = np.asarray(crit_mat.sum(axis=0)).flatten()

        # get token list
        vocab = np.array(vec.get_feature_names_out())

        # use same vocabulary to vectorize non-critical texts
        vec_nc = CountVectorizer(stop_words='english', ngram_range=(1, 1), vocabulary=vec.vocabulary_)
        not_crit_mat = vec_nc.fit_transform(not_critical_texts)
        not_crit_freq = np.asarray(not_crit_mat.sum(axis=0)).flatten()

        # compute frequency ratio between critical and non-critical for each token
        ratio = (crit_freq + 1e-6) / (not_crit_freq + 1e-6)

        # get indices of top tokens with highest ratios
        top_indices = ratio.argsort()[::-1][:top_n]

        # store top tokens as a pipeline attribute
        self.top_negative_tokens = vocab[top_indices].tolist()

        return self

    def _apply_description_features(self):
        '''generate features from violation_description'''
        self.df['desc_length_words'] = self.df['violation_description'].apply(lambda x: len(x.split()))
        self.df['desc_length_chars'] = self.df['violation_description'].apply(lambda x: len(x))

        # compute top discriminative critical tokens during training
        if self.mode == 'train':
            self._extract_critical_tokens(top_n=20)

        # count of discriminative critical tokens
        def critical_token_intensity(text):
            return sum(1 for word in text.lower().split() if word in self.top_negative_tokens)

        self.df['critical_token_count'] = self.df['violation_description'].apply(critical_token_intensity)

        # create sentiment cache
        unique_texts = self.df['violation_description'].unique()
        new_texts = [txt for txt in unique_texts if txt not in self.sentiment_cache]

        if new_texts:

            def analyze(text):
                try:
                    sentences = sent_tokenize(text[:512])
                    results = self.sentiment_analyzer(sentences)
                    return 1 if results[0]['label'].lower() == 'negative' else 0
                except:
                    return 0

            new_results = {text: analyze(text) for text in tqdm(new_texts, desc='Sentiment Analysis')}
            self.sentiment_cache.update(new_results)

        # map sentiment back to full DataFrame
        self.df['negative_sentiment'] = self.df['violation_description'].map(self.sentiment_cache)

        # apply tf-idf vectorization
        if self.mode == 'train':
            tfidf = TfidfVectorizer(
                sublinear_tf=True,
                analyzer='word',
                ngram_range=(1, 3), # allow trigrams
                max_features=1_000, # keep top 1000 features by term frequency across corpus
            )
            tfidf_matrix = tfidf.fit_transform(self.df['violation_description'])
            self.tfidf_vectorizer = tfidf

            # select top 10 features by f-classif score
            selector = SelectKBest(score_func=f_classif, k=10)
            X_selected = selector.fit_transform(tfidf_matrix, self.df[self.target])
            selected_feature_names = selector.get_support(indices=True)
            top_feature_names = [f'tfidf_{tfidf.get_feature_names_out()[i]}' for i in selected_feature_names]

            self.tfidf_selector = selector
            self.tfidf_selected_features = top_feature_names
            
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(self.df['violation_description'])
            X_selected = self.tfidf_selector.transform(tfidf_matrix)
            top_feature_names = self.tfidf_selected_features

        tfidf_df = pd.DataFrame(X_selected.toarray(), columns=top_feature_names, index=self.df.index)

        self.df = pd.concat([self.df, tfidf_df], axis=1)

        return self

    def _apply_general_encoding(self):
        '''apply one-hot and target encoding to categorical features'''
        cat_cols = self.df.select_dtypes(include=['string', 'object', 'category']).nunique()
        low_card = [col for col in cat_cols[cat_cols <= 5].index if col != self.target]
        not_encode = ['inspection_id', 'camis', 'violation_description']
        high_card = [col for col in cat_cols[(cat_cols > 5)].index if col not in not_encode]

        # one-hot encode low-cardinality categorical features
        for col in low_card:
            if self.mode == 'train':
                ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                enc = ohe.fit_transform(self.df[[col]])
                self.onehot_encoders[col] = ohe
                kept_cols = ohe.get_feature_names_out([col]).tolist()
                self.onehot_kept_columns[col] = kept_cols
            else:
                ohe = self.onehot_encoders.get(col)
                kept_cols = self.onehot_kept_columns.get(col, [])
                enc = ohe.transform(self.df[[col]])

            ohe_df = pd.DataFrame(enc, columns=kept_cols, index=self.df.index)

            # ensure all expected columns exist
            for expected_col in kept_cols:
                if expected_col not in ohe_df.columns:
                    ohe_df[expected_col] = 0
            ohe_df = ohe_df[kept_cols]  # enforce column order

            self.df = pd.concat([self.df.drop(columns=[col]), ohe_df], axis=1)

        # target encode high-cardinality categorical features
        for col in high_card:
            if self.mode == 'train':
                enc = TargetEncoder()
                self.df[col] = enc.fit_transform(self.df[col], self.df[self.target])
                self.target_encoders[col] = enc
            else:
                enc = self.target_encoders.get(col)
                self.df[col] = enc.transform(self.df[col])

        # score transformation
        min_val = self.df['score'].min()
        if min_val <= 0:
            self.df['score'] += abs(min_val) + 1e-6

        if self.mode == 'train':
            pt = PowerTransformer(method='box-cox')
            self.df['score'] = pt.fit_transform(self.df[['score']])
            self.power_transformer = pt
        else:
            self.df['score'] = self.power_transformer.transform(self.df[['score']])

        return self

    def _apply_restaurant_features(self):
        '''generate features based on restaurant history'''
        if self.mode == 'train':
            df_sorted = self.df.sort_values(by=['camis', 'inspection_date']).copy()
        else:
            # combine train and unseen for computing rolling features
            combined_df = pd.concat([self.train_df, self.df], axis=0)
            df_sorted = combined_df.sort_values(by=['camis', 'inspection_date']).copy()

        # compute rolling features
        df_sorted['avg_score_to_date'] = (
            df_sorted.groupby('camis')['score']
            .expanding().mean()
            .reset_index(level=0, drop=True)
        )

        df_sorted['inspection_count'] = (
            df_sorted.groupby('camis').cumcount() + 1
        )

        df_sorted['days_since_last'] = (
            df_sorted.groupby('camis')['inspection_date']
            .diff().dt.days.fillna(0)
        )

        if self.mode == 'train':
            self.df = df_sorted
        else:
            self.df[['avg_score_to_date', 'inspection_count', 'days_since_last']] = \
                df_sorted.loc[self.df.index, ['avg_score_to_date', 'inspection_count', 'days_since_last']]

        return self

    def _set_key_as_index(self):
        '''ensure "key" column is the index'''
        if self.df.index.name != 'key':
            self.df = self.df.set_index('key')
        return self

    def _drop_old_features(self):
        '''drop non-predictive or redundant columns'''
        old_feat = [
            'inspection_id', 'violation_description',
            'inspection_date', 'camis', 
            'latitude', 'longitude'
        ]
        self.df = self.df.drop(columns=old_feat)
        return self

    def _save_train_processed_df(self):
        '''store processed train data to reuse in unseen'''
        if self.mode == 'train':
            self.train_df = self.df.copy()
        return self

    def _normalize_feature_names(self):
        '''normalize text fields'''
        normalize = lambda name: '_'.join(re.findall(r'[A-Za-z]+', name)).lower()
        for col in self.df.columns:
            self.df = self.df.rename(columns={col: normalize(col)})
        return self

    def _apply_minmax_scaling(self):
        '''scale numerical features to [0,1] range'''
        # get features to scale (excluding already transformed ones like tf-idf)
        exclude_cols = set([
            self.target,
            *[col for col in self.df.columns if col.startswith('tfidf_')]
        ])
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        features_to_scale = [col for col in num_cols if col not in exclude_cols]

        if self.mode == 'train':
            self.minmax_scaler = MinMaxScaler()
            self.df[features_to_scale] = self.minmax_scaler.fit_transform(self.df[features_to_scale])
            self.minmax_features = features_to_scale
        else:
            self.df[self.minmax_features] = self.minmax_scaler.transform(self.df[self.minmax_features])

        return self

    def transform(self):
        '''run the full feature pipeline'''

        steps = [
            ('_set_key_as_index', self._set_key_as_index),
            ('_encode_target', self._encode_target),
            ('_apply_temporal_features', self._apply_temporal_features),
            ('_apply_restaurant_features', self._apply_restaurant_features),
            ('_apply_geographical_features', self._apply_geographical_features),
            ('_apply_general_encoding', self._apply_general_encoding),
            ('_apply_description_features', self._apply_description_features),
            ('_apply_minmax_scaling', self._apply_minmax_scaling),
            ('_drop_old_features', self._drop_old_features),
            ('_normalize_feature_names', self._normalize_feature_names),
            ('_save_train_processed_df', self._save_train_processed_df),
        ]

        for name, func in tqdm(steps, desc='Feature Engineering Steps', unit='step'):
            func()

        return self.df