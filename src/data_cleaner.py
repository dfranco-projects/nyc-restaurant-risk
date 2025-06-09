import re
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class DataCleaner:
    def __init__(self, data: pd.DataFrame | gpd.GeoDataFrame):
        self.data = data.copy()

    def drop_missing_key_fields(self):
        '''Drop rows with missing critical_flag, violation_code, or score.'''
        self.data = self.data.dropna(subset=['critical_flag', 'violation_code', 'score', 'inspection_date'])

    def create_id(self):
        '''Create inspection_id and key columns, set index to key.'''
        self.data['inspection_id'] = (
            self.data['camis'].astype(str) + '_' + self.data['inspection_date'].dt.strftime('%d-%m-%y')
        )
        self.data['inspection_id'] = self.data['inspection_id'].astype(str)

        self.data['key'] = (
            self.data['camis'].astype(str) + '_' +
            self.data['inspection_date'].dt.strftime('%d-%m-%y') + '_' +
            self.data['violation_code'].astype(str)
        )

        self.data = self.data.set_index('key')

    def drop_uninformative_columns(self, drop_location_codes=True):
        '''Drop fully empty, constant, or not relevant columns.'''
        cols_to_drop = ['location_point1', 'record_date', 'grade_date', 'phone']
        if drop_location_codes:
            cols_to_drop += [
                'community_board', 'council_district', 'census_tract',
                'bin', 'bbl', 'nta'
            ]
        for col in self.data.columns:
            if self.data[col].nunique(dropna=False) <= 1:
                cols_to_drop.append(col)
        self.data = self.data.drop(columns=[c for c in set(cols_to_drop) if c in self.data.columns])

    def drop_invalid_inspection_dates(self):
        '''Remove rows with invalid inspection dates.'''
        self.data = self.data[self.data['inspection_date'] != pd.Timestamp('1900-01-01')]

    def drop_health_code_rows(self):
        '''Drop rows where violation_code does not contain any letters.'''
        mask = self.data['violation_code'].astype(str).str.contains('[A-Za-z]')
        self.data = self.data[mask]

    def handle_duplicates(self):
        '''Drop exact duplicates and keep best row per key'''
        # drop exact duplicates
        self.data = self.data.reset_index().drop_duplicates()

        # check key duplicates but with score difference (if critical keep highest otherwise lowest)
        self.data = self.data.sort_values(['key', 'critical_flag', 'score'], ascending=False)
        self.data = self.data.drop_duplicates(subset='key', keep='first')
        self.data = self.data.set_index('key')

    def fix_grades_by_score(self):
        '''Fix grade feature according to score and DOHMH thresholds.'''
        # map N, Z, P to 'Not Graded'
        self.data['grade'] = self.data['grade'].replace({'N': 'Not Graded', 'Z': 'Not Graded', 'P': 'Not Graded'})

        def grade_from_score(score):
            if pd.isna(score):
                return np.nan
            if score <= 13:
                return 'A'
            elif score <= 27:
                return 'B'
            else:
                return 'C'
        self.data['grade_fixed'] = self.data['score'].apply(grade_from_score)
        inconsistent = self.data['grade'] != self.data['grade_fixed']
        self.data.loc[inconsistent, 'grade'] = self.data.loc[inconsistent, 'grade_fixed']
        self.data = self.data.drop(columns=['grade_fixed'])

    def normalize_text_fields(self):
        '''Standardize text fields to reduce cardinality.'''

        def string_preprocess(series: pd.Series, lemmatize: bool = True, stem: bool = False) -> pd.Series:
            '''Clean and preprocess a pandas Series of strings.'''
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer() if lemmatize else None
            stemmer = SnowballStemmer('english') if stem else None

            def clean_text(text):
                if not isinstance(text, str):
                    return ''
                
                text = text.lower()
                text = re.sub(r'<[^>]+>', ' ', text) 
                text = re.sub(r'http\S+', '', text)
                text = re.sub(r'\d+', '', text)
                text = re.sub(r'[^a-zA-Z\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()

                tokens = [word for word in text.split() if word not in stop_words]
                tokens = [word for word in tokens if len(word) > 1]

                if lemmatizer:
                    tokens = [lemmatizer.lemmatize(word) for word in tokens]
                if stemmer:
                    tokens = [stemmer.stem(word) for word in tokens]

                return ' '.join(tokens)

            return series.apply(clean_text)

        # fully normalize violation_description
        self.data['violation_description'] = string_preprocess(self.data['violation_description'], lemmatize=True, stem=False)
        self.data['violation_description'] = self.data['violation_description'].astype(str)
        
        title_cols = ['dba', 'street']
        for col in title_cols:
            self.data[col] = self.data[col].str.title()

    def reduce_high_cardinality(self, col, min_prop=0.04, other_label='Other'):
        '''
        For categorical columns (except id, restaurant, inspection & violation-related fields), 
        group categories with less than min_prop proportion as "other".
        '''
        keep_cols = {
            'camis', 'dba', 'street', 'zipcode', 'phone', 'building',
            'violation_code', 'violation_description', 'inspection_id'
        }
        if col not in keep_cols:
            value_counts = self.data[col].value_counts(normalize=True)
            to_keep = value_counts[value_counts >= min_prop].index
            self.data[col] = self.data[col].where(self.data[col].isin(to_keep), other_label)

    def fill_missing_values(self):
        '''Fill missing values for zipcode, building, and grade as specified.'''
        # phone, zipcode & building
        self.data['zipcode'] = self.data['zipcode'].fillna('00000')
        self.data['building'] = self.data['building'].fillna('0')

        # grade
        inspection_grades = (
            self.data.dropna(subset=['grade'])
            .groupby('inspection_id')['grade']
            .first()
        )
        self.data['grade'] = self.data['inspection_id'].map(inspection_grades)
        self.data['grade'] = self.data['grade'].fillna('Non-gradable')

        # latitude & longitude
        self.data['latitude'] = self.data['latitude'].fillna(0)
        self.data['longitude'] = self.data['longitude'].fillna(0)

    def clean_data(self):
        '''Perform all cleaning operations in order.'''
        steps = [
            ('drop_missing_key_fields', self.drop_missing_key_fields),
            ('create_id', self.create_id),
            ('drop_uninformative_columns', self.drop_uninformative_columns),
            ('drop_invalid_inspection_dates', self.drop_invalid_inspection_dates),
            ('handle_duplicates', self.handle_duplicates),
            ('drop_health_code_rows', self.drop_health_code_rows),
            ('fix_grades_by_score', self.fix_grades_by_score),
            ('fill_missing_values', self.fill_missing_values),
            ('normalize_text_fields', self.normalize_text_fields),
        ]

        for name, func in tqdm(steps, desc='cleaning steps', unit='step'):
            func()

        # get high cardinality features (higher than 7 unique vals)
        high_cardinality_cols = [
            col for col in self.data.select_dtypes(include=['string', 'object', 'category']).columns
            if self.data[col].nunique(dropna=False) > 7
        ]

        # reduce high cardinality on some of these features
        for col in tqdm(high_cardinality_cols, desc='reducing cardinality', unit='col'):
            self.reduce_high_cardinality(col, min_prop=0.04)

        return self.data

