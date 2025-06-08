import numpy as np
import pandas as pd
import geopandas as gpd

class DataCleaner:
    def __init__(self, data: pd.DataFrame | gpd.GeoDataFrame):
        self.data = data.copy()

    def drop_uninformative_columns(self, drop_location_codes=True):
        '''Drop fully empty, constant, or not relevant columns.'''
        cols_to_drop = ['location_point1', 'record_date', 'grade_date']
        if drop_location_codes:
            cols_to_drop += [
                'community_board', 'council_district', 'census_tract',
                'bin', 'bbl', 'nta'
            ]
        # Drop columns that are all NaN or constant
        for col in self.data.columns:
            if self.data[col].nunique(dropna=False) <= 1:
                cols_to_drop.append(col)
        self.data = self.data.drop(columns=[c for c in set(cols_to_drop) if c in self.data.columns])

    def drop_invalid_inspection_dates(self):
        '''Remove rows with missing or invalid inspection dates.'''
        self.data = self.data[self.data['inspection_date'].notna()]
        self.data = self.data[self.data['inspection_date'] != pd.Timestamp('1900-01-01')]

    def drop_missing_key_fields(self):
        '''Drop rows with missing critical_flag, violation_code, or score.'''
        self.data = self.data.dropna(subset=['critical_flag', 'violation_code', 'score'])

    def drop_health_code_rows(self):
        '''Drop rows where violation_code does not contain any letters.'''
        mask = self.data['violation_code'].astype(str).str.contains('[A-Za-z]')
        self.data = self.data[mask]

    def handle_duplicates(self):
        '''Drop fully duplicated rows and ensure unique key per camis, inspection_date, violation_code.'''
        self.data = self.data.drop_duplicates()
        # For partial duplicates, keep the row with the highest score if critical, else lowest
        if {'inspection_id', 'violation_code', 'critical_flag', 'score'}.issubset(self.data.columns):
            def resolve_group(g):
                if 'Critical' in g['critical_flag'].values:
                    return g.loc[g['score'].idxmax()]
                else:
                    return g.loc[g['score'].idxmin()]
            self.data = self.data.groupby(['inspection_id', 'violation_code'], as_index=False).apply(resolve_group)
            self.data = self.data.reset_index(drop=True)

    def fix_grades_by_score(self):
        '''Fix grade feature according to score and DOHMH thresholds.'''
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
        # Optionally, overwrite grade with fixed value where inconsistent
        inconsistent = self.data['grade'] != self.data['grade_fixed']
        self.data.loc[inconsistent, 'grade'] = self.data.loc[inconsistent, 'grade_fixed']
        self.data = self.data.drop(columns=['grade_fixed'])

    def handle_missing_coordinates(self, fill_with_zero=True):
        '''Impute missing or zero coordinates if spatial analysis is not critical.'''
        if fill_with_zero:
            self.data['latitude'] = self.data['latitude'].fillna(0)
            self.data['longitude'] = self.data['longitude'].fillna(0)

    def normalize_text_fields(self):
        '''Standardize text fields to reduce cardinality.'''
        text_cols = ['dba', 'street', 'violation_description']
        for col in text_cols:
            if col in self.data.columns:
                self.data[col] = (
                    self.data[col]
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    .str.replace(r'[^a-z0-9\s]', '', regex=True)
                )

    def reduce_high_cardinality(self, col, top_n=10, other_label='other'):
        '''Encode only the most frequent categories, group others as "other".'''
        if col in self.data.columns:
            top = self.data[col].value_counts().nlargest(top_n).index
            self.data[col] = self.data[col].where(self.data[col].isin(top), other_label)

    def final_sanity_checks(self):
        '''Print missing values in key fields.'''
        key_fields = ['grade', 'critical_flag', 'violation_code', 'score', 'inspection_date']
        print(self.data[key_fields].isnull().sum())

    def clean_data(self):
        '''Perform all cleaning operations.'''
        self.drop_uninformative_columns()
        self.drop_invalid_inspection_dates()
        self.drop_missing_key_fields()
        self.drop_health_code_rows()
        self.handle_duplicates()
        self.fix_grades_by_score()
        self.handle_missing_coordinates()
        self.normalize_text_fields()
        self.reduce_high_cardinality('dba', top_n=20)
        self.reduce_high_cardinality('violation_description', top_n=20)
        self.final_sanity_checks()
        return self.data