import numpy as np
import pandas as pd
import seaborn as sns
from math import ceil
import matplotlib.pyplot as plt

class DataInspector():
    def __init__(self, df: pd.DataFrame) -> None:
        '''
        Initialize the DataInspector with the data to inspect.
        
        Args:
            df (pd.Dataframe): : Dataframe which the inspection is performed.
        '''
        self.data = df
        self.summarized_data = None

    def inspect(self) -> pd.DataFrame:
        '''
        Builds a pandas dataframe with data types, missing values in absolute and relative frequencies, and summary statistics.

        Returns: 
            pd.DataFrame: A pandas dataframe with the inspection results.
        '''
        df =  self.data

        # Get columns types
        tab_info = pd.DataFrame(df.dtypes).rename(columns={0:'type'})

        # Missing values absolute frequency
        tab_info = pd.concat([tab_info, pd.DataFrame(df.isna().sum()).round().rename(columns={0:'NA Count'})], axis=1)

        # Missing values relative frequency
        tab_info = pd.concat([tab_info, pd.DataFrame(df.isna().sum() / len(df) * 100).rename(columns={0:'NA %'}).round(2)], axis=1)
        
        # Numerical pandas describe method
        tab_info = tab_info.join(df.describe().round(2).T.drop('count', axis=1)).fillna('-')

        # Object pandas describe method
        tab_info = tab_info.join(df.describe(include=['object', 'string']).T.drop('count', axis=1)).fillna('-')

        # Sort dataframe by type
        tab_info['type'] = tab_info['type'].astype(str)
        tab_info = tab_info.sort_values('type', ascending=True)

        self.summarized_data = tab_info

        print('\nData types, Missing values, and Summary statistics:')
        return tab_info
    
    def count_types(self): 
        '''
        Builds a pandas DataFrame that provides the count of features for each data type.

        Returns: 
            pd.DataFrame: A pandas dataframe with the count results.
        '''
        df = self.summarized_data
        print('\nCount of features per data type:')
        return pd.DataFrame(df.type.value_counts()).rename_axis('').T

    def dist_check(self, check_continuous=False, continuous_threshold=20, x=12, y_per_row=3, max_y=12):
        '''
        Displays distribution plots of numerical features in the dataset.

        Args:
            check_continuous (bool): If True, filter out features with fewer unique values than `continuous_threshold`.
            continuous_threshold (int): Minimum unique values for a feature to be considered continuous.
            x (int): Total figure width in inches.
            y_per_row (int): Height per row of plots.
            max_y (int): Max total figure height in inches.

        Returns:
            None: Displays plots.
        '''
        data = self.data

        # gets numeric columns, drop those with only NaNs
        nums = [col for col in data.select_dtypes(include=np.number).columns
                if not data[col].isna().all()]

        if check_continuous:
            nums = [col for col in nums if data[col].nunique(dropna=True) >= continuous_threshold]

        if not nums:
            print('No valid numerical features to plot.')
            return

        max_cols = 3
        total_feats = len(nums)
        n_cols = min(total_feats, max_cols)
        n_rows = ceil(total_feats / n_cols)

        # adjusts height dynamically, clamp with max_y
        y = min(y_per_row * n_rows, max_y)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(x, y), constrained_layout=True)
        axes = np.array(axes).reshape(-1)

        # plotting
        for ax, col in zip(axes, nums):
            sns.histplot(data[col].dropna(), kde=True, color='lightsteelblue', ax=ax)
            ax.set_title(col.replace('_', ' '), y=1.0, fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel('')

        for ax in axes[len(nums):]:
            ax.set_visible(False)

        plt.show()

    @staticmethod
    def shorten_labels(label, max_length=25):
        '''
        Shortens labels to a maximum length, splitting on spaces if possible.
        
        Args:
            labels (list): List of labels to shorten.
            max_length (int): Maximum length of each label.

        Returns:
            list: List of shortened labels.
        '''
        label = str(label)

        if len(label) <= max_length:
            return label
        
        mid = len(label) // 2
        left_space = label.rfind(' ', 0, mid)
        right_space = label.find(' ', mid)

        split_pos = left_space if left_space != -1 else right_space
        if split_pos != -1:
            return label[:split_pos] + '\n' + label[split_pos+1:]
        else:
            return label[:max_length-3] + '...'

    def check_low_cardinality_categoricals(self, cardinality_threshold=7, x=12, y_per_row=3, max_y=12):
        '''
        Identifies and visualizes low-cardinality non-numeric features.
        
        For each qualifying column:
            - Plots the value counts
            - Compares raw vs normalized unique counts (lowercased + stripped)
            - Prints warnings if normalization reduces uniqueness (data quality hint)

        Args:
            cardinality_threshold (int): Max unique values to consider a column low-cardinality.
            x (int): Total figure width.
            y_per_row (int): Vertical space per row.
            max_y (int): Max height of the figure.

        Returns:
            None
        '''
        print(f'Checking for low-cardinality categorical features with up to {cardinality_threshold} unique values...')
        data = self.data

        # get categorical columns
        cat_cols = data.select_dtypes(include=['object', 'category', 'string']).columns

        # drop those with only NaNs
        low_card_cols = [
            col for col in cat_cols
            if data[col].nunique(dropna=True) <= cardinality_threshold and not data[col].isna().all()
        ]

        if not low_card_cols:
            print('No low-cardinality categorical features found.')
            return

        n_feats = len(low_card_cols)
        n_cols = 1  # one plot per row
        n_rows = n_feats
        y = min(y_per_row * n_rows, max_y)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(x, y), constrained_layout=True)
        axes = np.array(axes).reshape(-1)

        # plotting 
        for ax, col in zip(axes, low_card_cols):
            
            vc = data[col].value_counts(dropna=False)
            shortened_labels = [self.shorten_labels(lbl) for lbl in vc.index.astype(str)]

            sns.barplot(
                x=vc.values,
                y=shortened_labels,
                ax=ax,
                palette='cividis',
                hue=shortened_labels,
                dodge=False,
                legend=False
            )
            ax.set_title(col.replace('_', ' '), fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel('')

            # check if unique values change after normalization
            normalized = data[col].dropna().astype(str).str.strip().str.lower()
            raw_unique = data[col].nunique(dropna=True)
            norm_unique = normalized.nunique()

            if norm_unique < raw_unique:
                print(f'⚠️ Warning: "{col}" has {raw_unique} unique values, but only {norm_unique} after normalization.')
                print('   → Possible duplicates due to casing or whitespace.\n')

        for ax in axes[len(low_card_cols):]:
            ax.set_visible(False)

        plt.show()


    def check_high_cardinality_categoricals(self, cardinality_threshold=7, cardinality_limit=100, x=12, y_per_row=3, max_y=12):
        '''
        Identifies and summarizes high-cardinality non-numeric features.
        
        For each qualifying column:
            - Prints number of unique values and sample unique values
            - Plots top N most frequent categories to give a sense of distribution

        Args:
            cardinality_threshold (int): Min unique values to consider a column high-cardinality.
            cardinality_limit (int): Max unique values to consider a column high-cardinality.
            x (int): Total figure width.
            y_per_row (int): Height per plot row.
            max_y (int): Max figure height.

        Returns:
            None
        '''
        print('Checking for high-cardinality categorical features with more than '
                f'{cardinality_threshold} and up to {cardinality_limit} unique values...')
        data = self.data

        cat_cols = data.select_dtypes(include=['object', 'category', 'string']).columns

        high_card_cols = [
            col for col in cat_cols
            if cardinality_threshold <= data[col].nunique(dropna=True) <= cardinality_limit
            and not data[col].isna().all()
        ]

        if not high_card_cols:
            print('No high-cardinality categorical features found.')
            return

        n_feats = len(high_card_cols)
        n_cols = 1 
        n_rows = n_feats
        y = min(y_per_row * n_rows, max_y)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(x, y), constrained_layout=True)
        axes = np.array(axes).reshape(-1)

        # plotting 
        for ax, col in zip(axes, high_card_cols):
            
            vc = data[col].value_counts(dropna=False)
            shortened_labels = [self.shorten_labels(lbl) for lbl in vc.index.astype(str)]

            sns.barplot(
                y=vc.values,
                x=shortened_labels,
                ax=ax,
                palette='cividis',
                hue=shortened_labels,
                dodge=False,
                legend=False
            )
            ax.set_title(col.replace('_', ' '), fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel('')
            labels = ax.get_xticklabels()
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)

        for ax in axes[len(high_card_cols):]:
            ax.set_visible(False)

        plt.show()

    def check_high_cardinality_normalization(self, cardinality_threshold=7):
        '''
        Checks high-cardinality non-numeric features to see if normalization reduces
        the number of unique values, hinting at possible duplicates due to casing or whitespace.

        Args:
            cardinality_threshold (int): Minimum unique values to consider a column high-cardinality.

        Returns:
            None: Prints warnings if found.
        '''
        print(f'Checking high-cardinality categorical features with more than {cardinality_threshold} unique values...\n')
        data = self.data

        # get categorical columns
        cat_cols = data.select_dtypes(include=['object', 'category', 'string']).columns

        # filter for high cardinality cols in given range and exclude all-NaN columns
        high_card_cols = [
            col for col in cat_cols
            if cardinality_threshold <= data[col].nunique(dropna=True)
            and not data[col].isna().all()
        ]

        if not high_card_cols:
            print('No high-cardinality categorical features found within the specified range.')
            return

        for col in high_card_cols:
            raw_unique = data[col].nunique(dropna=True)
            normalized = data[col].dropna().astype(str).str.strip().str.lower()
            norm_unique = normalized.nunique()

            if norm_unique < raw_unique:
                print(f'⚠️ Warning: "{col}" has {raw_unique} unique values, but only {norm_unique} after normalization.')
                print('   → Possible duplicates due to casing or whitespace.\n')





