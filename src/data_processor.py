import pandas as pd

class DataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().sort_values(['camis', 'inspection_date'])
        self.features = None

    def compute_score_features(self):
        """
        Compute score-related statistics per restaurant, including latest, average, worst, std, and count.

        Returns:
            DataProcessor: Self with updated `features` DataFrame containing score metrics.
        """
        grouped = self.df.groupby('camis')
        self.features = pd.DataFrame({
            'score_latest': grouped['score'].last(),
            'score_avg': grouped['score'].mean(),
            'score_worst': grouped['score'].max(),
            'score_std': grouped['score'].std(),
            'num_inspections': grouped['score'].count()
        })
        return self

    def compute_violation_features(self):
        """
        Computes total and critical violation counts per restaurant.

        Returns:
            DataProcessor: Self with `features` DataFrame updated with violation counts.
        """
        df = self.df.copy()
        df['is_critical'] = df['critical_flag'].fillna('').str.strip() == 'Critical'
        grouped = df.groupby('camis')
        self.features['critical_violations'] = grouped['is_critical'].sum()
        self.features['total_violations'] = grouped['critical_flag'].count()
        return self

    def compute_grade_features(self):
        """
        Computes grade-related metrics including latest and worst grades per restaurant.

        Returns:
            DataProcessor: Self with grade features added to `features` DataFrame.
        """
        grouped = self.df.groupby('camis')
        self.features['latest_grade'] = grouped['grade'].last()
        self.features['worst_grade'] = grouped['grade'].apply(
            lambda grades: (
                sorted(
                    [g for g in grades if pd.notna(g)],
                    key=lambda g: {'A': 0, 'B': 1, 'C': 2}.get(g, 3)
                )[-1] if any(pd.notna(g) for g in grades) else None
            )
        )
        return self

    def compute_time_features(self):
        """
        Computes inspection date metrics such as first/last inspections and intervals.

        Returns:
            DataProcessor: Self with time-related fields added to `features`.
        """
        grouped = self.df.groupby('camis')
        self.features['latest_inspection'] = grouped['inspection_date'].max()
        self.features['first_inspection'] = grouped['inspection_date'].min()
        self.features['days_since_last_inspection'] = (
            pd.Timestamp.now() - self.features['latest_inspection']
        ).dt.days
        self.features['inspection_period_days'] = (
            self.features['latest_inspection'] - self.features['first_inspection']
        ).dt.days
        return self

    def add_static_info(self):
        """
        Adds static metadata (e.g., restaurant name, borough, location) based on first occurrence.

        Returns:
            DataProcessor: Self with static info joined to `features`.
        """
        static = self.df.groupby('camis')[[
            'dba', 'boro', 'latitude', 'longitude', 'cuisine_description', 'zipcode', 'community_board'
        ]].first().rename(columns={
            'dba': 'restaurant_name',
            'boro': 'borough',
            'latitude': 'latitude',
            'longitude': 'longitude',
            'cuisine_description': 'cuisine',
            'zipcode': 'zipcode',
            'community_board': 'community_board'
        })
        self.features = self.features.join(static)
        return self

    def compute_rates(self):
        """
        Computes normalized rates such as inspections per year and violation ratios.

        Returns:
            DataProcessor: Self with rate-based fields added to `features`.
        """
        self.features['inspections_per_year'] = (
            self.features['num_inspections'] * 365 / self.features['inspection_period_days']
        ).fillna(0)

        self.features['violation_rate'] = (
            self.features['total_violations'] / self.features['num_inspections']
        ).fillna(0)

        self.features['critical_violation_rate'] = (
            self.features['critical_violations'] / self.features['num_inspections']
        ).fillna(0)

        return self

    def build(self):
        """
        Executes the full feature engineering pipeline step-by-step.

        Returns:
            pd.DataFrame: Final processed DataFrame with engineered restaurant features.
        """
        return (
            self.compute_score_features()
                .compute_violation_features()
                .compute_grade_features()
                .compute_time_features()
                .add_static_info()
                .compute_rates()
                .df
        )