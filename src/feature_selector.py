from sklearn.linear_model import Ridge, LogisticRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import make_scorer, mean_squared_error, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
import seaborn as sns
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt

class FeatureSelector:
    def __init__(self, train_df, task=None, target=None):
        assert task in ['regression', 'classification'], "task must be either 'regression' or 'classification'"
        assert target is not None, "target must be specified"

        self.task = task
        self.target = target
        self.train_df = train_df.copy()
        self.wrapper_estimator = self._get_wrapper_estimator()
        self.X_train = self.train_df.drop(columns=[self.target])
        self.y_train = self.train_df[self.target]
        self.wrapper_selected_features_ = []
        self.df = pd.DataFrame({'Features': self.X_train.columns})

    def _get_wrapper_estimator(self):
        if self.task == 'regression':
            return Ridge()
        else:
            return DecisionTreeClassifier(random_state=42)

    def _get_embedded_estimator(self):
        if self.task == 'regression':
            return Lasso(alpha=0.01, random_state=42, max_iter=5000)
        else:
            return LogisticRegression(penalty='l1', solver='saga', C=1*0.1, random_state=42, max_iter=1000)

    def _get_scorer(self):
        if self.task == 'regression':
            return make_scorer(mean_squared_error, greater_is_better=False, squared=False)
        else:
            return make_scorer(f1_score, average='binary')

    def forward_selection(self, max_features=None, cv=3):
        X, y = self.X_train.copy(), self.y_train.copy()
        remaining_features = list(X.columns)
        selected_features = []
        best_score = float('-inf')
        scorer = self._get_scorer()

        pbar = tqdm(total=max_features or len(remaining_features), desc='Forward Selection')
        while remaining_features:
            scores = []
            for feature in remaining_features:
                candidate_features = selected_features + [feature]
                estimator = clone(self.wrapper_estimator)
                score = np.mean(cross_val_score(estimator, X[candidate_features], y, scoring=scorer, cv=cv, n_jobs=-1))
                scores.append((score, feature))

            scores.sort(reverse=True)
            best_candidate_score, best_candidate_feature = scores[0]

            if best_candidate_score > best_score:
                best_score = best_candidate_score
                selected_features.append(best_candidate_feature)
                remaining_features.remove(best_candidate_feature)
                pbar.update(1)
            else:
                break

            if max_features and len(selected_features) >= max_features:
                break

        pbar.close()
        self.wrapper_selected_features_ = selected_features
        self.df['Wrapper_Forward'] = self.df['Features'].apply(
            lambda x: 'Keep' if x in selected_features else 'Discard'
        )
        return self

    def backward_selection(self, min_features_to_retain=5, cv=3):
        X, y = self.X_train.copy(), self.y_train.copy()
        all_features = list(X.columns)
        scorer = self._get_scorer()

        pbar = tqdm(total=len(all_features) - min_features_to_retain, desc='Backward Elimination')
        while len(all_features) > min_features_to_retain:
            worst_score = float('-inf') if self.task == 'regression' else float('inf')
            worst_feature = None

            for feature in all_features:
                candidate_features = [f for f in all_features if f != feature]
                estimator = clone(self.wrapper_estimator)
                score = np.mean(cross_val_score(estimator, X[candidate_features], y, scoring=scorer, cv=cv, n_jobs=-1))

                is_worse = score > worst_score if self.task == 'regression' else score < worst_score

                if is_worse:
                    worst_score = score
                    worst_feature = feature

            if worst_feature is not None:
                all_features.remove(worst_feature)
                pbar.update(1)
            else:
                break

        pbar.close()
        self.wrapper_selected_features_ = all_features
        self.df['Wrapper_Backward'] = self.df['Features'].apply(
            lambda x: 'Keep' if x in all_features else 'Discard'
        )
        return self

    def rfe_selection(self, min_features_to_retain=5, step=3, cv=3):
        X, y = self.X_train.copy(), self.y_train.copy()
        features = list(X.columns)
        scorer = self._get_scorer()

        n_iterations = max(0, (len(features) - min_features_to_retain + step - 1) // step)
        pbar = tqdm(total=n_iterations, desc='RFE Selection')

        while len(features) > min_features_to_retain:
            scores = []
            for feature in features:
                candidate_features = [f for f in features if f != feature]
                estimator = clone(self.wrapper_estimator)
                score = np.mean(cross_val_score(estimator, X[candidate_features], y, scoring=scorer, cv=cv, n_jobs=-1))
                scores.append((score, feature))

            scores.sort(reverse=True)
            worst_features = [feat for _, feat in scores[-step:]]
            for wf in worst_features:
                features.remove(wf)

            pbar.update(1)

        pbar.close()
        self.wrapper_selected_features_ = features
        self.df['Wrapper_RFE'] = self.df['Features'].apply(
            lambda x: 'Keep' if x in features else 'Discard'
        )
        return self

    def embedded_selection(self, alpha=0.1, cv=3):
        X, y = self.X_train.copy(), self.y_train.copy()

        # ----- LASSO or Logistic Regression with L1 -----
        estimator = clone(self._get_embedded_estimator())
        if self.task == 'regression':
            estimator.set_params(alpha=alpha)
        else:
            estimator.set_params(C=1*alpha)

        estimator.fit(X, y)
        coef = np.abs(estimator.coef_) if self.task == 'regression' else np.abs(estimator.coef_.ravel())
        selected_lasso_features = X.columns[coef > 0].tolist()

        self.df['Embedded_L1'] = self.df['Features'].apply(
            lambda x: 'Keep' if x in selected_lasso_features else 'Discard'
        )

        # ----- Random Forest -----
        if self.task == 'regression':
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        else:
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        rf = clone(rf)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        selected_rf_features = importances[importances > 0.005].index.tolist()

        self.df['Embedded_RF'] = self.df['Features'].apply(
            lambda x: 'Keep' if x in selected_rf_features else 'Discard'
        )
        return self

    def _heat_correlation(self, data, title='Correlation Heatmap', x=10, y=8, fontsize=8, fmt='.2f', annot_kws=10, shrink_cbar=.7):
        cmap = sns.color_palette('vlag_r', as_cmap=True)
        mask = np.triu(np.ones_like(data, dtype=bool))

        plt.figure(figsize=(x, y))
        sns.heatmap(data, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, cbar_kws={'shrink': shrink_cbar},
                    annot=True, fmt=fmt, annot_kws={'size': annot_kws})

        plt.title(title, fontsize=18, fontweight='bold')
        plt.xticks(rotation=90, fontsize=fontsize)
        plt.yticks(rotation=45, fontsize=fontsize)
        plt.show()
        return None

    def _remove_multicollinearity(self, features, corr_matrix, target_corr, corr_feature_thresh=0.7):
            # Remove features highly correlated (≥ corr_feature_thresh) with stronger target correlated features
            ranked_feats = target_corr.loc[features].abs().sort_values(ascending=False).index.tolist()
            to_keep = []
            for feat in ranked_feats:
                if all(abs(corr_matrix.loc[feat, other]) < corr_feature_thresh for other in to_keep):
                    to_keep.append(feat)
            return to_keep

    def filter_selection(self, feature_groups, corr_target_thresh=0.1, corr_feature_thresh=0.7, show_heatmap=True):
        X, y = self.X_train.copy(), self.y_train.copy()

        if 'Filter_Corr' not in self.df.columns:
            self.df['Filter_Corr'] = np.nan

        # Keep track of all features that pass threshold and are kept across all groups
        all_kept_features = []

        for group_name, features in feature_groups.items():
            group_features = [f for f in features if f in X.columns]
            if not group_features:
                continue

            data = X[group_features].copy()
            data[self.target] = y
            corr_matrix = data.corr(method='pearson')

            target_corr = corr_matrix[self.target].drop(self.target)
            # Step 1: Filter by correlation with target ≥ threshold
            passed_feats = target_corr[abs(target_corr) >= corr_target_thresh].index.tolist()

            # Step 2: Remove multicollinearity within passed_feats
            kept_feats = self._remove_multicollinearity(features=passed_feats, corr_matrix=corr_matrix, target_corr=target_corr, corr_feature_thresh=corr_feature_thresh)

            all_kept_features.extend(kept_feats)

            # Update Filter_Corr for this group (Keep / Discard / NaN)
            new_filter = self.df['Features'].apply(
                lambda f: 'Keep' if f in kept_feats else ('Discard' if f in group_features else np.nan)
            )
            self.df['Filter_Corr'] = self.df['Filter_Corr'].combine_first(new_filter)

            # Plot heatmap for this group
            if show_heatmap:
                heatmap_feats = group_features + [self.target]
                self._heat_correlation(
                    corr_matrix.loc[heatmap_feats, heatmap_feats],
                    title=f'Correlation: Filter Selection - {group_name}'
                )

        # Final check: Remove multicollinearity across all kept features from all groups
        if all_kept_features:
            data_all = X[all_kept_features].copy()
            data_all[self.target] = y
            corr_matrix_all = data_all.corr(method='pearson')
            target_corr_all = corr_matrix_all[self.target].drop(self.target)

            final_kept = self._remove_multicollinearity(features=all_kept_features, corr_matrix=corr_matrix_all, target_corr=target_corr_all, corr_feature_thresh=corr_feature_thresh)

            # Update Filter_Corr for final combined set
            new_filter_all = self.df['Features'].apply(
                lambda f: 'Keep' if f in final_kept else ('Discard' if f in all_kept_features else np.nan)
            )
            self.df['Filter_Corr'] = self.df['Filter_Corr'].combine_first(new_filter_all)

            if show_heatmap and final_kept:
                heatmap_feats_final = final_kept + [self.target]
                self._heat_correlation(
                    corr_matrix_all.loc[heatmap_feats_final, heatmap_feats_final],
                    title='Correlation: Filter Selection - Final Combined'
                )

    def final_selection(self):
        X, y = self.X_train.copy(), self.y_train.copy()

        # 1st level of filtering: keep features that passed at least once
        fs_cols = [col for col in self.df.columns if col != 'Features']
        self.df['Keep_Count'] = self.df[fs_cols].eq('Keep').sum(axis=1)
        self.df['Filter_Final'] = self.df.apply(lambda x: 'Keep' if x.Keep_Count > 1 else 'Discard', axis=1)
        self.df = self.df.drop(columns=['Keep_Count'])

        # 2nd level of filtering: drop explanatory features that are highly correlated with others with respect to the target
        keep_cols = self.df.loc[self.df.Filter_Final == 'Keep', 'Features'].to_list()
        

        data_all = X[keep_cols].copy()
        data_all[self.target] = y
        corr_matrix_all = data_all.corr(method='pearson')
        target_corr_all = corr_matrix_all[self.target].drop(self.target)

        final_kept = self._remove_multicollinearity(features=keep_cols, corr_matrix=corr_matrix_all, target_corr=target_corr_all, corr_feature_thresh=0.8)

        # Update Filter_Corr for final combined set
        self.df['Filter_Final'] = self.df['Features'].apply(
            lambda x: 'Keep' if x in final_kept else 'Discard'
        )    

        
        heatmap_feats_final = final_kept + [self.target]
        self._heat_correlation(
            corr_matrix_all.loc[heatmap_feats_final, heatmap_feats_final],
            title='Correlation: Filter Selection - Final Combined',
            x=12, y=10, fontsize=7, fmt='.2f', annot_kws=8, shrink_cbar=.7
        )

        return self


