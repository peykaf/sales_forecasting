import pathlib
import numpy as np
import shap
from catboost import CatBoostRegressor, Pool
import pandas as pd
from matplotlib import pyplot as plt

from pandas import DataFrame

from predictive.base import PredictiveModel
from predictive.constants import TARGET_COL, COLUMNS_TO_KEEP, PRED_COL


class CatboostRegressorModel(PredictiveModel):
    REF_YEAR = 2013
    SEED = 1234

    def __init__(self):
        self._regressor_model = CatBoostRegressor

        self._num_features = [
            'Discount',
            'Original Price',
            'Year',
            'Month_sin',
            'Month_cos',
            'DayOfWeek_sin',
            'DayOfWeek_cos',
            'WeekOfYear_sin',
            'WeekOfYear_cos'
        ]

        self._cat_features = [
            'Segment',
            'Region',
            'Product ID',
            'Sub-Category',
        ]

    @property
    def regressor_model(self) -> type[CatBoostRegressor]:
        return self._regressor_model

    def get_training_features(self) -> DataFrame:
        features = self._featurize(
                raw_data=self._get_data(),
            )
        features.drop(columns=[TARGET_COL], inplace=True)

        return features

    def get_training_target(self) -> DataFrame:
        data = self._get_data()[COLUMNS_TO_KEEP]
        data['Order Date'] = pd.to_datetime(data['Order Date'])
        return data

    def _get_data(self) -> DataFrame:
        base_path = pathlib.Path().resolve()
        data_path = base_path / 'data/stores_sales_forecasting.csv'

        try:
            return pd.read_csv(data_path)
        except Exception as e:
            print(f"Error reading the CSV file: {e}")
            return pd.read_csv(data_path, encoding='ISO-8859-1')


    def _featurize(self, raw_data: DataFrame) -> DataFrame:
        raw_data = self._remove_unnecessary_columns(raw_data)
        raw_data = self._generate_original_price(raw_data)

        features = self._cyclic_week_encoding(data=raw_data)
        features = self._linearize_year(data=features)

        return features

    @classmethod
    def _remove_unnecessary_columns(cls, raw_data: DataFrame) -> DataFrame:
        return raw_data[COLUMNS_TO_KEEP]

    @classmethod
    def _generate_original_price(cls, raw_data) -> DataFrame:
        raw_data['Unit Sales Price'] = raw_data['Sales'] / raw_data['Quantity']
        raw_data['Original Price'] = np.log(raw_data['Unit Sales Price'] / (1 - raw_data['Discount']))
        raw_data.drop(columns=['Unit Sales Price', 'Sales'], inplace=True)

        return raw_data

    @classmethod
    def _cyclic_week_encoding(cls, data: DataFrame) -> DataFrame:
        data['Order Date'] = pd.to_datetime(data['Order Date'])

        data['Year'] = data['Order Date'].dt.year
        data['_Month'] = data['Order Date'].dt.month
        data['_WeekOfYear'] = data['Order Date'].dt.isocalendar().week
        data['_DayOfWeek'] = data['Order Date'].dt.dayofweek

        data['Month_sin'] = np.sin(2 * np.pi * data['_Month'] / 12)
        data['Month_cos'] = np.cos(2 * np.pi * data['_Month'] / 12)

        data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['_DayOfWeek'] / 7)
        data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['_DayOfWeek'] / 7)
        data['WeekOfYear_sin'] = np.sin(2 * np.pi * data['_WeekOfYear'] / 53)
        data['WeekOfYear_cos'] = np.cos(2 * np.pi * data['_WeekOfYear'] / 53)

        data.drop(columns=['_Month', '_WeekOfYear', '_DayOfWeek'], inplace=True)

        return data

    @classmethod
    def _linearize_year(cls, data: DataFrame) -> DataFrame:
        data['Year'] = data['Year'] - cls.REF_YEAR
        return data

    def _predict(self, features: DataFrame) -> DataFrame:
        predictions_data = features.copy()

        predictions = self._regressor_model.predict(
            features[self._num_features + self._cat_features]
        )

        predictions_data[PRED_COL] = predictions

        return predictions_data

    def fit(self, features: DataFrame, target: DataFrame):
        """Learn the model's hyperparameters."""

        self._regressor_model = CatBoostRegressor(
            l2_leaf_reg=15,
            verbose=200,
            loss_function='RMSE',
            random_seed=self.SEED,
            iterations=5000,
            learning_rate=0.01,
            depth=4,
            cat_features=self._cat_features,
        )

        self._regressor_model.fit(
            X=features[self._num_features + self._cat_features],
            y=target[TARGET_COL],
        )

        return self

    def save_model(self, path: pathlib.Path):
        """Save the model on disk."""
        self._regressor_model.save_model('outputs/model.cmb')
        print(f"Model saved to {path}")

    def compute_feature_importance(self, features: DataFrame, target: DataFrame, name :str):
        """Compute feature importance using the trained model.
        This includes SHAP values, feature importance, and permutation importance.
        """

        pool = Pool(
            features[self._num_features + self._cat_features],
            target[TARGET_COL],
            cat_features=self._cat_features,
        )

        # shap values
        shap_values = self._regressor_model.get_feature_importance(pool, type='ShapValues')
        shap_explanation = shap.Explanation(
            values=shap_values[:, :-1],
            feature_names=features[self._num_features + self._cat_features].columns,
        )

        plt.figure()
        shap.plots.beeswarm(shap_explanation, show=False)
        plt.savefig(f'outputs/shap_beeswarm_{name}.png', bbox_inches='tight')
        plt.close()

        # feature importance
        importances = self._regressor_model.get_feature_importance(pool, type='FeatureImportance')
        feature_importance_df = pd.DataFrame({
            'Feature': features[self._num_features + self._cat_features].columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        feature_importance_df.to_csv(f'outputs/feature_importance_{name}.csv', index=False)


        perm_importance = self._regressor_model.get_feature_importance(pool, type='LossFunctionChange')
        perm_importance_df = pd.DataFrame({
            'Feature': features[self._num_features + self._cat_features].columns,
            'Importance': perm_importance
        }).sort_values(by='Importance', ascending=False)

        perm_importance_df.to_csv(f'outputs/permutation_importance_{name}.csv', index=False)
