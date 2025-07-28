import pandas as pd
from catboost import Pool
from matplotlib import pyplot as plt
from pandas import DataFrame
from predictive.constants import PRED_COL, TARGET_COL
import shap


def compute_metrics(df: DataFrame) -> dict[str, float]:

    df['error'] = df[PRED_COL] - df[TARGET_COL]
    df['abs_error'] = df['error'].abs()
    df['pct_error'] = df['error'] / df[TARGET_COL]
    df['abs_pct_error'] = df['pct_error'].abs()
    df['sq_error'] = df['error'] ** 2

    return {
        'mae': df['abs_error'].mean(),
        'mse': df['sq_error'].mean(),
        'bias': df['error'].mean(),
        'mape': df['abs_pct_error'].mean(),
        'median_ape': df['abs_pct_error'].median(),
        'mpe': df['pct_error'].mean(),
        'rmse': df['sq_error'].mean() ** 0.5
    }

def compute_feature_importance(
    model,
    features: DataFrame,
    target: DataFrame
) -> DataFrame:

    shap_values = model.get_feature_importance(pool, type='ShapValues')
    shap_df = pd.DataFrame(shap_values[:, :-1], columns=features.columns)


    explainer = shap.Explainer(model)
    shap.plots.beeswarm(shap_df, feature_names=features.columns)
    plt.savefig('shap_beeswarm.png', bbox_inches='tight')

    # importances = model.get_feature_importance(pool, type='FeatureImportance')

    # # View as DataFrame
    # feature_importance_df = pd.DataFrame({
    #     'Feature': features.columns,
    #     'Importance': importances
    # }).sort_values(by='Importance', ascending=False)
    #
    # perm_importance = model.get_feature_importance(pool, type='LossFunctionChange')
    #
    # perm_importance_df = pd.DataFrame({
    #     'Feature': features.columns,
    #     'Importance': perm_importance
    # }).sort_values(by='Importance', ascending=False)