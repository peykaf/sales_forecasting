import pathlib
from abc import ABC, abstractmethod
from pandas import DataFrame

from .evaluation import compute_metrics
from .utils import split_train_valid


class PredictiveModel(ABC):
    """
    Base class for predictive models.
    """

    @abstractmethod
    def fit(self, features: DataFrame, target: DataFrame):
        """Learn the model's hyperparameters based on the provided features and target."""

    def predict(self, raw_data: DataFrame) -> DataFrame:
        """Predict given a set of raw features using the fitted model."""
        return self._predict(features=raw_data)

    @abstractmethod
    def _featurize(self, raw_data: DataFrame) -> DataFrame:
        """
        Converts raw feature data into predictive features expected by the model.
        @param raw_data: Raw feature data
        @return: Predictive features DataFrame
        """

    @abstractmethod
    def _predict(self, features: DataFrame) -> DataFrame:
        """Predict given a set of precomputed features.
        @param features: features
        @return: Predicted  target
        """

    @abstractmethod
    def get_training_features(self) -> DataFrame:
        """Retrieve and pre-process data to obtain the training features."""

    @abstractmethod
    def get_training_target(self) -> DataFrame:
        """Retrieve and pre-process data to obtain the target."""

    def _log_metrics(self, features: DataFrame, target: DataFrame, name: str):
        """Log the essential components and metrics."""

        predictions = self._predict(features=features)

        join_columns = [
            'Segment',
            'Region',
            'Product ID',
            'Sub-Category',
            'Order Date',
            'Discount',
        ]

        predictions_vs_targets = (
            predictions
            .merge(target, on=join_columns, suffixes=('_pred', '_target'))
        )

        metrics = compute_metrics(predictions_vs_targets)
        print(f'Metrics for {name}: {metrics}')
        with open(f'outputs/{name}_metrics.txt', 'w') as f:
            for key, value in metrics.items():
                f.write(f'{key}: {value}\n')

        predictions_vs_targets.to_csv(f'outputs/{name}.csv', index=False)

        self.compute_feature_importance(features, target, name)


    @abstractmethod
    def compute_feature_importance(self, features: DataFrame, target: DataFrame, name: str):
        """Compute and log feature importance for the model."""

    @abstractmethod
    def save_model(self, path: pathlib.Path):
        """saves the model on disk."""

    def train(self):
        """Train the model using the training features and target."""

        path = pathlib.Path(__file__).parents[1]

        features = self.get_training_features()
        target = self.get_training_target()

        split_features, split_target = split_train_valid(
            features=features,
            target=target
        )

        self.fit(
            features=split_features.train,
            target=split_target.train
        )

        self._log_metrics(
            features=split_features.train,
            target=split_target.train,
            name='train',
        )

        self._log_metrics(
            features=split_features.valid,
            target=split_target.valid,
            name='valid',
        )

        self.save_model(path=path)

