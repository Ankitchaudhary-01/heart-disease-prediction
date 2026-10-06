"""Preprocessing pipeline definition for Heart Disease Prediction."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def create_preprocessor() -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer for numerical standardization and
    categorical one-hot encoding.
    
    Adheres strictly to featurization ordering:
    - Numerical: median imputation followed by StandardScaler.
    - Categorical: most frequent imputation followed by OneHotEncoder.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def create_model_pipeline(classifier) -> Pipeline:
    """
    Assemble the full pipeline: Preprocessor -> Estimator.
    Prevents data leakage across CV folds and test sets.
    """
    preprocessor = create_preprocessor()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline
