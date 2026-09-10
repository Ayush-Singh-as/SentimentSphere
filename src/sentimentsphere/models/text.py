"""A reproducible, CPU-friendly text baseline with preprocessing inside its pipeline."""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from sentimentsphere.core.config import TrainConfig
from sentimentsphere.data.text import clean_text


def model_text(text: str) -> str:
    """A custom sklearn preprocessor must perform case normalization itself."""
    return clean_text(text).casefold()


def make_text_model(config: TrainConfig) -> Any:
    """Word and character features preserve negation and tolerate small typos."""
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    preprocessor=model_text,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=config.max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    preprocessor=model_text,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=config.max_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    C=config.regularization,
                    class_weight=config.class_weight,
                    max_iter=1000,
                    random_state=config.seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
