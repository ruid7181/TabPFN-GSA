from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone

from tabpfn_gsa import GSAModel, tune_gsa


def make_dataset(n_samples: int = 80) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n_samples),
            "x2": rng.normal(size=n_samples),
            "coord_x": rng.uniform(0.0, 1.0, size=n_samples),
            "coord_y": rng.uniform(0.0, 1.0, size=n_samples),
        }
    )
    y = pd.Series(2.0 * X["x1"] - X["x2"], name="target")
    return X.iloc[:50].reset_index(drop=True), y.iloc[:50].reset_index(drop=True), X.iloc[50:].reset_index(drop=True)


class MeanFunctionModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.value)


def fit_mean_model(X, y, offset: float = 0.0) -> MeanFunctionModel:
    return MeanFunctionModel(float(np.mean(y)) + offset)


def predict_mean_model(model: MeanFunctionModel, X, scale: float = 1.0) -> np.ndarray:
    return model.predict(X) * scale


def test_unified_model_accepts_custom_fit_and_predict_functions() -> None:
    X_train, y_train, X_test = make_dataset()
    model = GSAModel(
        spa_cols=["coord_x", "coord_y"],
        x_cols=["x1", "x2"],
        K=4,
        s=0.1,
        random_state=0,
        model_kwargs={
            "fit_fn": fit_mean_model,
            "predict_fn": predict_mean_model,
            "fit_kwargs": {"offset": 1.0},
            "predict_kwargs": {"scale": 0.5},
        },
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert predictions.shape == (len(X_test),)
    assert np.isfinite(predictions).all()
    assert model.get_runtime_info().estimator_class == "fit_fn/predict_fn"
    assert model.get_runtime_info().backend_name == "custom"


def test_custom_function_model_requires_a_fit_function() -> None:
    with pytest.raises(ValueError, match="callable fit_fn"):
        GSAModel(
            spa_cols=["coord_x", "coord_y"],
            x_cols=["x1", "x2"],
            model_kwargs={"predict_fn": predict_mean_model},
        )


def test_custom_model_rejects_estimator_argument() -> None:
    with pytest.raises(ValueError, match="Do not pass estimator"):
        GSAModel(
            spa_cols=["coord_x", "coord_y"],
            x_cols=["x1", "x2"],
            model_kwargs={
                "estimator": object(),
                "fit_fn": fit_mean_model,
                "predict_fn": predict_mean_model,
            },
        )


def test_custom_model_rejects_unused_kwargs() -> None:
    with pytest.raises(ValueError, match="Unexpected custom model_kwargs keys"):
        GSAModel(
            spa_cols=["coord_x", "coord_y"],
            x_cols=["x1", "x2"],
            model_kwargs={
                "fit_fn": fit_mean_model,
                "predict_fn": predict_mean_model,
                "unused": True,
            },
        )


def test_custom_function_fit_must_return_state() -> None:
    def bad_fit(X, y):
        return None

    def predict_fn(model, X):
        return np.zeros(len(X))

    model = GSAModel(
        spa_cols=["coord_x", "coord_y"],
        x_cols=["x1", "x2"],
        model_kwargs={"fit_fn": bad_fit, "predict_fn": predict_fn},
    )
    X_train, y_train, _ = make_dataset()

    model.fit(X_train, y_train)
    with pytest.raises(ValueError, match="fit_fn must return"):
        model.predict(X_train)


def test_sampling_is_stable_across_batches_and_refits() -> None:
    X, y, queries = make_dataset(n_samples=400)
    model = GSAModel(
        spa_cols=["coord_x", "coord_y"], K=64, s=0.2, random_state=7,
        model_kwargs={"fit_fn": fit_mean_model, "predict_fn": predict_mean_model},
    ).fit(X, y)
    full = model.predict_with_uncertainty(queries)
    parts = [
        model.predict_with_uncertainty(queries.iloc[indices])
        for indices in np.array_split(np.arange(len(queries)), 7)
    ]
    np.testing.assert_array_equal(full.mean, np.concatenate([part.mean for part in parts]))
    np.testing.assert_array_equal(full.std, np.concatenate([part.std for part in parts]))
    np.testing.assert_array_equal(model.predict(queries.iloc[::-1])[::-1], full.mean)
    for index in [0, 100, 200]:
        single = model.predict_with_uncertainty(queries.iloc[[index]])
        np.testing.assert_array_equal(single.mean, full.mean[[index]])
        np.testing.assert_array_equal(single.std, full.std[[index]])
    model.fit(X, y)
    np.testing.assert_array_equal(model.predict(queries), full.mean)


def test_fit_does_not_train_a_global_model() -> None:
    X, y, queries = make_dataset(n_samples=400)
    fitted_sizes = []

    def fit_small_context(X, y):
        fitted_sizes.append(len(X))
        if len(X) >= 30:
            raise ValueError("Context is too large for this model.")
        return fit_mean_model(X, y)

    model = GSAModel(
        spa_cols=["coord_x", "coord_y"], K=64, s=0.1,
        model_kwargs={"fit_fn": fit_small_context, "predict_fn": predict_mean_model},
    ).fit(X, y)
    assert fitted_sizes == []
    assert np.isfinite(model.predict(queries)).all()
    assert fitted_sizes and max(fitted_sizes) < 30


def test_empty_neighborhood_explains_how_to_fix_it() -> None:
    X = pd.DataFrame({"coord_x": [0.0, 0.1], "coord_y": [0.0, 0.1]})
    queries = pd.DataFrame({"coord_x": [0.9], "coord_y": [0.9]})
    model = GSAModel(
        spa_cols=["coord_x", "coord_y"], K=64, s=0,
        spa_bounds={"coord_x": (0, 1), "coord_y": (0, 1)},
        model_kwargs={"fit_fn": fit_mean_model, "predict_fn": predict_mean_model},
    ).fit(X, pd.Series([1.0, 3.0]))
    with pytest.raises(ValueError, match="Increase s.*reduce K"):
        model.predict(queries)
    model.set_params(s=0.1).fit(X, pd.Series([1.0, 3.0]))
    np.testing.assert_array_equal(model.predict(queries), [2.0])


def test_refit_uses_updated_model_kwargs() -> None:
    X, y, queries = make_dataset()
    model = GSAModel(
        spa_cols=["coord_x", "coord_y"], K=4,
        model_kwargs={"fit_fn": fit_mean_model, "predict_fn": predict_mean_model},
    ).fit(X, y)
    original = model.predict(queries)
    model.set_params(model_kwargs={
        "fit_fn": fit_mean_model, "predict_fn": predict_mean_model,
        "fit_kwargs": {"offset": 100.0},
    }).fit(X, y)
    np.testing.assert_allclose(model.predict(queries), original + 100.0)


@pytest.mark.parametrize("output, message", [
    ([1.0], "one prediction per input row"),
    ([np.nan, np.nan], "NaN or infinite"),
    ([np.inf, np.inf], "NaN or infinite"),
    ([[1.0, 2.0]], "one prediction per input row"),
    (1.0, "one prediction per input row"),
])
def test_invalid_model_predictions_are_rejected(output, message) -> None:
    X, y, queries = make_dataset()
    model = GSAModel(
        spa_cols=["coord_x", "coord_y"], K=1,
        model_kwargs={
            "fit_fn": fit_mean_model,
            "predict_fn": lambda state, X: output,
        },
    ).fit(X, y)
    with pytest.raises(ValueError, match=message):
        model.predict(queries.iloc[:2])


class FreshRegressor(RegressorMixin, BaseEstimator):
    """Fail if cloning copies model state or reuses an already fitted instance."""

    def __init__(self, offset=0.0):
        self.offset = offset

    def __deepcopy__(self, memo):
        raise AssertionError("Do not deepcopy a loaded model.")

    def fit(self, X, y, bias=0.0):
        assert not hasattr(self, "mean_"), "Each grid needs a fresh model."
        self.mean_ = float(np.mean(y)) + self.offset + bias
        return self

    def predict(self, X, scale=1.0):
        return np.full((len(X), 1), self.mean_ * scale)


@pytest.mark.parametrize("explicit_predict", [False, True])
def test_bound_methods_are_isolated_and_cloneable(explicit_predict) -> None:
    X, y, queries = make_dataset(n_samples=200)
    source = FreshRegressor(offset=3).fit(X, y)
    original_mean = source.mean_
    kwargs = {
        "fit_fn": source.fit, "fit_kwargs": {"bias": 2.0},
        "predict_kwargs": {"scale": 0.5},
    }
    if explicit_predict:
        kwargs["predict_fn"] = source.predict
    model = GSAModel(spa_cols=["coord_x", "coord_y"], K=16, model_kwargs=kwargs)
    reference = GSAModel(
        spa_cols=["coord_x", "coord_y"], K=16,
        model_kwargs={
            "fit_fn": fit_mean_model, "fit_kwargs": {"offset": 5.0},
            "predict_fn": predict_mean_model, "predict_kwargs": {"scale": 0.5},
        },
    ).fit(X, y)
    model = clone(model).fit(X, y)
    result = model.predict_with_uncertainty(queries)
    expected = reference.predict_with_uncertainty(queries)
    np.testing.assert_allclose(result.mean, expected.mean)
    np.testing.assert_allclose(result.std, expected.std)
    np.testing.assert_array_equal(model.predict(queries), result.mean)
    assert source.mean_ == original_mean


def test_bound_methods_work_with_optuna() -> None:
    X, y, queries = make_dataset()
    source = FreshRegressor()
    model = GSAModel(
        spa_cols=["coord_x", "coord_y"],
        model_kwargs={"fit_fn": source.fit, "predict_fn": source.predict},
    )
    result = tune_gsa(
        model, X, y, K_values=[4, 16], s_values=[0.1], cv=2, n_trials=2,
    )
    assert np.isfinite(result.best_estimator.predict(queries)).all()
    assert not hasattr(source, "mean_")


def test_factory_can_omit_predict_fn() -> None:
    X, y, queries = make_dataset()
    model = GSAModel(
        spa_cols=["coord_x", "coord_y"], K=1,
        model_kwargs={"fit_fn": fit_mean_model},
    ).fit(X, y)
    np.testing.assert_allclose(model.predict(queries), float(y.mean()))


def test_context_query_functions_receive_local_data() -> None:
    X, y, queries = make_dataset()
    contexts = []

    def fit_context(X_train, y_train):
        state = (X_train.to_numpy(), y_train.to_numpy())
        contexts.append(state)
        return state

    def predict_context(state, X_test, scale=1.0):
        X_train, y_train = state
        assert X_train.shape[1] == X_test.shape[1]
        return np.full(len(X_test), y_train.mean() * scale)

    model = GSAModel(
        spa_cols=["coord_x", "coord_y"], K=64,
        model_kwargs={
            "fit_fn": fit_context, "predict_fn": predict_context,
            "predict_kwargs": {"scale": 2.0},
        },
    ).fit(X, y)
    predictions = model.predict(queries)
    assert np.isfinite(predictions).all()
    assert max(len(state[1]) for state in contexts) < len(X)
    assert len({id(state) for state in contexts}) == len(contexts)


def test_context_without_predict_fn_has_clear_error() -> None:
    X, y, queries = make_dataset()
    model = GSAModel(
        spa_cols=["coord_x", "coord_y"],
        model_kwargs={"fit_fn": lambda X, y: {"y": y}},
    ).fit(X, y)
    with pytest.raises(ValueError, match="Provide predict_fn"):
        model.predict(queries)


def test_mismatched_bound_methods_are_rejected() -> None:
    with pytest.raises(ValueError, match="same model instance"):
        GSAModel(
            spa_cols=["coord_x", "coord_y"],
            model_kwargs={"fit_fn": FreshRegressor().fit, "predict_fn": FreshRegressor().predict},
        )


def test_uncloneable_bound_model_explains_function_alternative() -> None:
    class PlainModel:
        def fit(self, X, y):
            raise AssertionError("Must not fit the shared model.")

    X, y, queries = make_dataset()
    model = GSAModel(
        spa_cols=["coord_x", "coord_y"], model_kwargs={"fit_fn": PlainModel().fit},
    ).fit(X, y)
    with pytest.raises(TypeError, match="fresh model or context state"):
        model.predict(queries)


@pytest.mark.parametrize("kwargs, message", [
    ({"fit_fn": 123}, "callable fit_fn"),
    ({"fit_fn": fit_mean_model, "predict_fn": 123}, "predict_fn must be callable"),
    ({"fit_fn": fit_mean_model, "fit_kwargs": []}, "fit_kwargs must be a dictionary"),
    ({"fit_fn": fit_mean_model, "predict_kwargs": []}, "predict_kwargs must be a dictionary"),
])
def test_invalid_custom_configuration_is_rejected_early(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        GSAModel(spa_cols=["coord_x", "coord_y"], model_kwargs=kwargs)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_tensor_predictions_are_converted_to_numpy(dtype) -> None:
    torch = pytest.importorskip("torch")
    X, y, queries = make_dataset()

    def predict_tensor(state, X):
        return torch.full(
            (len(X), 1), 2.0, dtype=getattr(torch, dtype), requires_grad=True,
        )

    model = GSAModel(
        spa_cols=["coord_x", "coord_y"],
        model_kwargs={"fit_fn": fit_mean_model, "predict_fn": predict_tensor},
    ).fit(X, y)
    np.testing.assert_array_equal(model.predict(queries), np.full(len(queries), 2.0))
