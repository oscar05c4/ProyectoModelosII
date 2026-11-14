"""Utilidades del proyecto PIU (Child Mind Institute).

BlockMissingIndicator: indicador de bloque clínico ausente.
OrdinalThresholdRegressor: regresor + 3 umbrales que maximizan QWK, con ponderación
opcional por clase (weight_power)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import cohen_kappa_score


class BlockMissingIndicator(BaseEstimator, TransformerMixin):
    """1 si TODAS las columnas numéricas de un instrumento (prefijo antes de '-') son NaN.
    Se descartan los bloques cuyo indicador es constante en entrenamiento."""

    def _blocks(self, cols):
        b = {}
        for c in cols:
            b.setdefault(c.split("-")[0], []).append(c)
        return b

    def _ind(self, X):
        return pd.DataFrame({b: X[cs].isna().all(axis=1).astype(float)
                             for b, cs in self.blocks_.items()}, index=X.index)

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.blocks_ = self._blocks(X.columns)
        ind = self._ind(X)
        self.keep_ = [b for b in ind.columns if 0 < ind[b].mean() < 1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return self._ind(X)[self.keep_].values

    def get_feature_names_out(self, input_features=None):
        return np.array([f"miss_block_{b}" for b in self.keep_])


class OrdinalThresholdRegressor(BaseEstimator):
    """Regresor + 3 umbrales que maximizan QWK sobre predicciones out-of-fold del PROPIO
    conjunto de entrenamiento (nunca ve validación ni test).

    weight_power: 0 = sin pesos; 1 = pesos balanceados por clase; 0.5 = intermedio.
    Los pesos se calculan con las clases del conjunto de entrenamiento que recibe fit().
    Requiere que el regresor sea el paso llamado 'reg' del Pipeline."""

    weight_power = 0.0   # valor por defecto (también para modelos .joblib antiguos)

    def __init__(self, estimator, init_thr=(0.5, 1.5, 2.5), inner_cv=5,
                 random_state=42, n_rounds=3, grid_size=60, weight_power=0.0):
        self.estimator = estimator
        self.init_thr = init_thr
        self.inner_cv = inner_cv
        self.random_state = random_state
        self.n_rounds = n_rounds
        self.grid_size = grid_size
        self.weight_power = weight_power

    def _qwk(self, y, oof, thr):
        return cohen_kappa_score(y, np.digitize(oof, thr),
                                 weights="quadratic", labels=[0, 1, 2, 3])

    def _weights(self, y):
        if self.weight_power == 0:
            return None
        y = np.asarray(y).astype(int)
        cnt = np.bincount(y, minlength=4).astype(float)
        w_c = (len(y) / (4 * np.maximum(cnt, 1))) ** self.weight_power
        return w_c[y]

    def fit(self, X, y):
        y = np.asarray(y)
        w = self._weights(y)
        fp = {} if w is None else {"reg__sample_weight": w}
        cv = StratifiedKFold(self.inner_cv, shuffle=True, random_state=self.random_state)
        oof = cross_val_predict(clone(self.estimator), X, y, cv=cv, params=fp)
        thr = np.array(self.init_thr, float)
        lo, hi = np.quantile(oof, 0.01), np.quantile(oof, 0.99)
        cand = np.linspace(lo, hi, self.grid_size)
        for _ in range(self.n_rounds):
            for j in range(3):
                best, best_t = -np.inf, thr[j]
                for t in cand:
                    tt = thr.copy()
                    tt[j] = t
                    if np.all(np.diff(tt) > 0):
                        s = self._qwk(y, oof, tt)
                        if s > best:
                            best, best_t = s, t
                thr[j] = best_t
        self.thr_ = thr
        self.estimator_ = clone(self.estimator).fit(X, y, **fp)
        return self

    def predict(self, X):
        return np.digitize(self.estimator_.predict(X), self.thr_)