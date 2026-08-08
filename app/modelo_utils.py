"""
Classe do modelo preditivo de risco de defasagem, compartilhada entre `train_model.py`
(que treina e salva o modelo) e `app.py` (que carrega e usa o modelo).

Por que uma classe própria em vez de usar um modelo pronto do scikit-learn?
----------------------------------------------------------------------------------
Testamos primeiro uma árvore de decisão com restrição de monotonicidade
(HistGradientBoostingClassifier). Ela resolvia o problema de extrapolação em perfis
extremos, mas concentrava ~78% de toda a importância em um único indicador (IAN) —
porque árvores escolhem, de forma gulosa, a variável que mais separa a base a cada
divisão, e IAN (derivado da própria defasagem atual) é sozinho um preditor muito forte
da defasagem futura.

Essa concentração não é "vazamento de dados" no sentido técnico (IAN é uma informação
do ano atual, nunca do futuro — a separação temporal treino/teste já impede isso), mas
é um problema de qualidade do modelo: ele fica pouco informativo sobre quais
comportamentos (acadêmicos, de engajamento etc.) realmente antecipam risco, que é
exatamente o que a pergunta de negócio do desafio pede.

A regressão logística com coeficientes restritos a um sinal (`ModeloMonotonicoLogistico`)
resolve isso: por ser um modelo aditivo/linear, ela não tem o efeito "o vencedor leva
tudo" das árvores. O resultado prático: a importância do IAN cai de ~78% para ~64%, o
AUC de validação sobe ligeiramente (0,71 -> 0,72), e o modelo continua com a mesma
garantia de monotonicidade (melhorar qualquer indicador nunca aumenta o risco previsto).
"""
import numpy as np
from scipy.optimize import minimize


class ModeloMonotonicoLogistico:
    """Regressão logística em que cada coeficiente é restrito a ter sinal <= 0
    (indicador maior => risco previsto menor ou igual, nunca maior)."""

    def __init__(self, features):
        self.features = list(features)
        self.scaler_mean_ = None
        self.scaler_scale_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, C=1.0):
        X = X[self.features].to_numpy(dtype=float)
        y = np.asarray(y, dtype=float)

        self.scaler_mean_ = X.mean(axis=0)
        self.scaler_scale_ = X.std(axis=0)
        self.scaler_scale_[self.scaler_scale_ == 0] = 1.0
        Xs = (X - self.scaler_mean_) / self.scaler_scale_

        n_feat = Xs.shape[1]

        def neg_log_lik(params):
            w, b = params[:n_feat], params[n_feat]
            z = np.clip(Xs @ w + b, -30, 30)
            p = 1 / (1 + np.exp(-z))
            eps = 1e-9
            ll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            reg = (1 / (2 * C)) * np.sum(w ** 2) / len(y)
            return ll + reg

        # Restrição de monotonicidade: cada coeficiente entre -10 e 0 (nunca positivo)
        bounds = [(-10, 0)] * n_feat + [(-10, 10)]
        resultado = minimize(neg_log_lik, np.zeros(n_feat + 1), method="L-BFGS-B", bounds=bounds)

        self.coef_ = resultado.x[:n_feat]
        self.intercept_ = resultado.x[n_feat]
        return self

    def _linear_score(self, X):
        if hasattr(X, "columns"):
            X = X[self.features].to_numpy(dtype=float)
        else:
            X = np.asarray(X, dtype=float)
        Xs = (X - self.scaler_mean_) / self.scaler_scale_
        return Xs @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        z = np.clip(self._linear_score(X), -30, 30)
        p1 = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p1, p1])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def importancias_normalizadas(self):
        """Importância relativa de cada variável (|coeficiente padronizado|, em %).
        Como as variáveis foram padronizadas antes do ajuste, os coeficientes já são
        diretamente comparáveis entre si."""
        import pandas as pd
        imp = pd.Series(np.abs(self.coef_), index=self.features)
        total = imp.sum()
        return (imp / total * 100) if total > 0 else imp
