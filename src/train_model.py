"""
Treina o modelo de risco de defasagem e salva o artefato em `models/`.

Este script foi movido para `src/` (junto com `modelo_utils.py` e `limpeza.py`) porque
é código reutilizável do projeto, não parte do app Streamlit em si. Rode sempre que a
base `data/raw/PEDE_consolidado_longo.csv` for atualizada (ex.: ciclo PEDE 2025, ou
depois de novos cadastros pelo app):

    python -m src.train_model     # a partir da raiz do repositório (não rode como
                                    # "python src/train_model.py" — o -m garante que o
                                    # import "src.modelo_utils" fique com o mesmo nome
                                    # que o app usa para reconstruir o modelo salvo)

Gera: models/modelo_risco_defasagem.pkl

--------------------------------------------------------------------------------------
Histórico de decisões sobre o modelo (para quem for revisar depois):

1ª versão: RandomForest / Regressão Logística "soltos", sem restrição nenhuma. Problema:
um aluno hipotético com todos os indicadores no valor máximo aparecia com risco
moderado/alto, por extrapolação ruim em perfis raros na base.

2ª versão: HistGradientBoostingClassifier com restrição de monotonicidade. Resolveu o
problema acima, mas concentrava ~78% de toda a importância em um único indicador (IAN),
por causa de como árvores de decisão escolhem variáveis de forma gulosa.

3ª versão: Regressão logística com coeficientes restritos a um único sinal, mesmas 5
variáveis (IAN, IDA, IEG, IAA, IPS). Reduziu a concentração em IAN para ~64% e melhorou
o AUC (0,71 -> 0,72), mas ainda com IAN dominante e AUC abaixo do que o desafio exige.

4ª versão (atual): mesma regressão logística restrita, mas com **duas variáveis novas**:
`Idade` e `Anos_na_PM` (tempo, em anos, que o aluno já está na Passos Mágicos — calculado
como `Ano - Ano de ingresso`). As duas são informações demográficas conhecidas no ano T,
não derivadas de nenhum indicador de defasagem — não introduzem vazamento de dados.
Resultado: IAN cai para ~44% da importância (era 64%), AUC sobe para ~0,78 (era 0,72).

Também adicionamos validação cruzada (5-fold estratificada) *dentro do conjunto de
treino* (2022->2023): é uma checagem extra de estabilidade, mostrando accuracy,
precision, recall, F1 e ROC AUC com desvio padrão. O AUC out-of-time contra 2023->2024
continua sendo a avaliação principal — é a única que de fato simula prever o futuro a
partir do passado; o CV só confirma que o modelo não é instável dentro do próprio
conjunto de treino.

Importante sobre vazamento de dados: em nenhuma das quatro versões qualquer variável do
ano T+1 (o ano que está sendo previsto) entrou como feature — a validação temporal
out-of-time (treinar em 2022->2023, testar em 2023->2024) é a mesma desde a 1ª versão.
A concentração de importância em IAN nunca foi vazamento; era uma característica de como
cada modelo distribui importância entre variáveis correlacionadas. As novas variáveis
(Idade, Anos_na_PM) ajudam porque carregam sinal genuinamente independente do IAN.
--------------------------------------------------------------------------------------
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.modelo_utils import ModeloMonotonicoLogistico

# Caminhos calculados a partir da posição deste arquivo (src/train_model.py), então o
# script funciona não importa de onde ele seja chamado (raiz do repo, dentro de src/, etc.).
RAIZ_PROJETO = Path(__file__).resolve().parent.parent
DATA_PATH = RAIZ_PROJETO / "data" / "raw" / "PEDE_consolidado_longo.csv"
MODEL_PATH = RAIZ_PROJETO / "models" / "modelo_risco_defasagem.pkl"

# Variáveis do ano T. IAN só assume 2.5 / 5 / 10 na base; as demais são contínuas.
# Anos_na_PM é calculada a partir de "Ano ingresso" (não existe pronta na base).
FEATURES = ["IAN", "IDA", "IEG", "IAA", "IPS", "Idade", "Anos_na_PM"]

# Valores que o IAN realmente assume na base (usado também pelo app, para restringir
# o campo a um seletor em vez de um slider contínuo "inválido").
VALORES_VALIDOS_IAN = [2.5, 5.0, 10.0]

# Fórmula oficial de ponderação do INDE (confirmada empiricamente contra a base real:
# erro médio de ~0,002 em 1.985 registros testados). Não estava na documentação
# original da Passos Mágicos disponibilizada para o desafio; foi reconstruída a partir
# dos próprios dados. Guardada aqui para reaproveitar na pergunta 8 do painel analítico.
PESOS_INDE = {
    "IAN": 0.10, "IDA": 0.20, "IEG": 0.20, "IAA": 0.10,
    "IPS": 0.10, "IPP": 0.10, "IPV": 0.20,
}


def engenharia_features(df):
    """Cria as colunas derivadas usadas pelo modelo (feature engineering)."""
    df = df.copy()
    df["Anos_na_PM"] = df["Ano"] - df["Ano ingresso"]
    return df


def monta_transicao_risco(df_origem, ano_t, ano_t1):
    """Constrói a matriz de transição temporal sem vazamento de dados.

    Alvo: o aluno permanece (ou passa a estar) em situação crítica no ano seguinte
    (Defasagem <= -1). Features vêm exclusivamente do ano T; o ano T+1 só é usado
    para calcular o alvo, nunca entra como variável explicativa.
    """
    atual = df_origem[df_origem["Ano"] == ano_t].set_index("RA")
    futuro = df_origem[df_origem["Ano"] == ano_t1].set_index("RA")

    alunos_comuns = atual.index.intersection(futuro.index)

    X = atual.loc[alunos_comuns, FEATURES].copy()
    defasagem_t1 = futuro.loc[alunos_comuns, "Defasagem"]
    y = (defasagem_t1 <= -1).astype(int)

    return X, y


def validacao_cruzada(X_train, y_train, n_splits=5, random_state=42):
    """5-fold CV estratificada, só dentro do conjunto de treino (2022->2023).
    Retorna média e desvio padrão de accuracy/precision/recall/F1/ROC AUC."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metricas = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}

    for idx_treino, idx_val in skf.split(X_train, y_train):
        X_t, X_v = X_train.iloc[idx_treino], X_train.iloc[idx_val]
        y_t, y_v = y_train.iloc[idx_treino], y_train.iloc[idx_val]

        modelo_fold = ModeloMonotonicoLogistico(features=FEATURES)
        modelo_fold.fit(X_t, y_t, C=1.0)

        proba = modelo_fold.predict_proba(X_v)[:, 1]
        pred = (proba >= 0.5).astype(int)

        metricas["accuracy"].append(accuracy_score(y_v, pred))
        metricas["precision"].append(precision_score(y_v, pred, zero_division=0))
        metricas["recall"].append(recall_score(y_v, pred, zero_division=0))
        metricas["f1"].append(f1_score(y_v, pred, zero_division=0))
        metricas["roc_auc"].append(roc_auc_score(y_v, proba))

    return {
        nome: {"media": float(np.mean(valores)), "desvio": float(np.std(valores))}
        for nome, valores in metricas.items()
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df = engenharia_features(df)

    # Validação temporal out-of-time: treina no ciclo mais antigo, testa no mais recente
    X_train, y_train = monta_transicao_risco(df, 2022, 2023)
    X_test, y_test = monta_transicao_risco(df, 2023, 2024)

    medias_treino = X_train.mean()
    X_train = X_train.fillna(medias_treino)
    X_test = X_test.fillna(medias_treino)

    print(f"Treino (2022->2023): {len(X_train)} alunos | % em risco: {y_train.mean() * 100:.1f}%")
    print(f"Teste  (2023->2024): {len(X_test)} alunos | % em risco: {y_test.mean() * 100:.1f}%")

    # --- Validação cruzada (robustez, só dentro do treino) ---
    cv_metricas = validacao_cruzada(X_train, y_train)
    print("\n=== VALIDAÇÃO CRUZADA (5-fold, dentro do treino 2022->2023) ===")
    for nome, valores in cv_metricas.items():
        print(f"{nome:10s}: {valores['media']:.3f} +/- {valores['desvio']:.3f}")

    # --- Modelo final, avaliado out-of-time (2023->2024) ---
    model = ModeloMonotonicoLogistico(features=FEATURES)
    model.fit(X_train, y_train, C=1.0)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"\nAUC no teste (out-of-time): {auc:.3f}")
    print(classification_report(y_test, model.predict(X_test)))

    importancias_pct = model.importancias_normalizadas().sort_values(ascending=False)
    print("Importância relativa de cada indicador:")
    print(importancias_pct.round(1))

    # Sanidade 1: aluno "perfeito" (indicadores no máximo, idade e tempo de casa altos)
    # deve ficar com risco baixo.
    perfil_perfeito = pd.DataFrame(
        [[10.0, 10.0, 10.0, 10.0, 10.0, 16.0, 6.0]], columns=FEATURES
    )
    risco_perfeito = model.predict_proba(perfil_perfeito)[0, 1]
    print(f"\nChecagem de sanidade 1 — risco previsto p/ perfil exemplar: {risco_perfeito * 100:.1f}%")
    assert risco_perfeito < 0.30, "Risco para o perfil exemplar ficou alto demais — revisar o modelo."

    # Sanidade 2: nenhum indicador deve concentrar mais de 70% da importância total —
    # trava automática para não regredir ao problema da árvore de decisão.
    maior_peso = importancias_pct.max()
    assert maior_peso < 70, (
        f"Um indicador está concentrando {maior_peso:.1f}% da importância — "
        "revisar a regularização do modelo."
    )

    # Sanidade 3: o desafio exige um modelo com poder preditivo real — trava mínima de AUC.
    assert auc >= 0.70, f"AUC ficou abaixo do mínimo aceitável ({auc:.3f} < 0.70)."

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "medias_treino": medias_treino.to_dict(),
            "auc_teste": auc,
            "importancias": importancias_pct.to_dict(),
            "cv_metricas": cv_metricas,
            "pesos_inde": PESOS_INDE,
            "valores_validos_ian": VALORES_VALIDOS_IAN,
            "metodologia": (
                "Regressão logística com coeficientes restritos a sinal negativo "
                "(indicador maior => risco menor ou igual). Features: IAN, IDA, IEG, "
                "IAA, IPS, Idade, Anos_na_PM (Ano - Ano de ingresso). Validação "
                "temporal out-of-time: treino 2022->2023, teste 2023->2024. Alvo: "
                "Defasagem <= -1 no ano seguinte. Validação cruzada (5-fold) dentro "
                "do treino, como checagem extra de estabilidade."
            ),
        },
        MODEL_PATH,
    )
    print(f"\nModelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()
