"""
Treina o modelo de risco de defasagem e salva o artefato usado pelo app.py.

Metodologia alinhada ao notebook de análise (`Cópia_de_02_analise_exploratoria_PEDE.ipynb`,
seção 9): validação temporal *out-of-time* (não é split aleatório) — treina com a transição
2022->2023 e testa com a transição 2023->2024, simulando o uso real do modelo (prever o futuro
com dados do passado).

Rode este script sempre que a base `PEDE_consolidado_longo.csv` for atualizada
(ex.: quando chegar o ciclo PEDE 2025, ou depois de novos cadastros):

    python train_model.py

Gera: modelo_risco_defasagem.pkl
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = "PEDE_consolidado_longo.csv"
MODEL_PATH = "modelo_risco_defasagem.pkl"

# Variáveis do ano T (status inicial + pedagógicas/comportamentais)
FEATURES = ["IAN", "IDA", "IEG", "IAA", "IPS", "IPP"]


def monta_transicao_risco(df_origem, ano_t, ano_t1):
    """Constrói a matriz de transição temporal sem vazamento de dados.

    Alvo: o aluno piorou a defasagem em T+1 (ficou mais defasado que em T) OU já
    permanece em situação crítica (Defasagem <= -1) em T+1.
    """
    atual = df_origem[df_origem["Ano"] == ano_t].set_index("RA")
    futuro = df_origem[df_origem["Ano"] == ano_t1].set_index("RA")

    alunos_comuns = atual.index.intersection(futuro.index)

    X = atual.loc[alunos_comuns, FEATURES].copy()

    defasagem_t = atual.loc[alunos_comuns, "Defasagem"]
    defasagem_t1 = futuro.loc[alunos_comuns, "Defasagem"]
    y = ((defasagem_t1 < defasagem_t) | (defasagem_t1 <= -1)).astype(int)

    return X, y


def main():
    df = pd.read_csv(DATA_PATH)

    # Validação temporal out-of-time: treina no ciclo mais antigo, testa no mais recente
    X_train, y_train = monta_transicao_risco(df, 2022, 2023)
    X_test, y_test = monta_transicao_risco(df, 2023, 2024)

    # IPP não existe em 2022 (coluna 100% vazia nesse ano) -> é descartada automaticamente
    cols_vazias = X_train.columns[X_train.isna().all()]
    if len(cols_vazias) > 0:
        print(f"Colunas descartadas por virem 100% vazias no treino: {list(cols_vazias)}")
        X_train = X_train.drop(columns=cols_vazias)
        X_test = X_test.drop(columns=cols_vazias, errors="ignore")

    medias_treino = X_train.mean()
    X_train = X_train.fillna(medias_treino)
    X_test = X_test.fillna(medias_treino)

    features_finais = list(X_train.columns)
    print(f"Features finais do modelo: {features_finais}")
    print(f"Treino (2022->2023): {len(X_train)} alunos | % em risco: {y_train.mean() * 100:.1f}%")
    print(f"Teste  (2023->2024): {len(X_test)} alunos | % em risco: {y_test.mean() * 100:.1f}%")

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_s, y_train)
    prob_lr = log_reg.predict_proba(X_test_s)[:, 1]
    print("\n=== REGRESSÃO LOGÍSTICA ===")
    print("AUC-ROC:", round(roc_auc_score(y_test, prob_lr), 3))
    print(classification_report(y_test, log_reg.predict(X_test_s)))

    rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    prob_rf = rf.predict_proba(X_test)[:, 1]
    print("\n=== RANDOM FOREST ===")
    auc_rf = roc_auc_score(y_test, prob_rf)
    print("AUC-ROC:", round(auc_rf, 3))
    print(classification_report(y_test, rf.predict(X_test)))

    joblib.dump(
        {
            "model": rf,
            "scaler": scaler,
            "features": features_finais,
            "medias_treino": medias_treino[features_finais].to_dict(),
            "auc_teste": auc_rf,
            "metodologia": "Validação temporal out-of-time: treino 2022->2023, teste 2023->2024",
        },
        MODEL_PATH,
    )
    print(f"\nModelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()
