"""
Treina o modelo de risco de defasagem e salva o artefato usado pelo app.py.

Rode este script sempre que a base `PEDE_consolidado_longo.csv` for atualizada
(ex.: quando chegar o ciclo PEDE 2025, ou depois de novos cadastros):

    python train_model.py

Gera: modelo_risco_defasagem.pkl

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

Importante sobre vazamento de dados: em nenhuma das quatro versões qualquer variável do
ano T+1 (o ano que está sendo previsto) entrou como feature — a validação temporal
out-of-time (treinar em 2022->2023, testar em 2023->2024) é a mesma desde a 1ª versão.
A concentração de importância em IAN nunca foi vazamento; era uma característica de como
cada modelo distribui importância entre variáveis correlacionadas. As novas variáveis
(Idade, Anos_na_PM) ajudam porque carregam sinal genuinamente independente do IAN.
--------------------------------------------------------------------------------------
"""
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from modelo_utils import ModeloMonotonicoLogistico

DATA_PATH = "PEDE_consolidado_longo.csv"
MODEL_PATH = "modelo_risco_defasagem.pkl"

# Variáveis do ano T. IAN só assume 2.5 / 5 / 10 na base; as demais são contínuas.
# Anos_na_PM é calculada a partir de "Ano ingresso" (não existe pronta na base).
FEATURES = ["IAN", "IDA", "IEG", "IAA", "IPS", "Idade", "Anos_na_PM"]

# Valores que o IAN realmente assume na base (usado também pelo app, para restringir
# o campo a um seletor em vez de um slider contínuo "inválido").
VALORES_VALIDOS_IAN = [2.5, 5.0, 10.0]


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
            "valores_validos_ian": VALORES_VALIDOS_IAN,
            "metodologia": (
                "Regressão logística com coeficientes restritos a sinal negativo "
                "(indicador maior => risco menor ou igual). Features: IAN, IDA, IEG, "
                "IAA, IPS, Idade, Anos_na_PM (Ano - Ano de ingresso). Validação "
                "temporal out-of-time: treino 2022->2023, teste 2023->2024. Alvo: "
                "Defasagem <= -1 no ano seguinte."
            ),
        },
        MODEL_PATH,
    )
    print(f"\nModelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()
