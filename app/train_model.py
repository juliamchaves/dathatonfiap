"""
Treina o modelo de risco de defasagem e salva o artefato usado pelo app.py.

Rode este script sempre que a base `PEDE_consolidado_longo.csv` for atualizada
(ex.: quando chegar o ciclo PEDE 2025, ou depois de novos cadastros):

    python train_model.py

Gera: modelo_risco_defasagem.pkl

--------------------------------------------------------------------------------------
Por que HistGradientBoostingClassifier com restrição de monotonicidade (e não Random
Forest / Regressão Logística "soltos")?

1. O IAN só assume 3 valores reais na base (2,5 / 5 / 10 — mapeados a partir da
   Defasagem). Um Random Forest treinado com poucas combinações de valores extremos
   (ex.: todos os indicadores em 10) pode extrapolar mal para esse canto raro do
   espaço de dados, gerando risco "não intuitivo" mesmo para um aluno exemplar.
2. Uma Regressão Logística "solta" pode até aprender coeficiente de sinal errado para
   alguma variável fraca (ex.: IPS levemente positivo em vez de negativo) por puro
   ruído na base, novamente quebrando a intuição de "indicador melhor = risco menor".
3. `monotonic_cst=[-1, -1, -1, -1, -1]` força, por construção, que aumentar qualquer
   um dos 5 indicadores NUNCA aumenta a probabilidade prevista de risco — é uma
   restrição de domínio (sabemos que isso deveria ser verdade) imposta ao modelo, não
   algo que ele precisa "adivinhar" sozinho a partir de poucos exemplos extremos.
--------------------------------------------------------------------------------------
"""
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = "PEDE_consolidado_longo.csv"
MODEL_PATH = "modelo_risco_defasagem.pkl"

# Variáveis do ano T. Todas contínuas na base, exceto IAN (só assume 2.5 / 5 / 10).
FEATURES = ["IAN", "IDA", "IEG", "IAA", "IPS"]

# Valores que o IAN realmente assume na base (usado também pelo app, para restringir
# o campo a um seletor em vez de um slider contínuo "inválido").
VALORES_VALIDOS_IAN = [2.5, 5.0, 10.0]


def monta_transicao_risco(df_origem, ano_t, ano_t1):
    """Constrói a matriz de transição temporal sem vazamento de dados.

    Alvo: o aluno permanece (ou passa a estar) em situação crítica no ano seguinte
    (Defasagem <= -1). Usar só esse critério — em vez de também contar "qualquer
    piora, mesmo entre dois anos adequados" — evita marcar como risco alunos que já
    estão bem e só oscilaram um pouco dentro da faixa adequada.
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

    # Validação temporal out-of-time: treina no ciclo mais antigo, testa no mais recente
    X_train, y_train = monta_transicao_risco(df, 2022, 2023)
    X_test, y_test = monta_transicao_risco(df, 2023, 2024)

    medias_treino = X_train.mean()
    X_train = X_train.fillna(medias_treino)
    X_test = X_test.fillna(medias_treino)

    print(f"Treino (2022->2023): {len(X_train)} alunos | % em risco: {y_train.mean() * 100:.1f}%")
    print(f"Teste  (2023->2024): {len(X_test)} alunos | % em risco: {y_test.mean() * 100:.1f}%")

    # monotonic_cst = -1 em cada posição: "quanto maior este indicador, menor (ou
    # igual) deve ser o risco previsto" — nunca o contrário.
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=3,
        learning_rate=0.1,
        monotonic_cst=[-1] * len(FEATURES),
        random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"\nAUC no teste (out-of-time): {auc:.3f}")
    print(classification_report(y_test, model.predict(X_test)))

    # Sanidade: um aluno "perfeito" (todos os indicadores no máximo) deve ficar com
    # risco baixo. Isso é garantido pela restrição monotônica, mas confirmamos aqui.
    perfil_perfeito = pd.DataFrame([[10.0] * len(FEATURES)], columns=FEATURES)
    risco_perfeito = model.predict_proba(perfil_perfeito)[0, 1]
    print(f"Checagem de sanidade — risco previsto p/ perfil com todos os indicadores em 10: "
          f"{risco_perfeito * 100:.1f}%")
    assert risco_perfeito < 0.20, "Risco para o perfil perfeito ficou alto demais — revisar o modelo."

    # HistGradientBoostingClassifier não tem .feature_importances_ nativo (não é uma
    # floresta de árvores simples) — calculamos importância por permutação e salvamos
    # o resultado, para o app não precisar recalcular isso a cada carregamento.
    perm = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=42, scoring="roc_auc")
    importancias = pd.Series(perm.importances_mean, index=FEATURES).clip(lower=0)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "medias_treino": medias_treino.to_dict(),
            "auc_teste": auc,
            "importancias": importancias.to_dict(),
            "valores_validos_ian": VALORES_VALIDOS_IAN,
            "metodologia": (
                "HistGradientBoostingClassifier com restrição de monotonicidade "
                "(indicador maior => risco menor ou igual). Validação temporal "
                "out-of-time: treino 2022->2023, teste 2023->2024. Alvo: "
                "Defasagem <= -1 no ano seguinte."
            ),
        },
        MODEL_PATH,
    )
    print(f"\nModelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()
