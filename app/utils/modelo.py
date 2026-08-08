"""Carga do modelo treinado e funções de predição de risco."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# O modelo foi pickled com esta classe, definida em src/modelo_utils.py — precisa estar
# importável aqui para o joblib.load conseguir reconstruir o objeto. app.py já adiciona
# a raiz do repositório ao sys.path antes de qualquer página ser importada, então este
# import funciona como um pacote "src.modelo_utils" normal.
from src.modelo_utils import ModeloMonotonicoLogistico  # noqa: F401

# app/utils/modelo.py -> app/utils -> app -> raiz do repo
RAIZ_PROJETO = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = RAIZ_PROJETO / "models" / "modelo_risco_defasagem.pkl"


@st.cache_resource
def carregar_modelo():
    try:
        return joblib.load(MODEL_PATH)
    except ModuleNotFoundError as e:
        st.error(
            "❌ Não foi possível carregar o modelo (`modelo_risco_defasagem.pkl`) — erro: "
            f"`{e}`.\n\n"
            "Isso normalmente acontece quando a versão do **scikit-learn** instalada no "
            "ambiente é diferente da versão usada para treinar o modelo. Confira se o "
            "`requirements.txt` está atualizado e force a reinstalação das dependências "
            "(no Community Cloud: menu **⋮ → Reboot app**; localmente: "
            "`pip install -r requirements.txt --force-reinstall`)."
        )
        st.stop()


def prever_risco(valores: dict, modelo, features, medias_treino) -> float:
    """valores: dict com as chaves em `features`. Retorna probabilidade de risco (0-1)."""
    entrada = pd.DataFrame(
        [[valores.get(f, medias_treino.get(f, 0)) for f in features]], columns=features
    )
    return float(modelo.predict_proba(entrada)[0, 1])


def selo_risco(proba: float):
    if proba >= 0.66:
        st.error(f"🔴 Risco ALTO de defasagem no próximo ciclo — {proba * 100:.1f}%")
    elif proba >= 0.33:
        st.warning(f"🟡 Risco MODERADO — {proba * 100:.1f}% — vale acompanhar de perto.")
    else:
        st.success(f"🟢 Risco BAIXO de defasagem no próximo ciclo — {proba * 100:.1f}%")
    st.progress(min(max(proba, 0.0), 1.0))
