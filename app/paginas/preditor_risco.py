"""Página 3 — Preditor de Risco: simulador interativo do modelo."""
import streamlit as st

from utils.modelo import prever_risco, selo_risco
from utils.tema import OPCOES_IAN


def render(modelo, modelo_pack, features, medias_treino, auc_teste):
    st.title("Preditor de Risco de Defasagem")
    st.markdown(
        """
        Informe os indicadores mais recentes do aluno para estimar a **probabilidade de ele
        permanecer (ou passar a estar) em situação crítica de defasagem no próximo ciclo**
        (`Defasagem ≤ -1`). O modelo foi validado de forma temporal (treino 2022→2023, teste
        2023→2024 — dados "do futuro" nunca entram no treino) e tem uma restrição matemática de
        monotonicidade: melhorar qualquer indicador nunca aumenta o risco calculado.
        """
    )
    if auc_teste:
        st.caption(f"Desempenho do modelo na validação out-of-time: AUC = {auc_teste:.3f}")

    st.markdown("---")

    valores_default = {
        "IAN": 10.0, "IDA": 6.7, "IEG": 8.6, "IAA": 7.7, "IPS": 6.0,
        "Idade": 12, "Anos_na_PM": 2,
    }
    campos_especiais = {"IAN", "Idade", "Anos_na_PM"}
    campos_indicadores = [f for f in features if f not in campos_especiais]

    col1, col2 = st.columns(2)
    entrada = {}
    with col1:
        if "IAN" in features:
            st.markdown("**IAN — Adequação de Nível**")
            entrada["IAN"] = st.selectbox(
                "IAN", options=[v for v, _ in OPCOES_IAN],
                format_func=lambda v: dict(OPCOES_IAN)[v],
                label_visibility="collapsed",
            )
        if "Idade" in features:
            entrada["Idade"] = st.slider("Idade", 6, 22, int(valores_default["Idade"]), 1)
        metade = (len(campos_indicadores) + 1) // 2
        for f in campos_indicadores[:metade]:
            entrada[f] = st.slider(f, 0.0, 10.0, valores_default.get(f, 7.0), 0.1, key=f"pred_{f}")
    with col2:
        if "Anos_na_PM" in features:
            entrada["Anos_na_PM"] = st.slider(
                "Anos na Passos Mágicos", 0, 10, int(valores_default["Anos_na_PM"]), 1
            )
        for f in campos_indicadores[metade:]:
            entrada[f] = st.slider(f, 0.0, 10.0, valores_default.get(f, 7.0), 0.1, key=f"pred_{f}")

    if st.button("Calcular risco", type="primary"):
        proba = prever_risco(entrada, modelo, features, medias_treino)
        st.markdown("---")
        selo_risco(proba)
        st.caption(
            "Esta é uma estimativa estatística baseada em padrões históricos da base PEDE — não "
            "substitui a avaliação da equipe pedagógica/psicopedagógica, mas pode ajudar a "
            "priorizar quem acompanhar primeiro."
        )
