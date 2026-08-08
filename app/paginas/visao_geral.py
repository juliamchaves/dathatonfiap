"""Página 1 — Visão Geral: KPIs consolidados e evolução do INDE."""
import plotly.express as px
import streamlit as st

from utils.tema import AZUL, CORES_PEDRA, ORDEM_PEDRA, rotula


def render(df, df_f):
    st.title("Painel PEDE — Associação Passos Mágicos")
    st.markdown(
        """
        Este painel resume os principais achados da análise da **Pesquisa Extensiva do
        Desenvolvimento Educacional (PEDE)** de 2022 a 2024, disponibiliza um modelo preditivo
        para identificar alunos em risco de aumento de defasagem escolar, e permite cadastrar
        novos alunos diretamente por aqui.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alunos-ano na base", f"{len(df_f):,}".replace(",", "."))
    c2.metric("INDE médio", f"{df_f['INDE'].mean():.2f}")
    c3.metric(
        "% nível adequado/adiantado",
        f"{(df_f['Cat_Defasagem'] == 'Adequado/Adiantado').mean() * 100:.1f}%",
    )
    c4.metric(
        "% defasagem severa",
        f"{(df_f['Cat_Defasagem'] == 'Severamente defasado').mean() * 100:.1f}%",
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Evolução do INDE médio por ano")
        inde_ano = df.groupby("Ano")["INDE"].mean().reset_index()
        fig = px.line(
            inde_ano, x="Ano", y="INDE", markers=True, text="INDE",
            color_discrete_sequence=[AZUL],
        )
        fig.update_xaxes(dtick=1)
        rotula(fig, posicao="top center")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribuição de alunos por Pedra")
        pedra_count = (
            df_f[df_f["Pedra_Atual"].isin(ORDEM_PEDRA)]["Pedra_Atual"]
            .value_counts()
            .reindex(ORDEM_PEDRA)
            .reset_index()
        )
        pedra_count.columns = ["Pedra", "Quantidade"]
        fig = px.bar(
            pedra_count, x="Pedra", y="Quantidade", color="Pedra", text="Quantidade",
            color_discrete_map=CORES_PEDRA,
        )
        rotula(fig, casas=0)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Use o menu à esquerda para explorar o **Painel Analítico** (respostas às 11 perguntas "
        "do desafio), testar o **Preditor de Risco** de defasagem, ou fazer o **Cadastro de um "
        "novo aluno**."
    )
