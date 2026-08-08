"""Página 4 — Cadastro de Novo Aluno: formulário + predição imediata de risco."""
from datetime import datetime

import pandas as pd
import streamlit as st

from utils.dados import NOVOS_ALUNOS_PATH, categoriza_defasagem
from utils.modelo import prever_risco, selo_risco
from utils.tema import OPCOES_IAN, ORDEM_PEDRA


def render(df, modelo, modelo_pack, features, medias_treino):
    st.title("Cadastro de Novo Aluno")
    st.markdown(
        """
        Preencha os dados abaixo para registrar um novo aluno na base local desta aplicação.
        Ao salvar, o cadastro é adicionado à tabela de alunos desta sessão e, como os
        indicadores já são preenchidos no formulário, você já vê **a estimativa de risco de
        defasagem** dele na hora.
        """
    )
    st.warning(
        "⚠️ No plano gratuito do Streamlit Community Cloud, o armazenamento em disco é "
        "**temporário**: os cadastros ficam disponíveis enquanto o app estiver ativo, mas podem "
        "ser perdidos se o serviço reiniciar/reimplantar. Use o botão **Baixar cadastros (CSV)** "
        "para guardar uma cópia definitiva."
    )

    with st.form("form_cadastro", clear_on_submit=False):
        st.subheader("Dados de identificação")
        c1, c2, c3 = st.columns(3)
        with c1:
            nome = st.text_input("Nome do aluno")
            ra = st.text_input("RA (se não souber, deixe em branco para gerar automaticamente)")
        with c2:
            ano = st.selectbox("Ano de referência", sorted(df["Ano"].unique().tolist() + [2025]), index=3)
            idade = st.number_input("Idade", min_value=5, max_value=25, value=12, step=1)
        with c3:
            genero = st.selectbox("Gênero", ["Feminino", "Masculino"])
            turma = st.text_input("Turma (ex.: 1A, 8B, ALFA)")

        c4, c5 = st.columns(2)
        with c4:
            fase_num = st.number_input("Fase (0 = Alfa)", min_value=0, max_value=8, value=1, step=1)
            instituicao = st.text_input("Instituição de ensino", value="Pública")
        with c5:
            pedra = st.selectbox("Pedra atual", ORDEM_PEDRA)
            anos_na_pm = st.number_input(
                "Anos na Passos Mágicos", min_value=0, max_value=15, value=1, step=1
            )

        defasagem = st.number_input(
            "Defasagem (Fase Efetiva − Fase Ideal)", min_value=-5, max_value=3, value=0, step=1
        )

        st.subheader("Indicadores (escala 0 a 10)")
        c6, c7, c8, c9 = st.columns(4)
        with c6:
            inde = st.slider("INDE", 0.0, 10.0, 7.4, 0.1)
            iaa = st.slider("IAA", 0.0, 10.0, 7.7, 0.1)
        with c7:
            ieg = st.slider("IEG", 0.0, 10.0, 8.6, 0.1)
            ips = st.slider("IPS", 0.0, 10.0, 6.0, 0.1)
        with c8:
            ipp = st.slider("IPP", 0.0, 10.0, 7.5, 0.1)
            ida = st.slider("IDA", 0.0, 10.0, 6.7, 0.1)
        with c9:
            ipv = st.slider("IPV", 0.0, 10.0, 7.0, 0.1)
            ian = st.selectbox(
                "IAN — Adequação de Nível", options=[v for v, _ in OPCOES_IAN],
                format_func=lambda v: dict(OPCOES_IAN)[v],
            )

        st.subheader("Notas por disciplina (opcional)")
        c10, c11, c12 = st.columns(3)
        with c10:
            mat = st.number_input("Matemática", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
        with c11:
            por = st.number_input("Português", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
        with c12:
            ing = st.number_input("Inglês", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

        enviado = st.form_submit_button("Salvar cadastro", type="primary")

    if enviado:
        if not nome.strip():
            st.error("Informe o nome do aluno antes de salvar.")
        else:
            ra_final = ra.strip() if ra.strip() else f"NOVO-{int(datetime.now().timestamp())}"
            novo = {
                "Ano": ano, "RA": ra_final, "Nome": nome.strip(), "Fase_Num": fase_num,
                "Turma": turma, "Idade": idade, "Gênero": genero,
                "Instituicao_Ensino": instituicao, "Pedra_Atual": pedra, "INDE": inde,
                "IAA": iaa, "IEG": ieg, "IPS": ips, "IPP": ipp, "IDA": ida,
                "Matematica": mat, "Portugues": por, "Ingles": ing, "IPV": ipv, "IAN": ian,
                "Anos_na_PM": anos_na_pm, "Defasagem": defasagem,
                "Cat_Defasagem": categoriza_defasagem(defasagem),
                "Cadastrado_em": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            st.session_state["novos_alunos"] = pd.concat(
                [st.session_state["novos_alunos"], pd.DataFrame([novo])], ignore_index=True
            )
            try:
                st.session_state["novos_alunos"].to_csv(NOVOS_ALUNOS_PATH, index=False)
            except Exception:
                pass

            st.success(f"Aluno **{nome}** (RA: {ra_final}) cadastrado com sucesso!")

            proba = prever_risco({
                "IAN": ian, "IDA": ida, "IEG": ieg, "IAA": iaa, "IPS": ips,
                "IPP": ipp, "Idade": idade, "Anos_na_PM": anos_na_pm,
            }, modelo, features, medias_treino)
            st.subheader("Estimativa de risco para este aluno")
            selo_risco(proba)

    st.markdown("---")
    st.subheader("Alunos cadastrados nesta base local")
    if len(st.session_state["novos_alunos"]) == 0:
        st.caption("Nenhum aluno cadastrado ainda.")
    else:
        st.dataframe(st.session_state["novos_alunos"], use_container_width=True)
        csv_bytes = st.session_state["novos_alunos"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar cadastros (CSV)", data=csv_bytes,
            file_name="novos_alunos_cadastrados.csv", mime="text/csv",
        )
