"""Carga e preparação dos dados usados pelo app (com cache do Streamlit)."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# app/utils/dados.py -> app/utils -> app -> raiz do repo
RAIZ_PROJETO = Path(__file__).resolve().parent.parent.parent
DATA_PATH = RAIZ_PROJETO / "data" / "raw" / "PEDE_consolidado_longo.csv"
NOVOS_ALUNOS_PATH = RAIZ_PROJETO / "data" / "raw" / "novos_alunos_cadastrados.csv"

CAMPOS_CADASTRO = [
    "Ano", "RA", "Nome", "Fase_Num", "Turma", "Idade", "Gênero",
    "Instituicao_Ensino", "Pedra_Atual", "INDE", "IAA", "IEG", "IPS", "IPP",
    "IDA", "Matematica", "Portugues", "Ingles", "IPV", "IAN", "Defasagem",
]


def categoriza_defasagem(d):
    if pd.isna(d):
        return np.nan
    if d >= 0:
        return "Adequado/Adiantado"
    if d >= -2:
        return "Moderadamente defasado"
    return "Severamente defasado"


@st.cache_data
def carregar_dados():
    df = pd.read_csv(DATA_PATH)
    df["Cat_Defasagem"] = df["Defasagem"].apply(categoriza_defasagem)
    df["Anos_na_PM"] = df["Ano"] - df["Ano ingresso"]
    return df


def carregar_novos_alunos():
    """Cadastros feitos pela própria aplicação (persistem em disco entre sessões,
    mas são resetados se o ambiente do Community Cloud reiniciar)."""
    if NOVOS_ALUNOS_PATH.exists():
        return pd.read_csv(NOVOS_ALUNOS_PATH)
    return pd.DataFrame(columns=CAMPOS_CADASTRO + ["Cat_Defasagem", "Cadastrado_em"])


def inicializar_sessao():
    """Garante que a tabela de novos cadastros exista no session_state."""
    if "novos_alunos" not in st.session_state:
        st.session_state["novos_alunos"] = carregar_novos_alunos()
