"""
Funções de limpeza da base PEDE (2022, 2023, 2024).

Este módulo existe para que a lógica de limpeza não fique duplicada — o notebook
`01_limpeza_dados_PEDE.ipynb` importa e chama estas mesmas funções, em vez de
redefini-las inline. Assim, se um bug de limpeza for corrigido aqui, tanto o notebook
quanto qualquer outro script que reprocesse os dados brutos usam a correção.
"""
import re

import numpy as np
import pandas as pd

PEDRA_FIX = {"Agata": "Ágata", "AGATA": "Ágata", "agata": "Ágata"}

COLUNAS_COMUNS = [
    "Ano", "RA", "Nome", "Fase", "Fase_Num", "Fase_Ideal", "Turma",
    "Idade", "Gênero", "Ano ingresso", "Instituicao_Ensino",
    "Pedra_Atual", "INDE", "IAA", "IEG", "IPS", "IPP", "IDA",
    "Matematica", "Portugues", "Ingles", "IPV", "IAN", "Defasagem",
    "Pendente_Inclusao",
]


def drop_fully_empty_columns(df):
    """Remove colunas 100% nulas (campos legado sem preenchimento no ano)."""
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    return df.drop(columns=empty_cols), empty_cols


def drop_duplicated_content_columns(df):
    """Remove colunas '.1' quando o conteúdo é idêntico ao da coluna original."""
    dupes = []
    for c in df.columns:
        if c.endswith(".1"):
            base = c[:-2]
            if base in df.columns and df[c].equals(df[base]):
                dupes.append(c)
    return df.drop(columns=dupes), dupes


def strip_strings(df):
    """Remove espaços extras/duplicados em todas as colunas de texto."""
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
            df[c] = df[c].apply(
                lambda x: re.sub(r"\s+", " ", x).strip() if isinstance(x, str) else x
            )
    return df


def fix_accent_variants(series, mapping):
    return series.replace(mapping)


def clean_2022(df):
    df = df.copy()
    df, empty_cols = drop_fully_empty_columns(df)
    df = strip_strings(df)

    # Padroniza Gênero para o mesmo vocabulário de 2023/2024
    df["Gênero"] = df["Gênero"].replace({"Menina": "Feminino", "Menino": "Masculino"})

    # Corrige acentuação inconsistente em "Pedra"
    for c in ["Pedra 20", "Pedra 21", "Pedra 22"]:
        if c in df.columns:
            df[c] = fix_accent_variants(df[c], PEDRA_FIX)

    df = df.rename(columns={
        "Ano nasc": "Ano_Nascimento",
        "Idade 22": "Idade",
        "Instituição de ensino": "Instituicao_Ensino",
        "Pedra 22": "Pedra_Atual",
        "INDE 22": "INDE",
        "Matem": "Matematica",
        "Portug": "Portugues",
        "Inglês": "Ingles",
        "Fase ideal": "Fase_Ideal",
        "Defas": "Defasagem",
    })

    df["RA"] = df["RA"].astype(str).str.strip()
    df["Fase"] = pd.to_numeric(df["Fase"], errors="coerce")
    df["Fase_Num"] = df["Fase"]          # já é numérica em 2022
    df["Pendente_Inclusao"] = False       # não existe esse caso em 2022
    df["Ano"] = 2022
    return df, empty_cols


def clean_2023(df):
    df = df.copy()
    df, empty_cols = drop_fully_empty_columns(df)
    df, dup_cols = drop_duplicated_content_columns(df)
    df = strip_strings(df)

    # Corrige o bug de Idade importada como data (1900-01-DD)
    def fix_idade(v):
        if hasattr(v, "day") and hasattr(v, "month"):
            return v.day
        return v
    df["Idade"] = df["Idade"].apply(fix_idade)
    df["Idade"] = pd.to_numeric(df["Idade"], errors="coerce")

    # Padroniza Data de Nasc (mistura de string e datetime)
    df["Data de Nasc"] = pd.to_datetime(df["Data de Nasc"], errors="coerce")

    for c in ["Pedra 20", "Pedra 21", "Pedra 22", "Pedra 2023"]:
        if c in df.columns:
            df[c] = fix_accent_variants(df[c], PEDRA_FIX)

    df = df.rename(columns={
        "Nome Anonimizado": "Nome",
        "Data de Nasc": "Data_Nascimento",
        "Instituição de ensino": "Instituicao_Ensino",
        "Pedra 2023": "Pedra_Atual",
        "INDE 2023": "INDE",
        "Mat": "Matematica",
        "Por": "Portugues",
        "Ing": "Ingles",
        "Fase Ideal": "Fase_Ideal",
    })

    df["RA"] = df["RA"].astype(str).str.strip()

    def parse_fase(v):
        if not isinstance(v, str):
            return np.nan
        v = v.strip().upper()
        if v == "ALFA":
            return 0
        m = re.search(r"\d+", v)
        return int(m.group()) if m else np.nan

    df["Fase_Num"] = df["Fase"].apply(parse_fase)
    df["Pendente_Inclusao"] = False
    df["Ano"] = 2023
    return df, empty_cols + dup_cols


def clean_2024(df):
    df = df.copy()
    df, empty_cols = drop_fully_empty_columns(df)
    df, dup_cols = drop_duplicated_content_columns(df)
    df = strip_strings(df)

    df["Data de Nasc"] = pd.to_datetime(df["Data de Nasc"], errors="coerce")
    df["Idade"] = pd.to_numeric(df["Idade"], errors="coerce")

    for c in ["Pedra 20", "Pedra 21", "Pedra 22", "Pedra 23", "Pedra 2024"]:
        if c in df.columns:
            df[c] = fix_accent_variants(df[c], PEDRA_FIX)

    # Sinaliza alunos "a incluir" em vez de descartar o registro
    df["Pendente_Inclusao"] = df["Pedra 2024"].eq("INCLUIR")

    def parse_fase_2024(v):
        if not isinstance(v, str):
            return np.nan
        v = v.strip().upper()
        if v == "ALFA":
            return 0
        m = re.match(r"(\d+)", v)
        return int(m.group(1)) if m else np.nan

    df["Fase_Num"] = df["Fase"].astype(str).apply(parse_fase_2024)
    df.loc[df["Pendente_Inclusao"], "Fase_Num"] = np.nan

    df = df.rename(columns={
        "Nome Anonimizado": "Nome",
        "Data de Nasc": "Data_Nascimento",
        "Instituição de ensino": "Instituicao_Ensino",
        "Escola": "Nome_Escola",
        "Pedra 2024": "Pedra_Atual",
        "INDE 2024": "INDE",
        "Mat": "Matematica",
        "Por": "Portugues",
        "Ing": "Ingles",
        "Fase Ideal": "Fase_Ideal",
    })

    df["RA"] = df["RA"].astype(str).str.strip()
    df["INDE"] = pd.to_numeric(df["INDE"], errors="coerce")
    df["Ano"] = 2024
    return df, empty_cols + dup_cols


def select_common(df):
    """Seleciona/ordena as colunas comuns aos três anos, preenchendo com NaN
    as que não existirem naquele ano específico."""
    out = df[[c for c in COLUNAS_COMUNS if c in df.columns]].copy()
    for c in COLUNAS_COMUNS:
        if c not in out.columns:
            out[c] = np.nan
    return out[COLUNAS_COMUNS]


def limpar_base_completa(caminho_excel):
    """Lê o Excel bruto (3 abas: PEDE2022/2023/2024), limpa cada ano e devolve o
    dataframe consolidado em formato longo (1 linha = 1 aluno em 1 ano)."""
    df_2022_raw = pd.read_excel(caminho_excel, sheet_name="PEDE2022")
    df_2023_raw = pd.read_excel(caminho_excel, sheet_name="PEDE2023")
    df_2024_raw = pd.read_excel(caminho_excel, sheet_name="PEDE2024")

    df_2022, _ = clean_2022(df_2022_raw)
    df_2023, _ = clean_2023(df_2023_raw)
    df_2024, _ = clean_2024(df_2024_raw)

    df_long = pd.concat(
        [select_common(df_2022), select_common(df_2023), select_common(df_2024)],
        ignore_index=True,
    )
    return df_long
