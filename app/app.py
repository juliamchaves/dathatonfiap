"""
Datathon Passos Mágicos — Aplicação Streamlit
Painel analítico (perguntas 1 a 11) + preditor de risco de defasagem + cadastro de novos alunos.

Alinhado ao notebook `Cópia_de_02_analise_exploratoria_PEDE.ipynb` (versão com as análises e o
modelo corrigidos: validação temporal out-of-time, indicadores antecedentes, etc.).

Para rodar localmente:
    streamlit run app.py

Estrutura de dados esperada na mesma pasta:
    PEDE_consolidado_longo.csv     -> base limpa e consolidada
    modelo_risco_defasagem.pkl     -> modelo treinado (gerado por train_model.py)
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================================
st.set_page_config(
    page_title="Passos Mágicos | Painel PEDE",
    page_icon="✨",
    layout="wide",
)

DATA_PATH = "PEDE_consolidado_longo.csv"
NOVOS_ALUNOS_PATH = "novos_alunos_cadastrados.csv"
MODEL_PATH = "modelo_risco_defasagem.pkl"

# ---- Paleta de cores do app: tons de azul e laranja ----
AZUL_ESCURO = "#1F4E8C"
AZUL = "#2B5C8F"
AZUL_CLARO = "#6FA8DC"
LARANJA_ESCURO = "#C9660B"
LARANJA = "#E67E22"
LARANJA_CLARO = "#F2B46D"

PALETA_SEQ = [AZUL, LARANJA, AZUL_CLARO, LARANJA_CLARO, AZUL_ESCURO, LARANJA_ESCURO]
ESCALA_DIVERGENTE = [[0.0, LARANJA_ESCURO], [0.5, "#FFFFFF"], [1.0, AZUL_ESCURO]]

# Define a paleta padrão do Plotly Express para qualquer gráfico sem cor explícita
px.defaults.color_discrete_sequence = PALETA_SEQ

CORES_PEDRA = {
    "Quartzo": AZUL_CLARO,
    "Ágata": LARANJA_CLARO,
    "Ametista": AZUL_ESCURO,
    "Topázio": LARANJA_ESCURO,
}
CORES_CAT_DEFASAGEM = {
    "Severamente defasado": LARANJA_ESCURO,
    "Moderadamente defasado": LARANJA_CLARO,
    "Adequado/Adiantado": AZUL_ESCURO,
}
CORES_MOVIMENTO = {
    "Desceu": LARANJA_ESCURO,
    "Manteve": LARANJA_CLARO,
    "Subiu": AZUL_ESCURO,
}
CORES_GENERO = {"Feminino": AZUL, "Masculino": LARANJA}
CORES_RESULTADO_IDA = {"Queda no IDA": LARANJA_ESCURO, "Manteve/Melhorou": AZUL}

# O IAN só assume estes 3 valores reais na base (vem da categoria de defasagem) —
# por isso é tratado como um seletor, e não como um slider contínuo.
OPCOES_IAN = [
    (10.0, "10 — Adequado/Adiantado (sem defasagem)"),
    (5.0, "5 — Moderadamente defasado"),
    (2.5, "2,5 — Severamente defasado"),
]

ORDEM_CAT = ["Severamente defasado", "Moderadamente defasado", "Adequado/Adiantado"]
ORDEM_PEDRA = ["Quartzo", "Ágata", "Ametista", "Topázio"]

CAMPOS_CADASTRO = [
    "Ano", "RA", "Nome", "Fase_Num", "Turma", "Idade", "Gênero",
    "Instituicao_Ensino", "Pedra_Atual", "INDE", "IAA", "IEG", "IPS", "IPP",
    "IDA", "Matematica", "Portugues", "Ingles", "IPV", "IAN", "Defasagem",
]


def rotula(fig, sufixo="", casas=2, posicao="outside"):
    """Mostra o valor de cada ponto/barra direto no gráfico (sem precisar passar o
    mouse), arredondado para `casas` decimais, com um sufixo opcional (ex.: '%')."""
    fig.update_traces(texttemplate=f"%{{text:.{casas}f}}{sufixo}", textposition=posicao)
    return fig


# =========================================================================
# CARGA DE DADOS E MODELO (com cache, para não recarregar a cada clique)
# =========================================================================
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
    return df


@st.cache_resource
def carregar_modelo():
    return joblib.load(MODEL_PATH)


def carregar_novos_alunos():
    """Cadastros feitos pela própria aplicação (persistem em disco entre sessões,
    mas são resetados se o ambiente do Community Cloud reiniciar)."""
    if os.path.exists(NOVOS_ALUNOS_PATH):
        return pd.read_csv(NOVOS_ALUNOS_PATH)
    return pd.DataFrame(columns=CAMPOS_CADASTRO + ["Cat_Defasagem", "Cadastrado_em"])


df = carregar_dados()
modelo_pack = carregar_modelo()
modelo = modelo_pack["model"]
FEATURES = modelo_pack["features"]
AUC_TESTE = modelo_pack.get("auc_teste")
MEDIAS_TREINO = modelo_pack.get("medias_treino", {})

if "novos_alunos" not in st.session_state:
    st.session_state["novos_alunos"] = carregar_novos_alunos()


# =========================================================================
# FUNÇÃO DE PREDIÇÃO (reutilizada nas páginas de Preditor e Cadastro)
# =========================================================================
def prever_risco(valores: dict) -> float:
    """valores: dict com as chaves em FEATURES. Retorna probabilidade de risco (0-1)."""
    entrada = pd.DataFrame([[valores.get(f, MEDIAS_TREINO.get(f, 0)) for f in FEATURES]], columns=FEATURES)
    return float(modelo.predict_proba(entrada)[0, 1])


def selo_risco(proba: float):
    if proba >= 0.66:
        st.error(f"🔴 Risco ALTO de defasagem no próximo ciclo — {proba * 100:.1f}%")
    elif proba >= 0.33:
        st.warning(f"🟡 Risco MODERADO — {proba * 100:.1f}% — vale acompanhar de perto.")
    else:
        st.success(f"🟢 Risco BAIXO de defasagem no próximo ciclo — {proba * 100:.1f}%")
    st.progress(min(max(proba, 0.0), 1.0))


# =========================================================================
# NAVEGAÇÃO (barra lateral)
# =========================================================================
st.sidebar.title("✨ Passos Mágicos")
st.sidebar.caption("Datathon — PEDE 2022-2024")

pagina = st.sidebar.radio(
    "Navegação",
    ["🏠 Visão Geral", "📊 Painel Analítico", "🔮 Preditor de Risco", "📝 Cadastro de Novo Aluno"],
)

st.sidebar.markdown("---")
anos_disponiveis = sorted(df["Ano"].unique())
anos_filtro = st.sidebar.multiselect(
    "Filtrar por ano", anos_disponiveis, default=anos_disponiveis
)
df_f = df[df["Ano"].isin(anos_filtro)] if anos_filtro else df

if len(st.session_state["novos_alunos"]) > 0:
    st.sidebar.markdown("---")
    st.sidebar.caption(f"📝 {len(st.session_state['novos_alunos'])} aluno(s) cadastrado(s) nesta base local")


# =========================================================================
# PÁGINA 1 — VISÃO GERAL
# =========================================================================
if pagina == "🏠 Visão Geral":
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


# =========================================================================
# PÁGINA 2 — PAINEL ANALÍTICO (perguntas 1 a 11)
# =========================================================================
elif pagina == "📊 Painel Analítico":
    st.title("Painel Analítico — Perguntas 1 a 11")

    aba = st.selectbox(
        "Escolha a pergunta",
        [
            "1. Adequação de nível (IAN)",
            "2. Desempenho acadêmico (IDA)",
            "3. Engajamento (IEG) x IDA/IPV",
            "4. Autoavaliação (IAA) x desempenho real",
            "5. Aspectos psicossociais (IPS) como indicador antecedente",
            "6. Aspectos psicopedagógicos (IPP) x IAN",
            "7. Ponto de virada (IPV)",
            "8. Multidimensionalidade x INDE",
            "9. Modelo preditivo (validação temporal out-of-time)",
            "10. Efetividade por Pedra",
            "11. Insights adicionais",
        ],
    )

    st.markdown("---")

    # ---------------------------------------------------------------
    if aba.startswith("1."):
        st.subheader("Perfil de defasagem (IAN) por ano")
        tab = (
            pd.crosstab(df["Ano"], df["Cat_Defasagem"], normalize="index") * 100
        )[ORDEM_CAT[::-1]].reset_index().melt(id_vars="Ano", var_name="Categoria", value_name="Percentual")
        fig = px.bar(
            tab, x="Ano", y="Percentual", color="Categoria", barmode="stack",
            category_orders={"Categoria": ORDEM_CAT[::-1]},
            color_discrete_map=CORES_CAT_DEFASAGEM, text="Percentual",
        )
        fig.update_xaxes(dtick=1)
        rotula(fig, sufixo="%", posicao="inside")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "**Leitura:** o nível adequado/adiantado sobe de **30% (2022) para 54% (2024)**, "
            "e a defasagem severa cai de 3,3% para 0,3% — melhora consistente ao longo dos ciclos."
        )

    # ---------------------------------------------------------------
    elif aba.startswith("2."):
        st.subheader("IDA médio por ano e por fase")
        col1, col2 = st.columns(2)
        with col1:
            ida_ano = df.groupby("Ano")["IDA"].mean().reset_index()
            fig = px.line(
                ida_ano, x="Ano", y="IDA", markers=True, title="IDA médio por ano",
                text="IDA", color_discrete_sequence=[AZUL],
            )
            fig.update_xaxes(dtick=1)
            rotula(fig, posicao="top center")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            ida_fase = df.groupby(["Fase_Num", "Ano"])["IDA"].mean().reset_index()
            fig = px.line(
                ida_fase, x="Fase_Num", y="IDA", color=ida_fase["Ano"].astype(str),
                markers=True, title="IDA médio por fase",
                labels={"color": "Ano"}, text="IDA",
            )
            rotula(fig, posicao="top center")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "**Leitura:** o IDA sobe de 6,09 (2022) para 6,66 (2023) e recua para 6,35 em 2024. "
            "Fases mais avançadas (a partir da Fase 3) tendem a ter IDA mais baixo."
        )

    # ---------------------------------------------------------------
    elif aba.startswith("3."):
        st.subheader("Engajamento (IEG) x Desempenho (IDA) e Ponto de Virada (IPV)")
        r1 = df["IEG"].corr(df["IDA"])
        r2 = df["IEG"].corr(df["IPV"])
        col1, col2 = st.columns(2)
        col1.metric("Correlação IEG x IDA", f"{r1:.2f}")
        col2.metric("Correlação IEG x IPV", f"{r2:.2f}")
        fig = px.scatter(
            df.sample(min(1500, len(df)), random_state=1), x="IEG", y="IDA", opacity=0.4,
            color_discrete_sequence=[AZUL],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "**Leitura:** correlação positiva e moderada — engajamento importa para o desempenho, "
            "mas não é o único fator."
        )

    # ---------------------------------------------------------------
    elif aba.startswith("4."):
        st.subheader("Autoavaliação (IAA) x Desempenho real (IDA) e Engajamento (IEG)")
        r1 = df["IAA"].corr(df["IDA"])
        r2 = df["IAA"].corr(df["IEG"])
        col1, col2 = st.columns(2)
        col1.metric("Correlação IAA x IDA", f"{r1:.2f}")
        col2.metric("Correlação IAA x IEG", f"{r2:.2f}")
        fig = px.scatter(
            df.sample(min(1500, len(df)), random_state=1), x="IAA", y="IDA", opacity=0.4,
            color_discrete_sequence=[LARANJA],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "**Leitura:** correlação fraca — a percepção do aluno sobre si mesmo não é um bom "
            "preditor do seu desempenho real."
        )

    # ---------------------------------------------------------------
    elif aba.startswith("5."):
        st.subheader("O IPS de um ano antecipa queda de IDA/IEG no ano seguinte?")

        df_hist = (
            df.dropna(subset=["RA", "Ano", "IPS", "IDA", "IEG"])
            .sort_values(["RA", "Ano"])
            .copy()
        )
        df_hist["IPS_Anterior"] = df_hist.groupby("RA")["IPS"].shift(1)
        df_hist["Delta_IDA"] = df_hist.groupby("RA")["IDA"].diff()
        df_hist["Delta_IEG"] = df_hist.groupby("RA")["IEG"].diff()
        df_hist = df_hist.dropna(subset=["IPS_Anterior", "Delta_IDA", "Delta_IEG"])

        r_ida = df_hist["IPS_Anterior"].corr(df_hist["Delta_IDA"])
        r_ieg = df_hist["IPS_Anterior"].corr(df_hist["Delta_IEG"])

        col1, col2 = st.columns(2)
        col1.metric("Correlação IPS anterior x Δ IDA", f"{r_ida:.2f}")
        col2.metric("Correlação IPS anterior x Δ IEG", f"{r_ieg:.2f}")

        df_hist["Resultado_IDA"] = np.where(df_hist["Delta_IDA"] < 0, "Queda no IDA", "Manteve/Melhorou")
        fig = px.box(
            df_hist, x="Resultado_IDA", y="IPS_Anterior", color="Resultado_IDA",
            points=False, title="IPS do ano anterior, por resultado de IDA no ano seguinte",
            color_discrete_map=CORES_RESULTADO_IDA,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "**Leitura:** níveis anteriores de IPS têm associação com alterações futuras de "
            "desempenho e engajamento. Alunos que caem no IDA/IEG tinham IPS anterior mais baixo "
            "na mediana. A relação é **mais evidente para o engajamento (r=0,19)** do que para o "
            "desempenho acadêmico (r=0,12) — sugerindo que o IPS pode funcionar como indicador "
            "antecedente de risco, principalmente ligado a vínculo e participação escolar."
        )

    # ---------------------------------------------------------------
    elif aba.startswith("6."):
        st.subheader("IPP por categoria de defasagem (IAN)")
        df_q6 = df.dropna(subset=["Cat_Defasagem", "IPP"])
        corr_ipp_ian = df_q6["IPP"].corr(df_q6["IAN"])
        fig = px.violin(
            df_q6, x="Cat_Defasagem", y="IPP", color="Cat_Defasagem",
            category_orders={"Cat_Defasagem": ORDEM_CAT[::-1]}, box=True,
            color_discrete_map=CORES_CAT_DEFASAGEM,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"**Leitura:** a correlação direta é baixa (r={corr_ipp_ian:.2f}), mas há uma redução "
            "gradual das medianas de IPP entre adequado → moderadamente defasado → severamente "
            "defasado. O IPP sugere uma tendência de confirmação da defasagem apontada pelo IAN, "
            "porém sem forte associação linear — é preciso considerar outros fatores no "
            "diagnóstico."
        )

    # ---------------------------------------------------------------
    elif aba.startswith("7."):
        st.subheader("O que mais se relaciona com o Ponto de Virada (IPV)?")
        cols_indicadores = ["IAA", "IEG", "IPS", "IPP", "IDA", "IAN"]
        corr_ipv = df[cols_indicadores + ["IPV"]].corr()["IPV"].drop("IPV").sort_values()
        fig = px.bar(
            x=corr_ipv.values, y=corr_ipv.index, orientation="h",
            labels={"x": "Correlação com IPV", "y": "Indicador"},
            text=corr_ipv.values, color_discrete_sequence=[AZUL],
        )
        rotula(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "**Leitura:** IPP (0,61), IDA e IEG (0,56 cada) são os mais associados ao ponto de "
            "virada — acompanhamento psicopedagógico + desempenho + engajamento."
        )

    # ---------------------------------------------------------------
    elif aba.startswith("8."):
        st.subheader("Quais indicadores mais explicam o INDE?")
        cols_inde = ["IAA", "IEG", "IPS", "IPP", "IDA", "IPV", "IAN"]
        corr_inde = df[cols_inde + ["INDE"]].corr()["INDE"].drop("INDE").sort_values()
        fig = px.bar(
            x=corr_inde.values, y=corr_inde.index, orientation="h",
            labels={"x": "Correlação com INDE", "y": "Indicador"},
            text=corr_inde.values, color_discrete_sequence=[AZUL],
        )
        rotula(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Matriz de correlação completa")
        matriz = df[cols_inde + ["INDE"]].corr().round(2)
        fig2 = px.imshow(
            matriz, text_auto=".2f", color_continuous_scale=ESCALA_DIVERGENTE, zmin=-1, zmax=1,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(
            "**Leitura:** IDA (0,79), IEG (0,75) e IPV (0,72) são os que mais elevam o INDE — "
            "desempenho acadêmico + engajamento + ponto de virada combinados."
        )

    # ---------------------------------------------------------------
    elif aba.startswith("9."):
        st.subheader("Modelo preditivo de risco de defasagem")
        st.markdown(
            """
            **Metodologia:** validação temporal *out-of-time* — o modelo é treinado com a
            transição **2022 → 2023** e testado na transição **2023 → 2024**, simulando o uso
            real (prever o futuro a partir de dados do passado, sem misturar os dois períodos).
            O alvo é: o aluno **permanece (ou passa a estar) em situação crítica** no ano
            seguinte (`Defasagem ≤ -1`).

            O modelo usa uma **restrição de monotonicidade**: por construção, aumentar qualquer
            um dos 5 indicadores nunca eleva o risco previsto — isso evita comportamento
            contraintuitivo em perfis raros na base (ex.: um aluno com todos os indicadores no
            máximo aparecendo com risco alto só por falta de exemplos parecidos no treino).
            """
        )
        if AUC_TESTE:
            st.metric("AUC — validação out-of-time", f"{AUC_TESTE:.3f}")

        importancias = pd.Series(modelo_pack.get("importancias", {})).sort_values()
        fig = px.bar(
            x=importancias.values, y=importancias.index, orientation="h",
            labels={"x": "Importância (por permutação)", "y": "Variável"},
            title="Quais indicadores mais pesam na previsão de risco?",
            text=importancias.values, color_discrete_sequence=[AZUL],
        )
        rotula(fig, casas=3)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "**Leitura:** o IAN concentra a maior parte do sinal preditivo — o que faz sentido, "
            "já que ele reflete diretamente a defasagem atual do aluno. IEG e IDA aparecem em "
            "seguida. É uma ferramenta estatisticamente válida para apoiar a priorização de "
            "intervenção preventiva — vá até **🔮 Preditor de Risco** para simular um aluno."
        )

    # ---------------------------------------------------------------
    elif aba.startswith("10."):
        st.subheader("INDE médio por Pedra, ao longo dos anos")
        inde_pedra = df.groupby(["Ano", "Pedra_Atual"])["INDE"].mean().reset_index()
        inde_pedra = inde_pedra[inde_pedra["Pedra_Atual"].isin(ORDEM_PEDRA)]
        fig = px.line(
            inde_pedra, x="Ano", y="INDE", color="Pedra_Atual", markers=True,
            category_orders={"Pedra_Atual": ORDEM_PEDRA}, color_discrete_map=CORES_PEDRA,
            text="INDE",
        )
        fig.update_xaxes(dtick=1)
        rotula(fig, posicao="top center")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Movimentação de Pedra entre anos consecutivos")
        ordem_map = {p: i for i, p in enumerate(ORDEM_PEDRA)}
        piv_pedra = df.pivot_table(index="RA", columns="Ano", values="Pedra_Atual", aggfunc="first")
        piv_pedra = piv_pedra.dropna(subset=[2022, 2023, 2024])

        def resume(a, b):
            ra, rb = a.map(ordem_map), b.map(ordem_map)
            return (rb > ra).mean() * 100, (rb == ra).mean() * 100, (rb < ra).mean() * 100

        s1, m1, d1 = resume(piv_pedra[2022], piv_pedra[2023])
        s2, m2, d2 = resume(piv_pedra[2023], piv_pedra[2024])
        resumo = pd.DataFrame({
            "Transição": ["2022→2023", "2022→2023", "2022→2023", "2023→2024", "2023→2024", "2023→2024"],
            "Movimento": ["Subiu", "Manteve", "Desceu"] * 2,
            "Percentual": [s1, m1, d1, s2, m2, d2],
        })
        fig = px.bar(
            resumo, x="Transição", y="Percentual", color="Movimento", barmode="stack",
            category_orders={"Movimento": ["Desceu", "Manteve", "Subiu"]},
            color_discrete_map=CORES_MOVIMENTO, text="Percentual",
        )
        rotula(fig, sufixo="%", posicao="inside")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Base: {len(piv_pedra)} alunos presentes nos três anos.")
        st.markdown(
            "**Leitura:** as Pedras refletem faixas crescentes de INDE (~5,40 no Quartzo até "
            "~8,47 no Topázio), mostrando um critério de nivelamento consistente. Acompanhando os "
            "mesmos alunos ao longo dos 3 anos, cerca de **26% sobem de categoria** e a maioria "
            "consolida sua posição nas faixas superiores — evidência de evolução acadêmica "
            "sustentável."
        )

    # ---------------------------------------------------------------
    elif aba.startswith("11."):
        st.subheader("Insights adicionais")

        df_insights = df.copy()
        df_insights["Instituicao_Ensino"] = (
            df_insights["Instituicao_Ensino"].astype(str).str.strip()
            .replace({"Escola Pública": "Pública", "Publica": "Pública"})
        )

        col1, col2 = st.columns(2)
        with col1:
            genero = df_insights.groupby("Gênero")["INDE"].mean().reset_index()
            fig = px.bar(
                genero, x="Gênero", y="INDE", color="Gênero", text="INDE",
                title="INDE médio por gênero", color_discrete_map=CORES_GENERO,
            )
            rotula(fig)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            inst = (
                df_insights.groupby("Instituicao_Ensino")["INDE"]
                .agg(["mean", "count"]).sort_values("count", ascending=False).head(5)
                .reset_index()
            )
            fig = px.bar(
                inst, x="Instituicao_Ensino", y="mean", text="mean",
                title="INDE médio por instituição (unificadas)",
                color_discrete_sequence=[AZUL],
            )
            fig.update_xaxes(tickangle=25)
            rotula(fig)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Matriz de Engajamento (IEG) x Desempenho (IDA), por Pedra")
        fig3 = px.scatter(
            df_insights, x="IEG", y="IDA", color="Pedra_Atual",
            category_orders={"Pedra_Atual": ORDEM_PEDRA}, color_discrete_map=CORES_PEDRA,
            opacity=0.5,
        )
        fig3.add_hline(y=df_insights["IDA"].mean(), line_dash="dash", line_color=LARANJA_ESCURO)
        fig3.add_vline(x=df_insights["IEG"].mean(), line_dash="dash", line_color=AZUL_ESCURO)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown(
            """
            1. **Gênero:** diferença de INDE médio pequena (≈0,15 ponto) — não parece um fator
               relevante de desigualdade dentro do programa.
            2. **Rede de ensino:** alunos em escolas privadas via apadrinhamento/bolsa têm INDE
               médio mais alto que os de escola pública — vale investigar se é efeito do suporte
               extra ou viés de seleção dos indicados a essas vagas.
            3. **Fases intermediárias como ponto de atenção:** o IDA cai nas fases mais avançadas
               (pergunta 2) — reforço extra a partir da Fase 3 pode valer a pena.
            4. **Sinal de risco antecipável:** o modelo da pergunta 9 mostra que dá para sinalizar,
               com boa antecedência, quais alunos correm risco de aumentar sua defasagem — abrindo
               espaço para intervenção preventiva antes que o problema se agrave.
            """
        )


# =========================================================================
# PÁGINA 3 — PREDITOR DE RISCO
# =========================================================================
elif pagina == "🔮 Preditor de Risco":
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
    if AUC_TESTE:
        st.caption(f"Desempenho do modelo na validação out-of-time: AUC = {AUC_TESTE:.3f}")

    st.markdown("---")

    valores_default = {"IAN": 10.0, "IDA": 6.7, "IEG": 8.6, "IAA": 7.7, "IPS": 6.0}

    col1, col2 = st.columns(2)
    entrada = {}
    campos = [f for f in FEATURES if f != "IAN"]
    metade = (len(campos) + 1) // 2
    with col1:
        st.markdown("**IAN — Adequação de Nível**")
        entrada["IAN"] = st.selectbox(
            "IAN", options=[v for v, _ in OPCOES_IAN],
            format_func=lambda v: dict(OPCOES_IAN)[v],
            label_visibility="collapsed",
        )
        for f in campos[:metade]:
            entrada[f] = st.slider(f, 0.0, 10.0, valores_default.get(f, 7.0), 0.1, key=f"pred_{f}")
    with col2:
        for f in campos[metade:]:
            entrada[f] = st.slider(f, 0.0, 10.0, valores_default.get(f, 7.0), 0.1, key=f"pred_{f}")

    if st.button("Calcular risco", type="primary"):
        proba = prever_risco(entrada)
        st.markdown("---")
        selo_risco(proba)
        st.caption(
            "Esta é uma estimativa estatística baseada em padrões históricos da base PEDE — não "
            "substitui a avaliação da equipe pedagógica/psicopedagógica, mas pode ajudar a "
            "priorizar quem acompanhar primeiro."
        )


# =========================================================================
# PÁGINA 4 — CADASTRO DE NOVO ALUNO
# =========================================================================
elif pagina == "📝 Cadastro de Novo Aluno":
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
                "Defasagem": defasagem, "Cat_Defasagem": categoriza_defasagem(defasagem),
                "Cadastrado_em": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            st.session_state["novos_alunos"] = pd.concat(
                [st.session_state["novos_alunos"], pd.DataFrame([novo])], ignore_index=True
            )
            # Persiste em disco (best-effort — funciona localmente e durante a sessão no Cloud)
            try:
                st.session_state["novos_alunos"].to_csv(NOVOS_ALUNOS_PATH, index=False)
            except Exception:
                pass

            st.success(f"Aluno **{nome}** (RA: {ra_final}) cadastrado com sucesso!")

            proba = prever_risco({"IAN": ian, "IDA": ida, "IEG": ieg, "IAA": iaa, "IPS": ips, "IPP": ipp})
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
