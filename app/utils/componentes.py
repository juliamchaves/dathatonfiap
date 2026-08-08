"""Componentes de interface reutilizados entre páginas (menu lateral, estilos)."""
import streamlit as st

from utils.tema import AZUL, AZUL_ESCURO

PAGINAS = ["🏠 Visão Geral", "📊 Painel Analítico", "🔮 Preditor de Risco", "📝 Cadastro de Novo Aluno"]


def injetar_css_menu():
    """Estiliza os botões do menu lateral: ocupam a largura toda, alinhados à
    esquerda, com destaque de cor na página ativa — sem depender de hacks em cima
    de widgets nativos do Streamlit (usa os próprios estados primary/secondary)."""
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] .stButton > button {{
            width: 100%;
            text-align: left;
            justify-content: flex-start;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 0.9rem;
            margin-bottom: 2px;
            font-weight: 400;
            box-shadow: none;
        }}
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            background-color: {AZUL_ESCURO};
            color: white;
            font-weight: 600;
        }}
        [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
            background-color: {AZUL};
            color: white;
        }}
        [data-testid="stSidebar"] .stButton > button[kind="secondary"] {{
            background-color: transparent;
            color: inherit;
        }}
        [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {{
            background-color: rgba(43, 92, 143, 0.12);
            color: inherit;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def renderizar_menu(df):
    """Desenha o menu lateral (título, botões de navegação, filtro de ano) e
    retorna (pagina_selecionada, dataframe_filtrado_por_ano)."""
    injetar_css_menu()

    st.sidebar.title("✨ Passos Mágicos")
    st.sidebar.caption("Datathon — PEDE 2022-2024")

    if "pagina_atual" not in st.session_state:
        st.session_state["pagina_atual"] = PAGINAS[0]

    for opcao in PAGINAS:
        ativa = st.session_state["pagina_atual"] == opcao
        if st.sidebar.button(
            opcao, key=f"nav_{opcao}", use_container_width=True,
            type="primary" if ativa else "secondary",
        ):
            st.session_state["pagina_atual"] = opcao
            st.rerun()

    pagina = st.session_state["pagina_atual"]

    st.sidebar.markdown("---")
    anos_disponiveis = sorted(df["Ano"].unique())
    anos_filtro = st.sidebar.multiselect(
        "Filtrar por ano", anos_disponiveis, default=anos_disponiveis
    )
    df_filtrado = df[df["Ano"].isin(anos_filtro)] if anos_filtro else df

    if len(st.session_state.get("novos_alunos", [])) > 0:
        st.sidebar.markdown("---")
        st.sidebar.caption(
            f"📝 {len(st.session_state['novos_alunos'])} aluno(s) cadastrado(s) nesta base local"
        )

    return pagina, df_filtrado
