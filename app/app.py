"""
Datathon Passos Mágicos — Aplicação Streamlit
Painel analítico (perguntas 1 a 11) + preditor de risco de defasagem + cadastro de novos alunos.

Estrutura do projeto (ver README.md na raiz do repositório para o passo a passo completo):

    app/                           -> este diretório: só o app Streamlit
        app.py                      -> este arquivo: só orquestra, não tem lógica de página
        .streamlit/config.toml       -> tema visual (cores, fonte)
        utils/                        -> funções compartilhadas do app (dados, modelo, tema, componentes)
        paginas/                      -> uma página do app por arquivo
    src/                            -> código Python reutilizável (fora do app)
        limpeza.py                    -> funções de limpeza da base (usadas pelo notebook 01)
        modelo_utils.py                -> classe do modelo (ModeloMonotonicoLogistico)
        train_model.py                  -> script que treina e gera o .pkl
    data/raw/                        -> base bruta e base limpa consolidada
    models/                          -> modelo treinado (modelo_risco_defasagem.pkl)
    notebooks/                       -> notebooks de limpeza e análise exploratória
    reports/                         -> apresentação de storytelling e figuras exportadas

Para rodar localmente (a partir da raiz do repositório):
    cd app
    streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

# Adiciona a raiz do repositório ao sys.path, para que "from src... import ..." funcione
# não importa de onde o Streamlit for iniciado (Community Cloud, local, etc.).
RAIZ_PROJETO = Path(__file__).resolve().parent.parent
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from paginas import cadastro_aluno, painel_analitico, preditor_risco, visao_geral
from utils.componentes import renderizar_menu
from utils.dados import carregar_dados, inicializar_sessao
from utils.modelo import carregar_modelo
from utils.tema import aplicar_tema_plotly

# =========================================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================================
st.set_page_config(
    page_title="Passos Mágicos | Painel PEDE",
    page_icon="✨",
    layout="wide",
)
aplicar_tema_plotly()

# =========================================================================
# CARGA DE DADOS E MODELO (com cache — só recarrega se os arquivos mudarem)
# =========================================================================
df = carregar_dados()
modelo_pack = carregar_modelo()
modelo = modelo_pack["model"]
FEATURES = modelo_pack["features"]
AUC_TESTE = modelo_pack.get("auc_teste")
MEDIAS_TREINO = modelo_pack.get("medias_treino", {})

inicializar_sessao()

# =========================================================================
# NAVEGAÇÃO (menu lateral) — devolve a página escolhida + dados filtrados por ano
# =========================================================================
pagina, df_filtrado = renderizar_menu(df)

# =========================================================================
# ROTEAMENTO — cada página vive no seu próprio módulo em paginas/
# =========================================================================
if pagina == "🏠 Visão Geral":
    visao_geral.render(df, df_filtrado)

elif pagina == "📊 Painel Analítico":
    painel_analitico.render(df, modelo, modelo_pack, FEATURES, AUC_TESTE)

elif pagina == "🔮 Preditor de Risco":
    preditor_risco.render(modelo, modelo_pack, FEATURES, MEDIAS_TREINO, AUC_TESTE)

elif pagina == "📝 Cadastro de Novo Aluno":
    cadastro_aluno.render(df, modelo, modelo_pack, FEATURES, MEDIAS_TREINO)
