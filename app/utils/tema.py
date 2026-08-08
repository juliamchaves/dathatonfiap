"""Paleta de cores e constantes visuais compartilhadas por todo o app."""
import plotly.express as px

# ---- Paleta de cores do app: tons de azul e laranja ----
AZUL_ESCURO = "#1F4E8C"
AZUL = "#2B5C8F"
AZUL_CLARO = "#6FA8DC"
LARANJA_ESCURO = "#C9660B"
LARANJA = "#E67E22"
LARANJA_CLARO = "#F2B46D"

PALETA_SEQ = [AZUL, LARANJA, AZUL_CLARO, LARANJA_CLARO, AZUL_ESCURO, LARANJA_ESCURO]
ESCALA_DIVERGENTE = [[0.0, LARANJA_ESCURO], [0.5, "#FFFFFF"], [1.0, AZUL_ESCURO]]

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

ORDEM_CAT = ["Severamente defasado", "Moderadamente defasado", "Adequado/Adiantado"]
ORDEM_PEDRA = ["Quartzo", "Ágata", "Ametista", "Topázio"]

# O IAN só assume estes 3 valores reais na base (vem da categoria de defasagem) —
# por isso é tratado como um seletor, e não como um slider contínuo.
OPCOES_IAN = [
    (10.0, "10 — Adequado/Adiantado (sem defasagem)"),
    (5.0, "5 — Moderadamente defasado"),
    (2.5, "2,5 — Severamente defasado"),
]


def aplicar_tema_plotly():
    """Define a paleta padrão do Plotly Express para qualquer gráfico do app que
    não especifique cor explicitamente. Chamar uma vez, no início do app.py."""
    px.defaults.color_discrete_sequence = PALETA_SEQ


def rotula(fig, sufixo="", casas=2, posicao="outside"):
    """Mostra o valor de cada ponto/barra direto no gráfico (sem precisar passar o
    mouse), arredondado para `casas` decimais, com um sufixo opcional (ex.: '%')."""
    fig.update_traces(texttemplate=f"%{{text:.{casas}f}}{sufixo}", textposition=posicao)
    return fig
