"""
Instrumentos oficiais de avaliação da Passos Mágicos, usados no Cadastro de Novo Aluno.

Fonte: material metodológico do PEDE fornecido pela Associação Passos Mágicos
("Composição do Índice de Desenvolvimento Educacional" e "Tabela 40 — Questões da
Autoavaliação e seus valores"). Os valores abaixo são citação direta desse material,
não uma reconstrução nossa.

Só implementamos aqui os instrumentos para os quais existe uma fórmula ou tabela de
pontos explícita na documentação (IAA e IDA). Os demais indicadores (IEG, IPS, IPP,
IPV) têm apenas uma definição conceitual no material disponível (ex.: "soma das
pontuações dos avaliadores / número de avaliadores"), sem o questionário específico
por trás — por isso continuam como entrada direta (slider) no formulário, em vez de
um questionário que teríamos que inventar.
"""

# ---------------------------------------------------------------------------
# IAA — Indicador de Autoavaliação
# Fórmula oficial: IAA = soma das pontuações das respostas / número de perguntas
# (6 perguntas, cada uma vale até 10/6 pontos, escala final 0-10)
# ---------------------------------------------------------------------------

PONTOS_MAX_POR_PERGUNTA_IAA = 10 / 6

PERGUNTAS_IAA = {
    "q1": "Como você se sente consigo mesmo?",
    "q2": "Como você se sente sobre os estudos?",
    "q3": "Como você se sente sobre a sua vida familiar?",
    "q4": "Como você se sente sobre sua relação com os amigos?",
    "q5": "Como você se sente sobre a Associação Passos Mágicos?",
    "q6": "Como você se sente sobre seus professores na Passos Mágicos?",
}

# Fases 0 a 2 (alunos mais novos): só 3 opções de resposta
PERCENTUAIS_IAA_INICIAL = {
    "A — melhor avaliação": 1.00,
    "B — avaliação intermediária": 0.70,
    "C — avaliação mais baixa": 0.35,
}

# Fases 3 a 8: 4 opções de resposta
PERCENTUAIS_IAA_AVANCADO = {
    "A — melhor avaliação": 1.00,
    "B": 0.75,
    "C": 0.50,
    "D — avaliação mais baixa": 0.25,
}


def opcoes_iaa(fase_num):
    """Retorna o dicionário de opções de resposta certo para a fase do aluno."""
    if fase_num is not None and fase_num <= 2:
        return PERCENTUAIS_IAA_INICIAL
    return PERCENTUAIS_IAA_AVANCADO


def calcular_iaa(respostas: dict, fase_num) -> float | None:
    """respostas: dict {codigo_pergunta: rotulo_opcao_escolhida}.
    Retorna None se alguma das 6 perguntas não foi respondida (ausente do dict OU
    presente com valor None, que é como o formulário registra "ainda não respondida")."""
    if len(respostas) != len(PERGUNTAS_IAA) or any(v is None for v in respostas.values()):
        return None
    opcoes = opcoes_iaa(fase_num)
    pontos = [PONTOS_MAX_POR_PERGUNTA_IAA * opcoes[resposta] for resposta in respostas.values()]
    return round(sum(pontos), 3)


# ---------------------------------------------------------------------------
# IDA — Indicador de Desempenho Acadêmico
# Fórmula oficial: IDA = (Nota Matemática + Nota Português + Nota Inglês) / 3
# ---------------------------------------------------------------------------

def calcular_ida(nota_matematica, nota_portugues, nota_ingles) -> float | None:
    notas = [nota_matematica, nota_portugues, nota_ingles]
    if any(n is None for n in notas):
        return None
    return round(sum(notas) / 3, 3)


# ---------------------------------------------------------------------------
# INDE — Índice de Desenvolvimento Educacional
# Fórmula oficial (ponderação muda conforme a fase do aluno)
# ---------------------------------------------------------------------------

PESOS_INDE_FASES_0_A_7 = {
    "IAN": 0.10, "IDA": 0.20, "IEG": 0.20, "IAA": 0.10,
    "IPS": 0.10, "IPP": 0.10, "IPV": 0.20,
}
PESOS_INDE_FASE_8 = {
    "IAN": 0.10, "IDA": 0.40, "IEG": 0.20, "IAA": 0.10, "IPS": 0.20,
}


def calcular_inde(fase_num, **indicadores) -> float | None:
    """indicadores: IAN, IDA, IEG, IAA, IPS e, se fase < 8, também IPP e IPV."""
    pesos = PESOS_INDE_FASE_8 if fase_num == 8 else PESOS_INDE_FASES_0_A_7
    valores = [indicadores.get(chave) for chave in pesos]
    if any(v is None for v in valores):
        return None
    return round(sum(indicadores[chave] * peso for chave, peso in pesos.items()), 3)
