"""Página 2 — Painel Analítico: respostas às 11 perguntas do desafio."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.tema import (
    AZUL,
    AZUL_ESCURO,
    LARANJA,
    LARANJA_ESCURO,
    CORES_CAT_DEFASAGEM,
    CORES_GENERO,
    CORES_MOVIMENTO,
    CORES_PEDRA,
    CORES_RESULTADO_IDA,
    ESCALA_DIVERGENTE,
    ORDEM_CAT,
    ORDEM_PEDRA,
    rotula,
)

OPCOES_PERGUNTAS = [
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
]


def _pergunta_1(df):
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
        "e a defasagem severa cai de 3,3% para 0,3%. Uma melhora consistente ao longo dos ciclos."
    )


def _pergunta_2(df):
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
            labels={"color": "Ano"},
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "**Leitura:** o IDA sobe de 6,09 (2022) para 6,66 (2023) e recua para 6,35 em 2024. "
        "As Fases mais avançadas (a partir da Fase 3) tendem a ter IDA mais baixo."
    )


def _pergunta_3(df):
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
        "**Leitura:** correlação positiva e moderada. O engajamento importa para o desempenho, "
        "mas não é o único fator. Outras variáveis devem ser levadas em consideração para uma leitura mais acertiva"
    )


def _pergunta_4(df):
    st.subheader("Autoavaliação (IAA) x Desempenho real (IDA) e Engajamento (IEG)")
    r1 = df["IAA"].corr(df["IDA"])
    r2 = df["IAA"].corr(df["IEG"])
    col1, col2 = st.columns(2)
    col1.metric("Correlação IAA x IDA", f"{r1:.2f}")
    col2.metric("Correlação IAA x IEG", f"{r2:.2f}")

    amostra = df.sample(min(1500, len(df)), random_state=1)
    df_iaa = pd.concat([
        amostra[["IAA", "IDA"]].rename(columns={"IDA": "Valor"}).assign(Comparação="IAA x IDA"),
        amostra[["IAA", "IEG"]].rename(columns={"IEG": "Valor"}).assign(Comparação="IAA x IEG"),
    ], ignore_index=True)
    fig = px.scatter(
        df_iaa, x="IAA", y="Valor", color="Comparação", opacity=0.4,
        color_discrete_map={"IAA x IDA": AZUL, "IAA x IEG": LARANJA},
        labels={"Valor": "IDA / IEG"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "**Leitura:** correlação relativamente fraca. A percepção do aluno sobre si "
        "mesmo não é um bom preditor nem do seu desempenho real (pontos azuis), nem do seu "
        "engajamento observado (pontos laranja)."
    )


def _pergunta_5(df):
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
        "desempenho acadêmico (r=0,12) O que nos sugere que o IPS pode funcionar como indicador "
        "antecedente de risco, principalmente ligado a vínculo e participação escolar."
    )


def _pergunta_6(df):
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
        "defasado. O IPP sugere uma tendência de confirmação da defasagem apontada pelo IAN "
    )


def _pergunta_7(df):
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
        "virada — acompanhamento psicopedagógico + desempenho + engajamento. Como IPV é uma "
        "síntese longitudinal aqui a correlação é a "
        "ferramenta correta de análise"
    )


def _pergunta_8(df, modelo_pack):
    st.subheader("Quais indicadores mais explicam o INDE?")
    pesos_inde = modelo_pack.get("pesos_inde")
    if pesos_inde:
        st.markdown(
           
        )
        formula_txt = " + ".join(f"{v:.2f}×{k}" for k, v in pesos_inde.items())
        st.code(f"INDE = {formula_txt}", language=None)
        cols_necessarias = list(pesos_inde.keys()) + ["INDE"]
        df_verif = df.dropna(subset=cols_necessarias).copy()
        df_verif["INDE_calculado"] = sum(
            df_verif[col] * peso for col, peso in pesos_inde.items()
        )
        erro_medio = (df_verif["INDE_calculado"] - df_verif["INDE"]).abs().mean()
        st.caption(
            f"Fórmula validada em {len(df_verif):,} registros — erro médio absoluto de "
            f"{erro_medio:.3f} ponto (numa escala de 0 a 10).".replace(",", ".")
        )
        st.markdown(
            "**Leitura:** o peso na fórmula é fixo, mas não diz o quanto cada indicador "
            "*de fato* influencia o resultado final. Um indicador pode ter peso alto e "
            "variar pouco entre os alunos, contribuindo pouco para diferenciar o INDE na "
            "prática, enquanto outro com peso menor pode ser o que mais acompanha (ou "
            "puxa) a nota. A correlação abaixo mostra exatamente isso: qual indicador, ao "
            "subir ou cair, mais se move junto com o INDE observado na base real."
        )
        st.markdown("---")
    st.subheader("Correlação observada (para referência)")
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
    st.caption(
        "Atenção: como o INDE é uma combinação linear dos indicadores acima, essas "
        "correlações são parcialmente esperadas por construção, a ordem relativa entre elas "
        "ainda é informativa."
    )


def _pergunta_9(df, modelo, modelo_pack, features, auc_teste):
    st.subheader("Modelo preditivo de risco de defasagem")
    st.markdown(
        """
        **Metodologia:** validação temporal *out-of-time* O modelo é treinado com a
        transição **2022 → 2023** e testado na transição **2023 → 2024**, simulando o uso
        real (prever o futuro a partir de dados do passado, sem misturar os dois períodos).
        O alvo é: o aluno **permanece (ou passa a estar) em situação crítica** no ano
        seguinte (`Defasagem ≤ -1`).

        O modelo é uma **regressão logística com restrição de monotonicidade**: cada
        coeficiente é obrigado a ter o mesmo sinal (indicador maior ⇒ risco menor ou igual,
        nunca maior). Além dos indicadores pedagógicos, o modelo usa duas variáveis
        demográficas, **Idade** e **Anos na Passos Mágicos**,  que carregam sinal preditivo
        genuinamente independente da defasagem atual, sem introduzir vazamento de dados (são
        informações conhecidas no próprio ano em que a previsão é feita).
        """
    )

    if auc_teste:
        st.metric("AUC — validação out-of-time (2023→2024)", f"{auc_teste:.3f}")

    cv_metricas = modelo_pack.get("cv_metricas")
    if cv_metricas:
        st.markdown("**Validação cruzada (5-fold, dentro do treino 2022→2023)** — checagem extra de estabilidade:")
        nomes_exibicao = {
            "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
            "f1": "F1 Score", "roc_auc": "ROC AUC",
        }
        cols_cv = st.columns(len(cv_metricas))
        for col, (chave, valores) in zip(cols_cv, cv_metricas.items()):
            col.metric(
                nomes_exibicao.get(chave, chave),
                f"{valores['media']:.3f}",
                f"± {valores['desvio']:.3f}",
                delta_color="off",
            )

    importancias_pct = pd.Series(modelo_pack.get("importancias", {})).sort_values()
    fig = px.bar(
        x=importancias_pct.values, y=importancias_pct.index, orientation="h",
        labels={"x": "Importância relativa (%)", "y": "Variável"},
        title="Quais indicadores mais pesam na previsão de risco?",
        text=importancias_pct.values, color_discrete_sequence=[AZUL],
    )
    rotula(fig, sufixo="%")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "**Leitura:** o IAN ainda é o maior peso individual (~44%) O que faz sentido, já "
        "que ele reflete diretamente a defasagem atual do aluno, mas "
        " **Idade** (~16%), **IDA** (~15%), **IEG** (~14%) e **Anos na Passos "
        "Mágicos** (~12%) somados já pesam mais que o IAN isoladamente. IAA e IPS seguem com "
        "peso zero: sob a restrição de monotonicidade, a correlação deles com o risco futuro "
        "não teve o sinal esperado (achado consistente com as perguntas 4 e 5, onde ambos "
        "também mostraram correlação fraca com os resultados reais). É uma ferramenta "
        "estatisticamente válida para apoiar a priorização de intervenção preventiva => vá até "
        "**🔮 Preditor de Risco** para simular um aluno."
    )


def _pergunta_10(df):
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
        "consolida sua posição nas faixas superiores. Evidência de evolução acadêmica "
        "sustentável."
    )


def _pergunta_11(df):
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
           médio mais alto que os de escola pública. Vale investigar se é efeito do suporte
           extra ou viés de seleção dos indicados a essas vagas.
        3. **Fases intermediárias como ponto de atenção:** o IDA cai nas fases mais avançadas
           (pergunta 2) O reforço extra a partir da Fase 3 pode valer a pena.
        4. **Sinal de risco antecipável:** o modelo da pergunta 9 mostra que dá para sinalizar,
           com boa antecedência, quais alunos correm risco de aumentar sua defasagem, abrindo
           espaço para intervenção preventiva antes que o problema se agrave.
        """
    )


def render(df, modelo, modelo_pack, features, auc_teste):
    st.title("Painel Analítico — Perguntas 1 a 11")

    aba = st.selectbox("Escolha a pergunta", OPCOES_PERGUNTAS)
    st.markdown("---")

    if aba.startswith("1."):
        _pergunta_1(df)
    elif aba.startswith("2."):
        _pergunta_2(df)
    elif aba.startswith("3."):
        _pergunta_3(df)
    elif aba.startswith("4."):
        _pergunta_4(df)
    elif aba.startswith("5."):
        _pergunta_5(df)
    elif aba.startswith("6."):
        _pergunta_6(df)
    elif aba.startswith("7."):
        _pergunta_7(df)
    elif aba.startswith("8."):
        _pergunta_8(df, modelo_pack)
    elif aba.startswith("9."):
        _pergunta_9(df, modelo, modelo_pack, features, auc_teste)
    elif aba.startswith("10."):
        _pergunta_10(df)
    elif aba.startswith("11."):
        _pergunta_11(df)
