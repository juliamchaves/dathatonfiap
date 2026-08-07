
💎 Lapidar — Plataforma de Inteligência Educacional

FIAP Datathon 2026 • Passos Mágicos

O **Lapidar** é uma plataforma de inteligência educacional desenvolvida para o Datathon FIAP 2026, utilizando dados da Associação Passos Mágicos.

O nome nasce da própria jornada que o programa propõe aos seus alunos — da pedra bruta (Quartzo) até a pedra lapidada (Topázio). O projeto transforma dados educacionais em informações estratégicas por meio de tratamento e análise de dados, visualizações interativas e um modelo de Machine Learning, permitindo compreender a trajetória dos estudantes e identificar, com antecedência, oportunidades de intervenção.

---

## 🎯 Missão

Transformar os dados da Pesquisa Extensiva do Desenvolvimento Educacional (PEDE) em conhecimento acionável, apoiando a Passos Mágicos a identificar precocemente alunos em risco de defasagem e a entender quais fatores mais influenciam o desenvolvimento educacional de cada estudante.

## 🔭 Visão

Ser uma ferramenta contínua de apoio à decisão pedagógica — não um relatório estático — para que a cada novo ciclo do PEDE a instituição tenha, em minutos, um raio-x atualizado da adequação de nível, do desempenho e do risco de defasagem de seus alunos.

---

## 📊 Indicadores Educacionais

As análises do Lapidar exploram os principais indicadores disponibilizados pela Associação Passos Mágicos:

| Indicador | Descrição |
|---|---|
| **IAN** | Indicador de Adequação de Nível |
| **IDA** | Indicador de Desempenho Acadêmico |
| **IEG** | Indicador de Engajamento |
| **IAA** | Indicador de Autoavaliação |
| **IPS** | Indicador Psicossocial |
| **IPP** | Indicador Psicopedagógico |
| **IPV** | Indicador de Ponto de Virada |
| **INDE** | Índice do Desenvolvimento Educacional |

---

## 🔎 Análises Desenvolvidas

O painel analítico responde às 11 questões de negócio propostas pelo Datathon:

1. **Defasagem Escolar — IAN**: perfil e evolução da adequação de nível dos alunos (Adequado/Adiantado, Moderadamente defasado, Severamente defasado) entre 2022 e 2024.
2. **Desempenho Acadêmico — IDA**: evolução do desempenho médio ao longo das fases e dos anos.
3. **Engajamento — IEG**: relação entre engajamento, desempenho acadêmico (IDA) e ponto de virada (IPV).
4. **Autoavaliação — IAA**: coerência entre a percepção dos alunos sobre si mesmos e seus resultados reais de desempenho e engajamento.
5. **Aspectos Psicossociais — IPS**: análise longitudinal para identificar se o IPS antecede quedas futuras de desempenho ou engajamento.
6. **Aspectos Psicopedagógicos — IPP**: comparação entre as avaliações psicopedagógicas e os níveis de defasagem identificados pelo IAN.
7. **Ponto de Virada — IPV**: quais indicadores mais se associam ao ponto de virada dos alunos.
8. **Multidimensionalidade — INDE**: quais combinações de indicadores mais elevam a nota global do aluno.
9. **Previsão de Risco — Machine Learning**: modelo preditivo de risco de defasagem (detalhado na seção **Modelo Preditivo** abaixo).
10. **Efetividade do Programa**: progressão do INDE e movimentação de alunos entre as Pedras (Quartzo, Ágata, Ametista, Topázio) ao longo do ciclo.
11. **Insights Adicionais**: gênero, rede de ensino e cruzamento engajamento × desempenho.

---

## 📈 Dashboard

O Lapidar possui um painel interativo desenvolvido com **Streamlit** e **Plotly**, organizado em quatro áreas:

- **🏠 Visão Geral** — KPIs consolidados e evolução do INDE.
- **📊 Painel Analítico** — as 11 análises acima, com gráficos interativos e leitura escrita de cada resultado.
- **🔮 Preditor de Risco** — simulador que estima a probabilidade de um aluno entrar em risco de defasagem no próximo ciclo.
- **📝 Cadastro de Novo Aluno** — formulário para registrar um novo aluno e já obter, na hora, a estimativa de risco calculada pelo modelo.

A identidade visual segue uma paleta própria em tons de azul e laranja, com rótulos numéricos exibidos diretamente nos gráficos.

---

## 🤖 Modelo Preditivo

O modelo estima a probabilidade de um aluno **permanecer ou passar a estar em situação crítica de defasagem** (`Defasagem ≤ -1`) no ciclo seguinte, a partir dos indicadores do ciclo atual.

- **Algoritmo**: `HistGradientBoostingClassifier` (scikit-learn) com **restrição de monotonicidade** — por construção, melhorar qualquer indicador nunca aumenta o risco previsto, evitando comportamento contraintuitivo em perfis extremos pouco representados na base.
- **Variáveis utilizadas**: IAN, IDA, IEG, IAA, IPS.
- **Validação**: temporal *out-of-time* — treino na transição 2022→2023, teste na transição 2023→2024 (nunca mistura dados do "futuro" no treino).
- **Desempenho**: AUC-ROC ≈ **0,71** no conjunto de teste.
- **Checagem de sanidade automática**: a cada retreino, o modelo é testado contra um aluno hipotético com todos os indicadores no valor máximo, que precisa obrigatoriamente ficar com risco baixo (< 20%).

O notebook `02_analise_exploratoria_PEDE.ipynb` documenta a comparação entre Regressão Logística, Random Forest e o modelo final, incluindo curva ROC, importância de variáveis (por permutação) e o ranking de alunos prioritários para intervenção.

---

## 🛠️ Tecnologias Utilizadas

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- OpenPyXL
- Joblib

---

## 📂 Estrutura do Projeto

```
dathatonfiap/
├── app/
│   ├── app.py
│   ├── train_model.py
│   ├── requirements.txt
│   └── modelo_risco_defasagem.pkl
│
├── data/
│   └── raw/
│       ├── BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx
│       └── PEDE_consolidado_longo.csv
│
├── notebooks/
│   ├── 01_limpeza_dados_PEDE.ipynb
│   └── 02_analise_exploratoria_PEDE.ipynb
│
└── README.md
```

---

## ▶️ Como executar o projeto

Clone o repositório
```bash
git clone https://github.com/juliamchaves/dathatonfiap.git
```

Entre no diretório do app
```bash
cd dathatonfiap/app
```

Crie um ambiente virtual

macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

Instale as dependências
```bash
pip install -r requirements.txt
```

Execute o Streamlit
```bash
streamlit run app.py
```

---

## 📦 Dependências

As principais dependências estão definidas no arquivo `requirements.txt`:

```
streamlit>=1.32
pandas>=2.0
numpy>=1.24
plotly>=5.18
scikit-learn==1.8.0
joblib>=1.3
```

---

## 🌐 Aplicação em produção

Deploy realizado no **Streamlit Community Cloud**. Link da aplicação: *[adicionar aqui a URL gerada após o deploy]*.

---

## 👥 Equipe

FIAP Datathon 2026 — Passos Mágicos

- Júlia — RM367721

---

## 💎 Lapidar

Da pedra bruta à pedra lapidada — transformando dados em oportunidade.
