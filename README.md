[README.md](https://github.com/user-attachments/files/30857587/README.md)
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

1. **Defasagem Escolar — IAN**: perfil e evolução da adequação de nível dos alunos entre 2022 e 2024.
2. **Desempenho Acadêmico — IDA**: evolução do desempenho médio ao longo das fases e dos anos.
3. **Engajamento — IEG**: relação entre engajamento, desempenho acadêmico (IDA) e ponto de virada (IPV).
4. **Autoavaliação — IAA**: coerência entre a percepção dos alunos sobre si mesmos e seus resultados reais.
5. **Aspectos Psicossociais — IPS**: se o IPS antecede quedas futuras de desempenho ou engajamento.
6. **Aspectos Psicopedagógicos — IPP**: comparação entre as avaliações psicopedagógicas e os níveis de defasagem (IAN).
7. **Ponto de Virada — IPV**: quais indicadores mais se associam ao ponto de virada.
8. **Multidimensionalidade — INDE**: a fórmula exata de ponderação do INDE (reconstruída e validada empiricamente), em vez de só correlação.
9. **Previsão de Risco — Machine Learning**: modelo preditivo de risco de defasagem (detalhado na seção **Modelo Preditivo** abaixo).
10. **Efetividade do Programa**: progressão do INDE e movimentação de alunos entre as Pedras ao longo do ciclo.
11. **Insights Adicionais**: gênero, rede de ensino e cruzamento engajamento × desempenho.

---

## 📈 Dashboard

Painel interativo em **Streamlit** + **Plotly**, com quatro áreas:

- **🏠 Visão Geral** — KPIs consolidados e evolução do INDE.
- **📊 Painel Analítico** — as 11 análises acima, com gráficos interativos e leitura escrita.
- **🔮 Preditor de Risco** — simulador de probabilidade de risco de defasagem.
- **📝 Cadastro de Novo Aluno** — formulário com estimativa de risco imediata.

Paleta própria em tons de azul e laranja, tema visual customizado (`.streamlit/config.toml`), rótulos numéricos direto nos gráficos.

---

## 🤖 Modelo Preditivo

Estima a probabilidade de um aluno **permanecer ou passar a estar em situação crítica de defasagem** (`Defasagem ≤ -1`) no ciclo seguinte.

- **Algoritmo**: regressão logística com **coeficientes restritos a um único sinal** (`src/modelo_utils.py`) — melhorar qualquer variável nunca aumenta o risco previsto.
- **Feature engineering**: além dos indicadores pedagógicos, usa **Idade** e **Anos na Passos Mágicos** (`Ano − Ano de ingresso`) — variáveis demográficas conhecidas no ano da previsão, sem vazamento de dados.
- **Importância**: IAN (~44%), Idade (~16%), IDA (~15%), IEG (~14%), Anos na Passos Mágicos (~12%). IAA e IPS ficam com peso zero.
- **Validação temporal (principal)**: *out-of-time* — treino 2022→2023, teste 2023→2024. AUC-ROC ≈ **0,78**.
- **Validação cruzada (robustez extra)**: 5-fold estratificada dentro do treino — accuracy 0,71 ± 0,02, precision 0,73 ± 0,02, recall 0,83 ± 0,02, F1 0,78 ± 0,01, ROC AUC 0,76 ± 0,03.
- **Checagens de sanidade automáticas**: aluno exemplar com risco < 30%, nenhuma variável > 70% de importância, AUC ≥ 0,70.

### Fórmula oficial do INDE (pergunta 8)

Reconstruída e validada empiricamente contra a base real (erro médio de ~0,002 ponto):

```
INDE = 0,10×IAN + 0,20×IDA + 0,20×IEG + 0,10×IAA + 0,10×IPS + 0,10×IPP + 0,20×IPV
```

---

## 🛠️ Tecnologias Utilizadas

Python · Streamlit · Pandas · NumPy · Plotly · Scikit-learn · SciPy · OpenPyXL · Joblib

---

## 📂 Estrutura do Projeto

```
dathatonfiap/
├── app/                            # Aplicação Streamlit (só interface)
│   ├── app.py                       # orquestra apenas — sem lógica de página
│   ├── requirements.txt
│   ├── .streamlit/
│   │   └── config.toml               # tema visual customizado
│   ├── utils/                        # funções compartilhadas do app
│   │   ├── tema.py                    # paleta de cores + helpers de gráfico
│   │   ├── dados.py                   # carga e engenharia de dados (com cache)
│   │   ├── modelo.py                  # carga do modelo + predição
│   │   └── componentes.py             # menu lateral e CSS
│   └── paginas/                      # uma página por arquivo
│       ├── visao_geral.py
│       ├── painel_analitico.py         # as 11 perguntas
│       ├── preditor_risco.py
│       └── cadastro_aluno.py
│
├── src/                             # código Python reutilizável (fora do app)
│   ├── limpeza.py                     # funções de limpeza (usadas pelo notebook 01)
│   ├── modelo_utils.py                 # classe do modelo (ModeloMonotonicoLogistico)
│   └── train_model.py                   # script de treino -> gera o .pkl em models/
│
├── data/
│   └── raw/
│       ├── BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx   # base bruta
│       └── PEDE_consolidado_longo.csv                 # base limpa consolidada
│
├── models/
│   └── modelo_risco_defasagem.pkl    # modelo treinado (gerado por src/train_model.py)
│
├── notebooks/
│   ├── 01_limpeza_dados_PEDE.ipynb        # limpeza (chama src/limpeza.py)
│   └── 02_analise_exploratoria_PEDE.ipynb # as 11 perguntas + modelo, em notebook
│
├── reports/                          # entregáveis de apresentação (ver reports/README.md)
│   ├── figuras/
│   └── (apresentação PPT/PDF — pendente)
│
└── README.md
```

**Por que essa organização:** `app/` é só interface — não tem lógica de negócio duplicada. `src/` é o código que qualquer parte do projeto (app, notebooks, scripts futuros) pode reaproveitar, para não haver duas versões da mesma função de limpeza ou do mesmo modelo. `data/`, `models/` e `reports/` separam claramente **dado bruto**, **artefato treinado** e **resultado final para apresentar**.

---

## ▶️ Como executar o projeto

Clone o repositório
```bash
git clone https://github.com/juliamchaves/dathatonfiap.git
cd dathatonfiap
```

Crie um ambiente virtual (na raiz do repositório)

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
pip install -r app/requirements.txt
```

### Rodar o app Streamlit
```bash
cd app
streamlit run app.py
```

### Retreinar o modelo (se a base mudar)
A partir da **raiz do repositório** (não de dentro de `src/`):
```bash
python -m src.train_model
```
> Use exatamente `python -m src.train_model` (com o `-m`), e não `python src/train_model.py`. Rodar como módulo garante que o nome interno do pacote (`src.modelo_utils`) fique consistente entre o treino e o carregamento do modelo no app — rodar do outro jeito quebra o carregamento do `.pkl` com `ModuleNotFoundError`.

### Reprocessar a limpeza dos dados
Abra `notebooks/01_limpeza_dados_PEDE.ipynb` num Jupyter/Colab a partir da pasta `notebooks/` — ele importa as funções de `src/limpeza.py` automaticamente.

---

## 📦 Dependências

Arquivo `app/requirements.txt`:
```
streamlit>=1.32
pandas>=2.0
numpy>=1.24
plotly>=5.18
scipy>=1.11
scikit-learn>=1.3
joblib>=1.3
```

---

## 🌐 Aplicação em produção

Deploy no **Streamlit Community Cloud**. Ao configurar o deploy, o **"Main file path"** deve apontar para `app/app.py` (não para a raiz do repositório).

Link da aplicação: *[adicionar aqui a URL gerada após o deploy]*.

---

## 👥 Equipe

FIAP Datathon 2026 — Passos Mágicos

- Júlia — RM367721

---

## 💎 Lapidar

Da pedra bruta à pedra lapidada — transformando dados em oportunidade.
