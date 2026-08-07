[README.md](https://github.com/user-attachments/files/30847392/README.md)
# dathatonfiap# Painel PEDE — Passos Mágicos (Streamlit)

App com **painel analítico** (respostas às 11 perguntas do desafio, alinhadas ao notebook de
análise corrigido), um **preditor de risco de defasagem** (validação temporal out-of-time) e uma
página de **cadastro de novos alunos**.

## Arquivos deste projeto

```
app.py                          -> aplicação principal (é isso que você "roda")
train_model.py                  -> script que treina e salva o modelo (já rodado uma vez p/ você)
modelo_risco_defasagem.pkl      -> modelo treinado (Random Forest, AUC ≈ 0,715, validação out-of-time)
PEDE_consolidado_longo.csv      -> base de dados limpa, usada pelo app
requirements.txt                -> lista de bibliotecas que o app precisa
novos_alunos_cadastrados.csv    -> gerado automaticamente quando alguém usa a página de cadastro
```

Os arquivos-base **precisam estar todos na mesma pasta** (e depois, no mesmo repositório do
GitHub) para o app funcionar, porque `app.py` os carrega por caminho relativo.

## Sobre o modelo preditivo



**Correção:** o modelo agora é um `HistGradientBoostingClassifier` com **restrição de
monotonicidade** (`monotonic_cst`): por construção, aumentar qualquer indicador nunca eleva o
risco previsto — não é mais algo que o modelo "aprende sozinho" e pode errar em casos raros, é uma
regra matemática garantida. O `train_model.py` roda automaticamente uma checagem de sanidade (um
perfil com todos os indicadores em 10 precisa ficar com risco abaixo de 20%) e já viria travando o
treino no futuro se essa premissa for quebrada.

- **Validação temporal out-of-time**: treina com a transição 2022→2023, testa com 2023→2024.
- **Alvo**: o aluno permanece (ou passa a estar) em situação crítica (`Defasagem ≤ -1`) no ano
  seguinte.
- **Variáveis**: IAN, IDA, IEG, IAA, IPS.
- **Desempenho**: AUC ≈ 0,71 na validação out-of-time.
- **Checagem**: aluno com todos os indicadores em 10 agora fica em ~8% de risco (antes: 35,7%).

## Sobre o cadastro de novos alunos

A página **📝 Cadastro de Novo Aluno** permite registrar um aluno novo (identificação + os 10
indicadores) direto pela interface. Ao salvar:
- o registro entra numa tabela local da aplicação (visível na mesma página);
- a probabilidade de risco de defasagem já é calculada e mostrada na hora;
- um botão permite baixar todos os cadastros feitos em `.csv`.

> ⚠️ **Importante:** no plano gratuito do Community Cloud, o disco do app é temporário — os
> cadastros ficam lá enquanto o app está ativo, mas podem se perder se o serviço reiniciar. Baixe
> o CSV periodicamente se quiser guardar os cadastros de forma permanente, ou (para algo mais
> robusto) troque a gravação em CSV por um banco de dados externo (ex.: Google Sheets, Supabase,
> etc.) — fico à disposição se quiser evoluir para isso depois.

---

## Passo 1 — Rodar localmente no seu computador (recomendado antes do deploy)

1. Tenha o Python instalado (3.10+). Verifique no terminal:
   ```bash
   python3 --version
   ```
2. Abra um terminal na pasta onde estão esses arquivos e instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Rode o app:
   ```bash
   streamlit run app.py
   ```
4. Vai abrir automaticamente no navegador em algo como `http://localhost:8501`. Se não abrir
   sozinho, copie esse endereço que aparece no terminal e cole no navegador.
5. Para parar, `Ctrl + C` no terminal.

> Não tem Python "de sobra" na sua máquina, ou não quer instalar nada? Dá pra fazer esse teste
> local também pelo **GitHub Codespaces** (um VS Code no navegador, gratuito até um certo limite
> de horas) — abra o repositório no GitHub, clique em **Code → Codespaces → Create codespace**, e
> rode os mesmos comandos acima no terminal que aparece.

---

## Passo 2 — Subir os arquivos para o GitHub

Você já tem um repositório (`juliamchaves/dathatonfiap`, usado para hospedar o CSV). Duas formas
de colocar os arquivos lá:

**Opção A — pelo site do GitHub (mais simples, sem usar terminal):**
1. Entre no repositório no navegador.
2. Crie uma pasta para o app (ex.: `streamlit_app/`) usando **Add file → Create new file** e
   digitando `streamlit_app/app.py` como nome (o GitHub cria a pasta sozinho).
3. Cole o conteúdo de cada arquivo (`app.py`, `requirements.txt`, `train_model.py`) e clique em
   **Commit changes**.
4. Para os arquivos binários/grandes (`modelo_risco_defasagem.pkl`, `PEDE_consolidado_longo.csv`),
   use **Add file → Upload files** e arraste-os para dentro da mesma pasta `streamlit_app/`.

**Opção B — pelo terminal (git):**
```bash
git clone https://github.com/juliamchaves/dathatonfiap.git
cd dathatonfiap
mkdir streamlit_app
# copie app.py, requirements.txt, train_model.py, modelo_risco_defasagem.pkl e
# PEDE_consolidado_longo.csv para dentro de streamlit_app/
git add streamlit_app
git commit -m "Adiciona app Streamlit do Datathon"
git push
```

---

## Passo 3 — Deploy no Streamlit Community Cloud

1. Acesse **https://share.streamlit.io** e faça login com sua conta do **GitHub**.
2. Clique em **Create app** (ou **New app**).
3. Escolha **"Deploy a public app from GitHub"**.
4. Preencha:
   - **Repository**: `juliamchaves/dathatonfiap`
   - **Branch**: `main` (ou a branch onde você deu commit)
   - **Main file path**: `streamlit_app/app.py` (o caminho até o `app.py`, dentro do repo)
5. Clique em **Deploy**.
6. Aguarde alguns minutos — ele vai instalar as dependências do `requirements.txt` e iniciar o
   app. Você vai ver os logs de build na tela.
7. Quando terminar, você recebe uma URL pública tipo:
   `https://dathatonfiap-streamlit-app.streamlit.app` — esse é o link que você entrega no
   Datathon.

### Se der erro no deploy

- **`ModuleNotFoundError: No module named '_loss'`** (ou qualquer erro parecido ao carregar o
  `.pkl`): isso é **incompatibilidade de versão do scikit-learn** — o modelo foi salvo com a
  versão `1.8.0`, e o ambiente tem uma versão diferente instalada (geralmente porque já havia uma
  versão em cache antes de você adicionar/editar o `requirements.txt`). Para corrigir:
  1. Confirme que `requirements.txt` tem exatamente a linha `scikit-learn==1.8.0`.
  2. **No Community Cloud**: abra o menu **⋮** (canto superior direito do app) → **Reboot app**
     (ou **Manage app → Reboot**) para forçar a reinstalação das dependências do zero.
  3. **No GitHub Codespaces / máquina local**: rode
     `pip install -r requirements.txt --force-reinstall --no-cache-dir`.
  4. Se ainda persistir, rode `python train_model.py` na sua própria máquina/ambiente (com o
     scikit-learn já na versão correta) para gerar um `.pkl` novo, compatível com o ambiente onde
     ele vai rodar, e suba esse arquivo atualizado para o repositório.
- **"ModuleNotFoundError" para outras bibliotecas**: falta alguma linha no `requirements.txt` —
  adicione e o app refaz o deploy automaticamente a cada novo `git push`.
- **"FileNotFoundError: PEDE_consolidado_longo.csv"** (ou o `.pkl`): confirme que esses arquivos
  estão de fato dentro da mesma pasta do `app.py` no repositório (não só na sua máquina).
- **App "dormiu"**: no plano gratuito, apps sem uso por um tempo entram em modo inativo. Basta
  abrir o link de novo e clicar em "Yes, wake this app up" — volta ao ar em segundos.
- Para forçar reprocessamento depois de trocar dados/modelo, use o menu **⋮ → Rerun** ou
  **Clear cache** no canto superior direito do app.

---

## Atualizando o modelo no futuro

Se a base de dados for atualizada (ex.: PEDE 2025), rode de novo:
```bash
python train_model.py
```
Isso gera um novo `modelo_risco_defasagem.pkl`. Suba o arquivo atualizado para o GitHub
(commit + push) e o Streamlit Community Cloud atualiza o app sozinho.
