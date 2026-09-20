# Cranfield Information Retrieval

Trabalho Prático 1 da disciplina SCC0282 - Recuperação de Informação (2º Sem/2026, ICMC-USP).
Sistema de recuperação textual sobre a coleção Cranfield, implementando e comparando o
Modelo Vetorial (TF-IDF + similaridade de cosseno) e o Modelo Probabilístico (BM25), com
avaliação quantitativa (Precision@10, Recall@10, MAP, F1@10, MRR, NDCG@10), variação de
parâmetros do BM25, modificação de consultas e análise de erros.

## Base de dados

Coleção clássica **Cranfield** (1400 documentos, 225 consultas, julgamentos de relevância),
obtida via [ir_datasets](https://ir-datasets.com/cranfield.html) (`ir_datasets.load("cranfield")`).

Caso o download pelo `ir_datasets` falhe, o projeto usa como fallback os arquivos originais do
[Glasgow IR Group](http://ir.dcs.gla.ac.uk/resources/test_collections/cran/), já incluídos em
`data/raw/` (`cran.all.1400`, `cran.qry`, `cranqrel`). Essa lógica está implementada em
`src/cranfield_data.py`, que é usada por todos os notebooks para carregar documentos, consultas
e qrels.

## Instalação e execução

1. Python 3.11+ (testado com Python 3.13, via Anaconda).
2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Na primeira execução, o NLTK baixa automaticamente os recursos necessários (tokenizador
   `punkt` e lista de `stopwords`) através de `ensure_nltk_resources()` em
   `src/pre_processing.py`.

4. Execute os notebooks em `notebooks/`, na ordem, com Jupyter (`jupyter lab` ou `jupyter
   notebook`), ou via linha de comando:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploratory_analysis.ipynb
   ```

   Ordem de execução (cada notebook depende de artefatos gerados pelo anterior, salvos em
   `data/processed/`):

   | Notebook | Conteúdo |
   |---|---|
   | `01_exploratory_analysis.ipynb` | Análise exploratória da coleção e pré-processamento (Caso 1: 4 configurações) |
   | `02_modelo_vetorial.ipynb` | Modelo Vetorial (Caso 2) |
   | `03_bm25.ipynb` | Modelo Probabilístico BM25 (Caso 3) |
   | `04_avaliacao.ipynb` | Avaliação quantitativa (Caso 4) — gera os rankings/métricas usados pelos notebooks seguintes |
   | `05_comparacao_modelos.ipynb` | Comparação entre modelos (Caso 5) |
   | `06_analise_por_consulta.ipynb` | Análise por consulta (Caso 6) |
   | `07_variacao_parametros_bm25.ipynb` | Variação dos parâmetros do BM25 (Caso 7) |
   | `08_modificacao_consultas.ipynb` | Modificação de consultas (Caso 8) |
   | `09_analise_erros.ipynb` | Análise de erros (Caso 9) |

## Linguagem e principais bibliotecas

- **Python** 3.11+
- `nltk` — tokenização, stopwords, stemming (Porter)
- `pandas` / `numpy` — manipulação de dados
- `scikit-learn` — `TfidfVectorizer` e `cosine_similarity` (ponderação de termos e similaridade
  no Modelo Vetorial)
- `scipy` — estruturas de matriz esparsa
- `matplotlib` — visualizações
- `ir_datasets` — carregamento da coleção Cranfield

O BM25 (`src/bm25.py` / `src/probabilistic_model.py`) é implementado explicitamente
(índice invertido, IDF e fórmula do score calculados manualmente), sem uso de bibliotecas
prontas de BM25.

## Estrutura do projeto

```
src/
  cranfield_data.py       # carregamento da coleção (ir_datasets + fallback local)
  pre_processing.py       # tokenização, stopwords, stemming, 4 configurações
  vectorial_model.py      # funções de base do Modelo Vetorial (TF-IDF + cosseno)
  vector_model.py         # classe VectorSpaceModel usada pelos notebooks
  probabilistic_model.py  # funções de base do BM25 (índice invertido, score)
  bm25.py                 # classe BM25 usada pelos notebooks
  evaluation.py           # métricas de avaliação (precision, recall, AP, NDCG, ...)
  metrics.py              # evaluate_model(), usada pelos notebooks
notebooks/                # Casos 1-9 do enunciado
data/
  raw/                    # arquivos originais do Cranfield (fallback)
  processed/              # dados pré-processados e rankings/métricas persistidos
  Trabalho_Pratico_1_RI_2026.pdf  # enunciado do trabalho
```
