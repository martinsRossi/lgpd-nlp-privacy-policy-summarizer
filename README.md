# Sumarização Automática de Políticas de Privacidade com Processamento de Linguagem Natural

**Trabalho de Conclusão de Curso**  
Faculdade Anchieta - Curso de Ciência da Computação  
Ano: 2025

---

## Resumo

Este projeto apresenta um sistema automatizado para sumarização e simplificação de políticas de privacidade, utilizando técnicas avançadas de Processamento de Linguagem Natural (PLN) e aprendizado de máquina. O sistema classifica automaticamente trechos de políticas em categorias baseadas na Lei Geral de Proteção de Dados (LGPD), gera sumários extrativos e abstrativos, e simplifica termos técnicos para aumentar a compreensibilidade por usuários leigos.

**Palavras-chave:** Processamento de Linguagem Natural, LGPD, Sumarização Automática, Machine Learning, Políticas de Privacidade

---

## Objetivos

### Objetivo Geral
Desenvolver um sistema automatizado capaz de processar políticas de privacidade, classificá-las segundo categorias da LGPD, e gerar sumários simplificados que facilitem a compreensão de usuários não especialistas.

### Objetivos Específicos
- Implementar técnicas de sumarização extrativa (TextRank, LexRank, LSA) e abstrativa (modelos Transformer)
- Desenvolver classificador supervisionado para categorização automática de trechos conforme LGPD
- Criar módulo de simplificação lexical para substituição de termos técnicos e jurídicos
- Avaliar a qualidade dos sumários através de métricas objetivas (ROUGE, BLEU)
- Disponibilizar interface web interativa para uso do sistema

---

## Funcionalidades

### 1. Ingestão de Documentos
O sistema suporta múltiplos formatos de entrada:
- Arquivos de texto (.txt)
- Documentos PDF
- URLs de políticas online
- Entrada direta de texto

### 2. Pré-processamento Textual
- Limpeza de HTML, URLs e caracteres especiais
- Tokenização de sentenças e palavras
- Remoção de stopwords (opcional)
- Stemming para redução morfológica
- Suporte para português e inglês

### 3. Classificação LGPD
Sistema de classificação automática baseado em 12 categorias alinhadas à LGPD:
- Coleta de Dados
- Finalidade do Tratamento
- Compartilhamento com Terceiros
- Armazenamento e Retenção
- Segurança da Informação
- Direitos do Titular
- Cookies e Tecnologias de Rastreamento
- Transferência Internacional
- Dados de Crianças e Adolescentes
- Informações de Contato (DPO)
- Alterações na Política
- Outros

**Modelo:** Regressão Logística com vetorização TF-IDF  
**Acurácia obtida:** 85.4% no conjunto de teste

### 4. Sumarização Extrativa
Extração das sentenças mais relevantes do documento original:
- **Algoritmos implementados:** TextRank, LexRank, LSA
- **Taxa de compressão configurável** (padrão: 30%)
- Preservação da ordem original das sentenças
- Identificação de sentenças-chave por relevância estatística

### 5. Sumarização Abstrativa
Geração de texto novo através de modelos de linguagem:
- **Modelos disponíveis:** T5-small, PTT5 (português), GPT-2
- Geração semântica com controle de comprimento máximo
- Beam search para otimização de qualidade
- Parafraseamento inteligente mantendo o sentido original

### 6. Simplificação Lexical
- Substituição automática de mais de 40 termos técnicos e jurídicos
- Geração de glossário explicativo
- Redução de complexidade sintática
- Cálculo de índice de legibilidade

### 7. Avaliação de Qualidade
Métricas automáticas para avaliação dos sumários:
- **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation): ROUGE-1, ROUGE-2, ROUGE-L
- **BLEU** (Bilingual Evaluation Understudy): BLEU-1 até BLEU-4
- Métricas de classificação: Acurácia, Precisão, Recall, F1-Score

### 8. Interface Web
Aplicação web desenvolvida em Streamlit com:
- Navegação intuitiva em etapas
- Visualizações gráficas de resultados
- Exportação completa de artefatos
- Barra de progresso e feedback em tempo real

---

## Arquitetura do Sistema

O sistema foi desenvolvido seguindo uma arquitetura modular com separação clara de responsabilidades:

```
┌─────────────────┐
│  Entrada de     │
│  Documentos     │
│  (TXT/PDF/URL)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Pré-processamento│
│  - Tokenização  │
│  - Limpeza      │
│  - Normalização │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Classificação  │
│      LGPD       │
│  (ML Supervisionado)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Sumarização    │
│  - Extrativa    │
│  - Abstrativa   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Simplificação  │
│  - Termos       │
│  - Glossário    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Avaliação      │
│  (ROUGE/BLEU)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Interface Web  │
│   (Streamlit)   │
└─────────────────┘
```

---

## Resultados Experimentais

### Desempenho do Classificador LGPD

O modelo de classificação foi treinado com um dataset rotulado contendo 199 exemplos de trechos de políticas de privacidade de múltiplas empresas. Os resultados obtidos demonstram a eficácia da abordagem:

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| Acurácia Geral | 85.4% | Percentual de classificações corretas |
| Precisão (Weighted) | 83.4% | Confiabilidade das predições positivas |
| Recall (Weighted) | 85.4% | Capacidade de identificar todas as instâncias |
| F1-Score (Weighted) | 85.3% | Média harmônica entre precisão e recall |

### Desempenho por Categoria

| Categoria | Precisão | Recall | F1-Score | Suporte |
|-----------|----------|--------|----------|---------|
| Alterações | 86.7% | 86.7% | 86.7% | 15 |
| Armazenamento | 100.0% | 56.3% | 72.0% | 16 |
| Coleta de Dados | 100.0% | 63.2% | 77.4% | 19 |
| Compartilhamento | 80.0% | 88.9% | 84.2% | 18 |
| Contato (DPO) | 92.3% | 75.0% | 82.8% | 16 |
| Cookies | 92.3% | 80.0% | 85.7% | 15 |
| Crianças | 86.4% | 95.0% | 90.5% | 20 |
| Direitos do Titular | 72.4% | 91.3% | 80.8% | 23 |
| Finalidade | 44.4% | 95.2% | 60.6% | 21 |
| Transferência Internacional | 91.7% | 73.3% | 81.5% | 15 |
| Segurança | 71.4% | 38.5% | 50.0% | 13 |
| Outros | 100.0% | 50.0% | 66.7% | 12 |

**Observações:**
- 9 das 12 categorias alcançaram F1-Score superior a 75%
- As categorias com menor desempenho (Finalidade, Segurança, Outros) apresentam maior ambiguidade semântica
- O modelo demonstrou boa capacidade de generalização considerando o tamanho limitado do dataset

### Convergência do Treinamento

O treinamento foi realizado com validação cruzada estratificada (5-fold) ao longo de 50 épocas:

- **Loss de treinamento final:** 0.131
- **Loss de validação final:** 0.231
- **Acurácia de treinamento final:** 86.9%
- **Acurácia de validação final:** 76.9%
- **Convergência alcançada:** Sim (época 10)

As curvas de aprendizado indicam convergência adequada sem sinais significativos de overfitting.

---

## Instalação e Configuração

### Requisitos do Sistema

- Python 3.12 ou superior
- pip (gerenciador de pacotes Python)
- 4GB RAM mínimo (8GB recomendado)
- Conexão com internet (para download de modelos pré-treinados)

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/tcc-lgpd-sumarizacao.git
cd tcc-lgpd-sumarizacao
```

2. Crie um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure os recursos do NLTK e spaCy:
```bash
python -m nltk.downloader punkt stopwords rslp
python -m spacy download pt_core_news_sm
```

### Execução

Para iniciar a aplicação web:
```bash
streamlit run app.py
```

A aplicação estará disponível em: `http://localhost:8501`

### Treinamento do Modelo

Para retreinar o classificador LGPD:
```bash
python -m scripts.treinar_classificador_global
```

---

## Estrutura do Projeto

```
TCC_NEW_def_ultimato_final/
│
├── app.py                          # Aplicação web principal (Streamlit)
├── requirements.txt                # Dependências do projeto
├── config.yaml                     # Configurações do sistema
├── Makefile                        # Automação de tarefas
├── README.md                       # Documentação principal
│
├── src/                            # Código-fonte modular
│   ├── __init__.py
│   ├── ingestao.py                 # Ingestão de documentos
│   ├── preprocessamento.py         # Pré-processamento de texto
│   ├── classificador_lgpd.py       # Classificação por categorias LGPD
│   ├── sumarizacao_extrativa.py    # Sumarização extrativa
│   ├── sumarizacao_abstrativa.py   # Sumarização abstrativa
│   ├── simplificacao.py            # Simplificação lexical
│   ├── avaliacao.py                # Métricas de avaliação
│   ├── treinamento.py              # Treinamento de modelos
│   ├── analisador_consumidor.py    # Análise orientada ao consumidor
│   ├── modelo_conformidade_lgpd.py # Modelo de conformidade
│   └── gold_standard_lgpd.py       # Gold standard LGPD
│
├── data/                           # Dados e datasets
│   ├── lgpd_rotulado_global.csv    # Dataset de treinamento (dataset incremental)
│   ├── lgpd_rotulado.csv
│   └── gold_standard.txt           # Políticas de referência
│
├── scripts/                        # Scripts auxiliares
│   ├── treinar_classificador_global.py
│   ├── validacao_multiplas_empresas.py
│   └── gerar_graficos_treinamento.py
│
├── results/                        # Resultados de experimentos
│   ├── relatorio_metricas.txt
│   └── *.png                      # Gráficos gerados
│
└── tests/                          # Testes unitários
    ├── test_classificador.py
    └── test_imei_fix.py
```

---

## Guia de Uso

### 1. Carregamento de Documento

Na interface web, acesse a seção **"Carregar Texto"** e escolha uma das opções:
- **Upload de arquivo:** Selecione um arquivo .txt ou .pdf
- **URL:** Insira o link da política de privacidade
- **Texto direto:** Cole o conteúdo manualmente

### 2. Pré-processamento

Acesse **"Pré-processamento"** e configure:
- Remoção de stopwords (palavras comuns)
- Aplicação de stemming (redução morfológica)
- Nível de limpeza desejado

Clique em "Processar Texto" para aplicar as transformações.

### 3. Classificação LGPD

Em **"Classificação LGPD"**, clique em "Classificar Trechos" para:
- Segmentar o texto em sentenças
- Classificar cada sentença em uma das 12 categorias LGPD
- Visualizar a distribuição das categorias

### 4. Geração de Sumário

Na seção **"Sumarização"**:
- Escolha o método (Extrativo ou Abstrativo)
- Configure a taxa de redução desejada
- Selecione o algoritmo específico (TextRank, T5, etc.)
- Gere o sumário

### 5. Simplificação

Acesse **"Simplificação"** para:
- Substituir termos técnicos por linguagem simples
- Gerar glossário explicativo
- Visualizar o índice de legibilidade

### 6. Avaliação (Opcional)

Se possuir um sumário de referência:
- Acesse **"Avaliação"**
- Forneça o texto de referência
- Visualize métricas ROUGE e BLEU

### 7. Exportação

Em **"Exportar"**, gere um pacote ZIP contendo:
- Sumários gerados
- Texto simplificado
- Métricas e gráficos
- Relatório completo

---

## Tecnologias Utilizadas

### Processamento de Linguagem Natural
- **NLTK 3.8+** - Natural Language Toolkit para tokenização, stopwords e stemming
- **spaCy 3.7+** - Biblioteca avançada para análise morfossintática e NER
- **Transformers 4.35+** (HuggingFace) - Modelos pré-treinados T5, PTT5 e GPT-2
- **Sumy 0.11+** - Implementações de algoritmos de sumarização extrativa

### Aprendizado de Máquina
- **scikit-learn 1.3+** - Algoritmos de classificação e vetorização TF-IDF
- **PyTorch 2.1+** - Framework de deep learning para modelos Transformer
- **NumPy 1.24+** - Computação numérica e álgebra linear
- **Pandas 2.1+** - Manipulação e análise de dados estruturados

### Avaliação e Métricas
- **rouge-score 0.1+** - Cálculo de métricas ROUGE
- **evaluate 0.4+** - Framework de avaliação de modelos

### Visualização
- **Matplotlib 3.8+** - Geração de gráficos estáticos
- **Seaborn 0.13+** - Visualizações estatísticas
- **Plotly 5.18+** - Gráficos interativos

### Interface e Utilitários
- **Streamlit 1.28+** - Framework para aplicações web
- **Loguru 0.7+** - Sistema de logging estruturado
- **PyPDF2 3.0+** - Extração de texto de documentos PDF
- **BeautifulSoup4 4.12+** - Parsing e limpeza de HTML

---

## Referências

1. BRASIL. **Lei nº 13.709, de 14 de agosto de 2018.** Lei Geral de Proteção de Dados Pessoais (LGPD). Diário Oficial da União, Brasília, DF, 15 ago. 2018.

2. MIHALCEA, R.; TARAU, P. **TextRank: Bringing Order into Texts.** In: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE PROCESSING (EMNLP), 2004, Barcelona. Proceedings... Barcelona: ACL, 2004. p. 404-411.

3. RAFFEL, C. et al. **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.** Journal of Machine Learning Research, v. 21, n. 140, p. 1-67, 2020.

4. LIN, C.-Y. **ROUGE: A Package for Automatic Evaluation of Summaries.** In: TEXT SUMMARIZATION BRANCHES OUT WORKSHOP, 2004, Barcelona. Proceedings... Barcelona: ACL, 2004. p. 74-81.

5. PAPINENI, K. et al. **BLEU: a Method for Automatic Evaluation of Machine Translation.** In: ANNUAL MEETING OF THE ASSOCIATION FOR COMPUTATIONAL LINGUISTICS, 40., 2002, Philadelphia. Proceedings... Philadelphia: ACL, 2002. p. 311-318.

6. BIRD, S.; KLEIN, E.; LOPER, E. **Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit.** Sebastopol: O'Reilly Media, 2009.

7. HONNIBAL, M.; MONTANI, I. **spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.** 2017. Disponível em: https://spacy.io. Acesso em: 2025.

8. WOLF, T. et al. **Transformers: State-of-the-Art Natural Language Processing.** In: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE PROCESSING (EMNLP), 2020. Proceedings... Online: ACL, 2020. p. 38-45.

9. PEDREGOSA, F. et al. **Scikit-learn: Machine Learning in Python.** Journal of Machine Learning Research, v. 12, p. 2825-2830, 2011.

10. AUTORIDADE NACIONAL DE PROTEÇÃO DE DADOS (ANPD). **Guia Orientativo para Definições dos Agentes de Tratamento de Dados Pessoais e do Encarregado.** Brasília: ANPD, 2021.

---

## Autor

**Trabalho de Conclusão de Curso**  
Faculdade Anchieta  
Curso: Ciência da Computação  
Ano: 2025

---

## Licença

Este projeto está licenciado sob a Licença MIT - consulte o arquivo `LICENSE` para detalhes.

---

## Observações Importantes

Este sistema foi desenvolvido com propósito educacional e de pesquisa acadêmica. As sumarizações geradas são automatizadas e devem ser utilizadas como material complementar, não substituindo a leitura integral das políticas de privacidade originais. Para questões jurídicas específicas, recomenda-se consultar profissionais especializados em direito digital e proteção de dados.
