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

ALI, Asjad. **Understanding the NLP Pipeline: A Comprehensive Guide.** Medium, 1 jan. 2024. Disponível em: https://medium.com/@asjad_ali/understanding-the-nlp-pipeline-a-comprehensive-guide-828b2b3cd4e2. Acesso em: 2 out. 2025.

BIRD, Steven et al. **Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit.** [S. l.: s. n.], 2009. Disponível em: https://tjzhifei.github.io/resources/NLTK.pdf. Acesso em: 2 out. 2025.

BRASIL. **Lei nº 13.709, de 14 de agosto de 2018.** Dispõe sobre a proteção de dados pessoais e altera a Lei nº 12.965, de 23 de abril de 2014 (Marco Civil da Internet). Lei Geral de Proteção de Dados Pessoais (LGPD), Brasília, DF: Diário Oficial da União, 15 ago. 2018.

BROWN, Tom B. et al. **Language Models are Few-Shot Learners.** ArXiv, 22 jul. 2020. Disponível em: https://arxiv.org/abs/2005.14165. Acesso em: 1 out. 2025.

CAMACHO-COLLADOS, Jose. **Embeddings in Natural Language Processing.** ACL Anthology, 2020. Disponível em: https://aclanthology.org/2020.coling-tutorials.2.pdf. Acesso em: 1 out. 2025.

CAMBRIA, Erik et al. **Jumping NLP Curves: A Review of Natural Language Processing Research.** IEEE Computational Intelligence Magazine, 2014. Disponível em: https://sentic.net/jumping-nlp-curves.pdf. Acesso em: 2 out. 2025.

CHOMSKY, Noam. **Aspects of the Theory of Syntax.** [S. l.: s. n.], 1965. Disponível em: https://www.colinphillips.net/wp-content/uploads/2015/09/chomsky1965-ch1.pdf. Acesso em: 2 out. 2025.

DENG, Li et al. **Recent Trends in Deep Learning Based Natural Language Processing.** ArXiv, 9 ago. 2017. Disponível em: https://arxiv.org/pdf/1708.02709. Acesso em: 2 out. 2025.

DEVLIN, Jacob. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.** ArXiv, 24 maio 2020. Disponível em: https://arxiv.org/abs/1810.04805. Acesso em: 1 out. 2025.

DOSHI, Ketan. **Transformers Explained Visually (Part 1): Overview of Functionality.** Towards Data Science, 13 dez. 2020. Disponível em: https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452/. Acesso em: 30 set. 2025.

ERKAN, Gunes et al. **LexRank: Graph-based Lexical Centrality as Salience in Text Summarization.** Journal of Artificial Intelligence Research, 1 dez. 2004. Disponível em: https://www.jair.org/index.php/jair/article/view/10396/. Acesso em: 11 nov. 2025.

FLORIDI, Luciano et al. **GPT‑3: Its Nature, Scope, Limits, and Consequences.** Springer Nature Link, 1 nov. 2020. Disponível em: https://link.springer.com/article/10.1007/s11023-020-09548-1. Acesso em: 10 nov. 2025.

FOOTE, Keith D. **A Brief History of Natural Language Processing.** Dataversity, 6 jul. 2023. Disponível em: https://www.dataversity.net/articles/a-brief-history-of-natural-language-processing-nlp/. Acesso em: 2 out. 2025.

GAMBHIR, Mahak. **Recent automatic text summarization techniques: a survey.** Università degli Studi di Milano-Bicocca, 29 mar. 2016. Disponível em: https://elearning.unimib.it/pluginfile.php/1593406/mod_resource/content/1/Gambhir-Gupta2017_Article_RecentAutomaticTextSummarizati.pdf. Acesso em: 1 out. 2025.

GONG, Yihong et al. **Generic text summarization using relevance measure and latent semantic analysis.** ACM Digital Library, 1 set. 2001. Disponível em: https://dl.acm.org/doi/10.1145/383952.383955. Acesso em: 11 nov. 2025.

GOODFELLOW, Ian et al. **Deep Learning: An MIT Press book.** [S. l.], 2016. Disponível em: https://www.deeplearningbook.org/. Acesso em: 1 out. 2025.

JURAFSKY, Dan et al. **Speech and Language Processing (3rd ed. draft).** [S. l.: s. n.], 2025. Disponível em: https://web.stanford.edu/~jurafsky/slp3/. Acesso em: 2 out. 2025.

KHATTER, Kiran et al. **Natural language processing: state of the art, current trends and challenges.** ResearchGate, 2022. Disponível em: https://www.researchgate.net/publication/319164243_Natural_language_processing_state_of_the_art_current_trends_and_challenges. Acesso em: 2 out. 2025.

LIN, Chin-Yew. **ROUGE: A Package for Automatic Evaluation of Summaries.** ACL Anthology, 2004. Disponível em: https://aclanthology.org/W04-1013/. Acesso em: 2 out. 2025.

LIU, Yang. **Fine-tune BERT for Extractive Summarization.** ArXiv, 5 set. 2019. Disponível em: https://arxiv.org/abs/1903.10318. Acesso em: 1 out. 2025.

MANNING, Christopher D. et al. **Foundations of Statistical Natural Language Processing.** The MIT Press, 1999. Disponível em: http://icog-labs.com/wp-content/uploads/2014/07/Christopher_D._Manning_Hinrich_Schütze_Foundations_Of_Statistical_Natural_Language_Processing.pdf. Acesso em: 2 out. 2025.

MIHALCEA, Rada et al. **TextRank: Bringing Order into Texts.** ACL Anthology, 2004. Disponível em: https://aclanthology.org/W04-3252/. Acesso em: 11 nov. 2025.

NASCIMENTO, Francileuza; BARROS, Milena; PINTO, Anderson. **A Proteção de Dados Pessoais e a LGPD no Brasil: Desafios e Perspectivas.** Revista Ibero-Americana de Humanidades, Ciências e Educação — REASE, São Paulo, v. 10, n. 12, 2024. Disponível em: https://periodicorease.pro.br/rease/article/view/17628. Acesso em: 2 set. 2025.

OBAR, Jonathan A.; OELDORF-HIRSCH, Anne. **The Biggest Lie on the Internet: Ignoring the Privacy Policies and Terms of Service Policies of Social Networking Services.** Information, Communication & Society, TPRC 44: The 44th Research Conference on Communication, Information and Internet Policy, 1 jun. 2018, p. 1–20. Disponível em: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2757465. Acesso em: 2 set. 2025.

**Privacy and Freedom.** Washington and Lee Law Review, vol. 25, n. 1, 1968. Disponível em: https://scholarlycommons.law.wlu.edu/wlulr/vol25/iss1/20/. Acesso em: 2 set. 2025.

RADEV, Dragomir R. et al. **Centroid-based summarization of multiple documents.** Information Processing & Management (IPM), 2004. DOI: 10.1016/j.ipm.2003.10.006. Disponível em: https://scispace.com/pdf/centroid-based-summarization-of-multiple-documents-1gu2p7ixap.pdf. Acesso em: 11 nov. 2025.

RADEV, Dragomir R. et al. **Centroid-based summarization of multiple documents: sentence extraction, utility-based evaluation, and user studies.** ACL Anthology, 2000. Disponível em: https://aclanthology.org/W00-0403/. Acesso em: 2 out. 2025.

RADFORD, Alec et al. **Improving Language Understanding by Generative Pre-Training.** OpenAI, [20--?]. Disponível em: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf. Acesso em: 1 out. 2025.

RAFFEL, Colin et al. **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.** ArXiv, 19 set. 2023. Disponível em: https://arxiv.org/abs/1910.10683. Acesso em: 1 out. 2025.

RIBEIRO PEREIRA, Felipe. **Utilização de Processamento de Linguagem Natural e Ontologias na Análise Qualitativa de Frases Curtas.** Novas Tecnologias na Educação, 3 dez. 2013. Disponível em: https://seer.ufrgs.br/index.php/renote/article/view/44431/28444. Acesso em: 2 out. 2025.

SEE, Abigail et al. **Get To The Point: Summarization with Pointer-Generator Networks.** ACL Anthology, 2017. Disponível em: https://aclanthology.org/P17-1099/. Acesso em: 2 out. 2025.

STRYKER, Cole et al. **What is NLP (natural language processing)?** IBM, [20--?]. Disponível em: https://www.ibm.com/think/topics/natural-language-processing. Acesso em: 2 out. 2025.

**The Cost of Reading Privacy Policies.** I/S: A Journal of Law and Policy for the Information Society, 2008. Disponível em: https://www.is-journal.org/. Acesso em: 2 set. 2025.

**The Right to Privacy.** Harvard Law Review, Vol. IV, 15 dez. 1890. Disponível em: https://groups.csail.mit.edu/mac/classes/6.805/articles/privacy/Privacy_brand_warr2.html. Acesso em: 2 set. 2025.

VASWANI, Ashish. **Attention Is All You Need.** ArXiv, 2 ago. 2023. Disponível em: https://arxiv.org/abs/1706.03762. Acesso em: 1 out. 2025.

**What is Natural Language Processing (NLP)?** AWS, [20--]. Disponível em: https://aws.amazon.com/what-is/nlp/. Acesso em: 2 out. 2025.

WOLF, Thomas. **Transformers: State-of-the-Art Natural Language Processing.** ArXiv, 14 jul. 2020. Disponível em: https://arxiv.org/abs/1910.03771. Acesso em: 1 out. 2025.

ZANATTA, Rafael A. F. **A Proteção Coletiva dos Dados Pessoais no Brasil: vetores de interpretação.** [S. l.: s. n.], 2023. Disponível em: https://www.dataprivacybr.org/wp-content/uploads/2025/05/2023.-ZANATTA-Rafael.-A-protecao-coletiva-dos-dados-pessoais-no-Brasil.pdf. Acesso em: 1 out. 2025.

ZHANG, Jingqing. **PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization.** ArXiv, 10 jul. 2020. Disponível em: https://arxiv.org/pdf/1912.08777. Acesso em: 1 out. 2025.

---

## Autores

**Trabalho de Conclusão de Curso**  
Faculdade Anchieta  
Curso: Ciência da Computação  
Ano: 2025

- **2204029 – Gabriel Cunha de Araujo**
- **2205385 – Kelvin Pimenta Dias**
- **2304805 – Renan Martins Rossi**

---

## Licença

Este projeto está licenciado sob a Licença MIT - consulte o arquivo `LICENSE` para detalhes.

---

## Observações Importantes

Este sistema foi desenvolvido com propósito educacional e de pesquisa acadêmica. As sumarizações geradas são automatizadas e devem ser utilizadas como material complementar, não substituindo a leitura integral das políticas de privacidade originais. Para questões jurídicas específicas, recomenda-se consultar profissionais especializados em direito digital e proteção de dados.
