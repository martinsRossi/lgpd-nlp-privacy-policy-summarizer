# Makefile para Sumarização de Políticas de Privacidade LGPD

.PHONY: help install setup train run test clean docs all

help:
	@echo "Comandos disponíveis:"
	@echo "  make install    - Instala todas as dependências"
	@echo "  make setup      - Configuração inicial completa"
	@echo "  make train      - Treina o modelo LGPD"
	@echo "  make run        - Inicia a interface Streamlit"
	@echo "  make test       - Executa testes"
	@echo "  make clean      - Remove arquivos temporários"
	@echo "  make docs       - Gera documentação"
	@echo "  make all        - Executa setup completo e inicia aplicação"

install:
	@echo "📦 Instalando dependências..."
	pip install -r requirements.txt
	@echo "✅ Dependências instaladas!"

setup: install
	@echo "🔧 Configuração inicial..."
	python -m nltk.downloader punkt stopwords rslp punkt_tab
	python -m spacy download pt_core_news_sm --quiet || echo "⚠️  Modelo spaCy pt não instalado"
	@echo "📁 Criando diretórios..."
	mkdir -p logs outputs models docs/figuras
	@echo "✅ Setup concluído!"

train:
	@echo "🎓 Treinando modelo LGPD..."
	python -c "from src.treinamento import TreinadorModelo; from src.classificador_lgpd import ClassificadorLGPD; import pandas as pd; from pathlib import Path; from sklearn.model_selection import train_test_split; df = pd.read_csv('data/lgpd_rotulado.csv'); X = df['texto'].tolist(); y = df['categoria'].tolist(); X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42); clf = ClassificadorLGPD(); result = clf.treinar(X_train, y_train); clf.salvar_modelo(Path('models/classificador_lgpd.pkl')); print('✅ Modelo treinado e salvo!')"
	@echo "✅ Treinamento concluído!"

run:
	@echo "🚀 Iniciando interface Streamlit..."
	streamlit run app.py

test:
	@echo "🧪 Executando testes..."
	pytest tests/ -v --cov=src --cov-report=html
	@echo "✅ Testes concluídos! Relatório em htmlcov/index.html"

clean:
	@echo "🧹 Limpando arquivos temporários..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache htmlcov .coverage
	@echo "✅ Limpeza concluída!"

docs:
	@echo "📚 Gerando documentação..."
	@echo "Documentação disponível em docs/"
	@echo "✅ Documentação gerada!"

all: setup train run

# Comandos adicionais para desenvolvimento

dev-install:
	@echo "🔧 Instalando dependências de desenvolvimento..."
	pip install -r requirements.txt
	pip install black flake8 mypy

format:
	@echo "🎨 Formatando código..."
	black src/ app.py
	@echo "✅ Código formatado!"

lint:
	@echo "🔍 Verificando código..."
	flake8 src/ app.py --max-line-length=120
	@echo "✅ Verificação concluída!"

demo:
	@echo "🎬 Executando demonstração..."
	streamlit run app.py --server.headless true

# Comandos para dados

download-data:
	@echo "📥 Baixando dados adicionais..."
	# Adicione scripts para baixar políticas de privacidade reais
	@echo "✅ Dados baixados!"

# Backup e versionamento

backup:
	@echo "💾 Criando backup..."
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz src/ data/ docs/ app.py requirements.txt
	@echo "✅ Backup criado!"
