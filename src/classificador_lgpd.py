"""
Módulo de Classificação de Trechos conforme LGPD
Classifica segmentos de políticas de privacidade em categorias da Lei Geral de Proteção de Dados

⚠️ METODOLOGIA DE TREINAMENTO GLOBAL E INCREMENTAL:

Este classificador é um MODELO GLOBAL que aprende padrões linguísticos de conformidade LGPD
através de TODAS as políticas coletadas de múltiplas empresas.

CARACTERÍSTICAS:
- ✅ Treinamento GLOBAL: Um único modelo para todas as empresas
- ✅ Dataset UNIFICADO: data/lgpd_rotulado_global.csv agrega exemplos de todas as políticas
- ✅ Aprendizado INCREMENTAL: Cada nova política contribui exemplos → retreinamento → nova versão
- ✅ Versionamento: models/classificador_lgpd_v1.0.pkl, v1.1.pkl, v2.0.pkl, etc.
- ✅ Generalização: Aprende padrões que funcionam para qualquer empresa

WORKFLOW:
1. Nova política → Rotulação manual → Adicionar ao dataset global
2. Quando threshold é atingido (+50 exemplos) → Retreinar modelo
3. Salvar nova versão com metadados (acurácia, num_exemplos, empresas)
4. Modelo v1.1 substitui v1.0 em produção

NÃO CONFUNDIR COM:
- Gold Standard LGPD: Benchmark fixo de avaliação (não treina)
- Modelo de Conformidade: Modelo separado que prevê score (treina separadamente)

Para metodologia completa, veja: docs/METODOLOGIA_CORRIGIDA.md
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path


# Categorias da LGPD
CATEGORIAS_LGPD = {
    'coleta_dados': 'Coleta de Dados Pessoais',
    'finalidade': 'Finalidade do Tratamento',
    'compartilhamento': 'Compartilhamento com Terceiros',
    'armazenamento': 'Armazenamento e Retenção',
    'seguranca': 'Medidas de Segurança',
    'direitos_titular': 'Direitos do Titular',
    'cookies': 'Cookies e Rastreamento',
    'internacional': 'Transferência Internacional',
    'criancas': 'Dados de Crianças e Adolescentes',
    'contato': 'Informações de Contato/DPO',
    'alteracoes': 'Alterações na Política',
    'outros': 'Outros/Geral'
}

# Palavras-chave para cada categoria (para classificação baseada em regras)
PALAVRAS_CHAVE_LGPD = {
    'coleta_dados': [
        'coletamos', 'coleta', 'dados pessoais', 'informações pessoais', 
        'cadastro', 'registro', 'fornecidos', 'obtemos', 'solicitamos'
    ],
    'finalidade': [
        'finalidade', 'propósito', 'objetivo', 'utilizamos para', 
        'usamos para', 'processamos para', 'tratamos para'
    ],
    'compartilhamento': [
        'compartilhamos', 'compartilhar', 'terceiros', 'parceiros',
        'fornecedores', 'divulgação', 'revelamos'
    ],
    'armazenamento': [
        'armazenamos', 'armazenamento', 'retenção', 'mantemos',
        'conservamos', 'prazo', 'período de guarda'
    ],
    'seguranca': [
        'segurança', 'proteção', 'medidas técnicas', 'criptografia',
        'protegemos', 'salvaguardas', 'confidencialidade'
    ],
    'direitos_titular': [
        'direitos', 'acesso', 'retificação', 'exclusão', 'portabilidade',
        'oposição', 'revogação', 'consentimento', 'solicitar'
    ],
    'cookies': [
        'cookies', 'rastreamento', 'pixels', 'web beacon',
        'tecnologias similares', 'navegação'
    ],
    'internacional': [
        'transferência internacional', 'outros países', 'exterior',
        'fora do brasil', 'adequação'
    ],
    'criancas': [
        'crianças', 'menores', 'adolescentes', 'idade',
        'responsável legal', 'pais', 'tutores'
    ],
    'contato': [
        'contato', 'dpo', 'encarregado', 'privacidade@',
        'fale conosco', 'ouvidoria'
    ],
    'alteracoes': [
        'alterações', 'modificações', 'atualização', 'revisão',
        'mudanças', 'atualizamos'
    ]
}


class ClassificadorLGPD:
    """Classificador de trechos de políticas de privacidade conforme LGPD"""
    
    def __init__(self, modelo_tipo: str = 'logistic'):
        """
        Inicializa o classificador
        
        Args:
            modelo_tipo: Tipo do modelo ('naive_bayes', 'logistic', 'random_forest')
        """
        self.modelo_tipo = modelo_tipo
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.label_encoder = LabelEncoder()
        self.modelo = None
        self.treinado = False
        
        logger.info(f"Inicializando ClassificadorLGPD (modelo: {modelo_tipo})")
        
        # Inicializar modelo
        if modelo_tipo == 'naive_bayes':
            self.modelo = MultinomialNB()
        elif modelo_tipo == 'logistic':
            self.modelo = LogisticRegression(max_iter=1000, random_state=42)
        elif modelo_tipo == 'random_forest':
            self.modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Tipo de modelo não suportado: {modelo_tipo}")
    
    def classificar_por_regras(self, texto: str) -> str:
        """
        Classificação baseada em palavras-chave (método de fallback)
        
        Args:
            texto: Texto a ser classificado
            
        Returns:
            Categoria LGPD identificada
        """
        texto_lower = texto.lower()
        pontuacoes = {}
        
        for categoria, palavras in PALAVRAS_CHAVE_LGPD.items():
            pontuacao = sum(1 for palavra in palavras if palavra in texto_lower)
            pontuacoes[categoria] = pontuacao
        
        categoria_max = max(pontuacoes, key=pontuacoes.get)
        
        # Se nenhuma palavra-chave foi encontrada, retorna 'outros'
        if pontuacoes[categoria_max] == 0:
            return 'outros'
        
        return categoria_max
    
    def treinar(
        self,
        textos: List[str],
        categorias: List[str],
        test_size: float = 0.2,
        validacao: bool = True
    ) -> Dict:
        """
        Treina o modelo de classificação
        
        Args:
            textos: Lista de textos de treinamento
            categorias: Lista de categorias correspondentes
            test_size: Proporção do conjunto de teste
            validacao: Se deve separar conjunto de validação
            
        Returns:
            Dicionário com métricas de treinamento
        """
        logger.info(f"Iniciando treinamento com {len(textos)} exemplos")
        
        # Encode das labels
        y = self.label_encoder.fit_transform(categorias)
        
        # Vetorização
        X = self.vectorizer.fit_transform(textos)
        
        # Split treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Treinamento
        logger.info("Treinando modelo...")
        self.modelo.fit(X_train, y_train)
        
        # Avaliação
        score_train = self.modelo.score(X_train, y_train)
        score_test = self.modelo.score(X_test, y_test)
        
        self.treinado = True
        
        resultado = {
            'acuracia_treino': score_train,
            'acuracia_teste': score_test,
            'num_exemplos': len(textos),
            'num_categorias': len(self.label_encoder.classes_),
            'categorias': list(self.label_encoder.classes_)
        }
        
        logger.success(f"Treinamento concluído - Acurácia teste: {score_test:.3f}")
        return resultado
    
    def classificar(self, texto: str, usar_regras: bool = False) -> Dict:
        """
        Classifica um texto em uma categoria LGPD
        
        Args:
            texto: Texto a ser classificado
            usar_regras: Se deve usar classificação por regras ao invés do modelo
            
        Returns:
            Dicionário com categoria e confiança
        """
        if usar_regras or not self.treinado:
            categoria = self.classificar_por_regras(texto)
            return {
                'categoria': categoria,
                'categoria_nome': CATEGORIAS_LGPD.get(categoria, 'Outros'),
                'confianca': 0.5,  # Confiança padrão para regras
                'metodo': 'regras'
            }
        
        # Classificação com modelo treinado
        X = self.vectorizer.transform([texto])
        pred = self.modelo.predict(X)[0]
        categoria = self.label_encoder.inverse_transform([pred])[0]
        
        # Obter probabilidade/confiança
        if hasattr(self.modelo, 'predict_proba'):
            proba = self.modelo.predict_proba(X)[0]
            confianca = float(proba.max())
        else:
            confianca = 0.8  # Confiança padrão se não tiver predict_proba
        
        return {
            'categoria': categoria,
            'categoria_nome': CATEGORIAS_LGPD.get(categoria, 'Outros'),
            'confianca': confianca,
            'metodo': 'modelo'
        }
    
    def classificar_lote(self, textos: List[str]) -> List[Dict]:
        """
        Classifica múltiplos textos de uma vez
        
        Args:
            textos: Lista de textos
            
        Returns:
            Lista de dicionários com classificações
        """
        return [self.classificar(texto) for texto in textos]
    
    def classificar_sentencas(self, sentencas: List[str]) -> pd.DataFrame:
        """
        Classifica cada sentença e retorna DataFrame estruturado
        
        Args:
            sentencas: Lista de sentenças
            
        Returns:
            DataFrame com sentenças classificadas
        """
        resultados = []
        
        for i, sentenca in enumerate(sentencas):
            classificacao = self.classificar(sentenca)
            resultados.append({
                'id': i,
                'sentenca': sentenca,
                'categoria': classificacao['categoria'],
                'categoria_nome': classificacao['categoria_nome'],
                'confianca': classificacao['confianca']
            })
        
        df = pd.DataFrame(resultados)
        logger.info(f"Classificadas {len(df)} sentenças")
        return df
    
    def salvar_modelo(self, caminho: Path):
        """
        Salva o modelo treinado em disco
        
        Args:
            caminho: Caminho para salvar o modelo
        """
        if not self.treinado:
            logger.warning("Modelo não foi treinado ainda")
            return
        
        modelo_data = {
            'modelo': self.modelo,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'modelo_tipo': self.modelo_tipo
        }
        
        joblib.dump(modelo_data, caminho)
        logger.success(f"Modelo salvo em {caminho}")
    
    def carregar_modelo(self, caminho: Path):
        """
        Carrega um modelo salvo
        
        Args:
            caminho: Caminho do modelo salvo
        """
        modelo_data = joblib.load(caminho)
        
        self.modelo = modelo_data['modelo']
        self.vectorizer = modelo_data['vectorizer']
        self.label_encoder = modelo_data['label_encoder']
        self.modelo_tipo = modelo_data['modelo_tipo']
        self.treinado = True
        
        logger.success(f"Modelo carregado de {caminho}")
    
    def obter_features_importantes(self, top_n: int = 10) -> Dict:
        """
        Retorna as features mais importantes para cada categoria
        
        Args:
            top_n: Número de features por categoria
            
        Returns:
            Dicionário com features importantes
        """
        if not self.treinado:
            logger.warning("Modelo não treinado")
            return {}
        
        if not hasattr(self.modelo, 'feature_importances_') and not hasattr(self.modelo, 'coef_'):
            logger.warning("Modelo não suporta análise de features")
            return {}
        
        feature_names = self.vectorizer.get_feature_names_out()
        resultado = {}
        
        if hasattr(self.modelo, 'coef_'):  # LogisticRegression
            for i, categoria in enumerate(self.label_encoder.classes_):
                coef = self.modelo.coef_[i]
                top_indices = np.argsort(coef)[-top_n:][::-1]
                top_features = [(feature_names[idx], coef[idx]) for idx in top_indices]
                resultado[categoria] = top_features
        
        return resultado
