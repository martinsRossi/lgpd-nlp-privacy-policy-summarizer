"""
Módulo de Sumarização Extrativa
Implementa técnicas de sumarização extrativa usando TextRank
"""

from typing import List, Dict, Optional
from loguru import logger
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
import networkx as nx


class SumarizadorExtrativo:
    """Classe para sumarização extrativa de políticas de privacidade"""
    
    def __init__(self, idioma: str = 'portuguese'):
        """
        Inicializa o sumarizador extrativo
        
        Args:
            idioma: Idioma do texto (portuguese, english)
        """
        self.idioma = idioma
        logger.info(f"Inicializando SumarizadorExtrativo (idioma: {idioma})")
    
    def textrank_manual(
        self,
        sentencas: List[str],
        taxa_reducao: float = 0.3,
        usar_tfidf: bool = True
    ) -> List[str]:
        """
        Implementação manual do TextRank
        
        Args:
            sentencas: Lista de sentenças
            taxa_reducao: Porcentagem de sentenças a manter (0.3 = 30%)
            usar_tfidf: Se deve usar TF-IDF ao invés de bag-of-words
            
        Returns:
            Lista de sentenças selecionadas
        """
        if len(sentencas) <= 2:
            return sentencas
        
        # Vetorização
        if usar_tfidf:
            vectorizer = TfidfVectorizer()
        else:
            vectorizer = CountVectorizer()
        
        try:
            X = vectorizer.fit_transform(sentencas)
        except Exception as e:
            logger.error(f"Erro na vetorização: {e}")
            return sentencas[:max(1, int(len(sentencas) * taxa_reducao))]
        
        # Calcular similaridade
        similarity_matrix = cosine_similarity(X)
        
        # Criar grafo
        nx_graph = nx.from_numpy_array(similarity_matrix)
        
        # Calcular scores com PageRank
        try:
            scores = nx.pagerank(nx_graph, max_iter=100)
        except:
            # Fallback: usar média de similaridade
            scores = {i: similarity_matrix[i].sum() for i in range(len(sentencas))}
        
        # Ordenar por score
        ranked_sentencas = sorted(
            [(scores[i], i, sent) for i, sent in enumerate(sentencas)],
            reverse=True
        )
        
        # Selecionar top sentenças
        num_sentencas = max(1, int(len(sentencas) * taxa_reducao))
        top_sentencas = ranked_sentencas[:num_sentencas]
        
        # Reordenar pela posição original
        top_sentencas_ordenadas = sorted(top_sentencas, key=lambda x: x[1])
        resultado = [sent for _, _, sent in top_sentencas_ordenadas]
        
        logger.info(f"TextRank: {len(sentencas)} → {len(resultado)} sentenças")
        return resultado
    
    def textrank_sumy(
        self,
        texto: str,
        num_sentencas: Optional[int] = None,
        taxa_reducao: float = 0.3
    ) -> str:
        """
        TextRank usando biblioteca Sumy
        
        Args:
            texto: Texto completo
            num_sentencas: Número de sentenças no sumário (se None, usa taxa_reducao)
            taxa_reducao: Porcentagem de redução
            
        Returns:
            Texto sumarizado
        """
        parser = PlaintextParser.from_string(texto, Tokenizer(self.idioma))
        summarizer = TextRankSummarizer()
        
        if num_sentencas is None:
            total_sentencas = len(list(parser.document.sentences))
            num_sentencas = max(1, int(total_sentencas * taxa_reducao))
        
        summary = summarizer(parser.document, num_sentencas)
        resultado = ' '.join(str(sentence) for sentence in summary)
        
        logger.info(f"TextRank (Sumy): gerado sumário com {num_sentencas} sentenças")
        return resultado
    
    def lexrank(
        self,
        texto: str,
        num_sentencas: Optional[int] = None,
        taxa_reducao: float = 0.3
    ) -> str:
        """
        LexRank usando biblioteca Sumy
        
        Args:
            texto: Texto completo
            num_sentencas: Número de sentenças no sumário
            taxa_reducao: Porcentagem de redução
            
        Returns:
            Texto sumarizado
        """
        parser = PlaintextParser.from_string(texto, Tokenizer(self.idioma))
        summarizer = LexRankSummarizer()
        
        if num_sentencas is None:
            total_sentencas = len(list(parser.document.sentences))
            num_sentencas = max(1, int(total_sentencas * taxa_reducao))
        
        summary = summarizer(parser.document, num_sentencas)
        resultado = ' '.join(str(sentence) for sentence in summary)
        
        logger.info(f"LexRank: gerado sumário com {num_sentencas} sentenças")
        return resultado
    
    def lsa(
        self,
        texto: str,
        num_sentencas: Optional[int] = None,
        taxa_reducao: float = 0.3
    ) -> str:
        """
        LSA (Latent Semantic Analysis) usando Sumy
        
        Args:
            texto: Texto completo
            num_sentencas: Número de sentenças no sumário
            taxa_reducao: Porcentagem de redução
            
        Returns:
            Texto sumarizado
        """
        parser = PlaintextParser.from_string(texto, Tokenizer(self.idioma))
        summarizer = LsaSummarizer()
        
        if num_sentencas is None:
            total_sentencas = len(list(parser.document.sentences))
            num_sentencas = max(1, int(total_sentencas * taxa_reducao))
        
        summary = summarizer(parser.document, num_sentencas)
        resultado = ' '.join(str(sentence) for sentence in summary)
        
        logger.info(f"LSA: gerado sumário com {num_sentencas} sentenças")
        return resultado
    
    def sumarizar(
        self,
        texto: str,
        metodo: str = 'textrank',
        taxa_reducao: float = 0.3,
        num_sentencas: Optional[int] = None
    ) -> Dict:
        """
        Método principal de sumarização extrativa
        
        Args:
            texto: Texto completo
            metodo: Método de sumarização ('textrank', 'lexrank', 'lsa')
            taxa_reducao: Porcentagem de redução (0.3 = mantém 30%)
            num_sentencas: Número específico de sentenças (override taxa_reducao)
            
        Returns:
            Dicionário com sumário e metadados
        """
        logger.info(f"Iniciando sumarização extrativa (método: {metodo})")
        
        # Contar sentenças originais
        from nltk.tokenize import sent_tokenize
        sentencas_originais = sent_tokenize(texto, language='portuguese' if self.idioma == 'portuguese' else 'english')
        num_original = len(sentencas_originais)
        
        # Aplicar método escolhido
        if metodo == 'textrank':
            sumario = self.textrank_sumy(texto, num_sentencas, taxa_reducao)
        elif metodo == 'lexrank':
            sumario = self.lexrank(texto, num_sentencas, taxa_reducao)
        elif metodo == 'lsa':
            sumario = self.lsa(texto, num_sentencas, taxa_reducao)
        else:
            logger.warning(f"Método desconhecido: {metodo}. Usando TextRank.")
            sumario = self.textrank_sumy(texto, num_sentencas, taxa_reducao)
        
        # Contar sentenças do sumário
        sentencas_sumario = sent_tokenize(sumario, language='portuguese' if self.idioma == 'portuguese' else 'english')
        num_sumario = len(sentencas_sumario)
        
        # Calcular taxa de compressão
        taxa_compressao = len(sumario) / len(texto) if len(texto) > 0 else 0
        
        resultado = {
            'sumario': sumario,
            'texto_original': texto,
            'metodo': metodo,
            'num_sentencas_original': num_original,
            'num_sentencas_sumario': num_sumario,
            'taxa_reducao_sentencas': num_sumario / num_original if num_original > 0 else 0,
            'taxa_compressao_caracteres': taxa_compressao,
            'num_caracteres_original': len(texto),
            'num_caracteres_sumario': len(sumario)
        }
        
        logger.success(
            f"Sumarização concluída: {num_original} → {num_sumario} sentenças "
            f"({taxa_compressao:.1%} de compressão)"
        )
        
        return resultado
    
    def sumarizar_por_categoria(
        self,
        df_classificado,
        metodo: str = 'textrank',
        taxa_reducao: float = 0.3
    ) -> Dict[str, str]:
        """
        Sumariza texto agrupado por categoria LGPD
        
        Args:
            df_classificado: DataFrame com colunas 'sentenca' e 'categoria'
            metodo: Método de sumarização
            taxa_reducao: Taxa de redução
            
        Returns:
            Dicionário {categoria: sumário}
        """
        sumarios = {}
        
        for categoria in df_classificado['categoria'].unique():
            sentencas_cat = df_classificado[
                df_classificado['categoria'] == categoria
            ]['sentenca'].tolist()
            
            if len(sentencas_cat) == 0:
                continue
            
            texto_cat = ' '.join(sentencas_cat)
            resultado = self.sumarizar(texto_cat, metodo, taxa_reducao)
            sumarios[categoria] = resultado['sumario']
        
        logger.info(f"Sumarizadas {len(sumarios)} categorias")
        return sumarios
