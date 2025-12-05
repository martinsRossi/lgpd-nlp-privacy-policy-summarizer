"""
Módulo de Pré-processamento de Texto
Responsável por limpeza, tokenização e normalização de políticas de privacidade
"""

import re
import string
from typing import List, Optional
from loguru import logger
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import RSLPStemmer
import spacy
from bs4 import BeautifulSoup


class PreprocessadorTexto:
    """Classe para pré-processamento de texto de políticas de privacidade"""
    
    def __init__(self, idioma: str = 'portuguese'):
        """
        Inicializa o preprocessador
        
        Args:
            idioma: Idioma para processamento (portuguese, english)
        """
        self.idioma = idioma
        logger.info(f"Inicializando PreprocessadorTexto (idioma: {idioma})")
        
        # Download de recursos necessários do NLTK
        self._baixar_recursos_nltk()
        
        # Configurar stopwords
        if idioma == 'portuguese':
            self.stopwords = set(stopwords.words('portuguese'))
            self.stemmer = RSLPStemmer()
        else:
            self.stopwords = set(stopwords.words('english'))
            self.stemmer = nltk.stem.PorterStemmer()
        
        # Tentar carregar spaCy
        try:
            if idioma == 'portuguese':
                self.nlp = spacy.load('pt_core_news_sm')
            else:
                self.nlp = spacy.load('en_core_web_sm')
            logger.success(f"Modelo spaCy carregado: {idioma}")
        except OSError:
            logger.warning(f"Modelo spaCy não encontrado para {idioma}. Algumas funcionalidades estarão limitadas.")
            self.nlp = None
    
    def _baixar_recursos_nltk(self):
        """Baixa recursos necessários do NLTK"""
        recursos = ['punkt', 'stopwords', 'rslp', 'punkt_tab']
        for recurso in recursos:
            try:
                nltk.data.find(f'tokenizers/{recurso}')
            except LookupError:
                try:
                    nltk.download(recurso, quiet=True)
                except:
                    pass
    
    def remover_html(self, texto: str) -> str:
        """
        Remove tags HTML do texto
        
        Args:
            texto: Texto com possíveis tags HTML
            
        Returns:
            Texto limpo sem HTML
        """
        soup = BeautifulSoup(texto, 'html.parser')
        return soup.get_text(separator=' ')
    
    def limpar_texto(self, texto: str) -> str:
        """
        Realiza limpeza básica do texto
        
        Args:
            texto: Texto bruto
            
        Returns:
            Texto limpo
        """
        # Remover HTML
        texto = self.remover_html(texto)
        
        # Remover URLs
        texto = re.sub(r'http\S+|www\S+', '', texto)
        
        # Remover emails
        texto = re.sub(r'\S+@\S+', '', texto)
        
        # Normalizar espaços em branco
        texto = re.sub(r'\s+', ' ', texto)
        
        # Remover caracteres especiais excessivos mantendo pontuação básica
        texto = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', '', texto)
        
        # Remover espaços extras
        texto = texto.strip()
        
        logger.debug(f"Texto limpo: {len(texto)} caracteres")
        return texto
    
    def tokenizar_sentencas(self, texto: str) -> List[str]:
        """
        Divide o texto em sentenças
        
        Args:
            texto: Texto completo
            
        Returns:
            Lista de sentenças
        """
        try:
            sentencas = sent_tokenize(texto, language='portuguese' if self.idioma == 'portuguese' else 'english')
            
            # Filtrar sentenças inválidas
            sentencas_validas = []
            for sent in sentencas:
                sent_limpa = sent.strip()
                
                # Ignorar sentenças muito curtas (< 15 caracteres)
                if len(sent_limpa) < 15:
                    continue
                
                # Ignorar sentenças que são apenas números de seção (ex: "2.", "3.1", "4.")
                if re.match(r'^\d+\.(\d+)?\.?\s*$', sent_limpa):
                    continue
                
                # Ignorar sentenças que são apenas títulos em MAIÚSCULA sem conteúdo
                if sent_limpa.isupper() and len(sent_limpa) < 60:
                    continue
                
                # Ignorar sentenças com apenas pontuação
                sem_pontuacao = re.sub(r'[^\w\s]', '', sent_limpa)
                if len(sem_pontuacao.strip()) < 10:
                    continue
                
                sentencas_validas.append(sent)
            
            logger.debug(f"Texto dividido em {len(sentencas)} sentenças ({len(sentencas_validas)} válidas)")
            return sentencas_validas
        except Exception as e:
            logger.error(f"Erro ao tokenizar sentenças: {e}")
            # Fallback simples
            return texto.split('.')
    
    def tokenizar_palavras(self, texto: str, lowercase: bool = True) -> List[str]:
        """
        Divide o texto em palavras (tokens)
        
        Args:
            texto: Texto para tokenização
            lowercase: Se deve converter para minúsculas
            
        Returns:
            Lista de tokens
        """
        if lowercase:
            texto = texto.lower()
        
        tokens = word_tokenize(texto, language='portuguese' if self.idioma == 'portuguese' else 'english')
        logger.debug(f"Texto tokenizado em {len(tokens)} palavras")
        return tokens
    
    def remover_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords dos tokens
        
        Args:
            tokens: Lista de tokens
            
        Returns:
            Lista de tokens sem stopwords
        """
        tokens_filtrados = [t for t in tokens if t.lower() not in self.stopwords]
        logger.debug(f"Removidas {len(tokens) - len(tokens_filtrados)} stopwords")
        return tokens_filtrados
    
    def remover_pontuacao(self, tokens: List[str]) -> List[str]:
        """
        Remove pontuação dos tokens
        
        Args:
            tokens: Lista de tokens
            
        Returns:
            Lista de tokens sem pontuação
        """
        tokens_sem_pont = [t for t in tokens if t not in string.punctuation]
        return tokens_sem_pont
    
    def aplicar_stemming(self, tokens: List[str]) -> List[str]:
        """
        Aplica stemming nos tokens
        
        Args:
            tokens: Lista de tokens
            
        Returns:
            Lista de tokens com stemming aplicado
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def processar_completo(
        self,
        texto: str,
        remover_stop: bool = True,
        remover_pont: bool = True,
        aplicar_stem: bool = False,
        manter_sentencas: bool = False
    ) -> dict:
        """
        Pipeline completo de pré-processamento
        
        Args:
            texto: Texto bruto
            remover_stop: Se deve remover stopwords
            remover_pont: Se deve remover pontuação
            aplicar_stem: Se deve aplicar stemming
            manter_sentencas: Se deve retornar também as sentenças
            
        Returns:
            Dicionário com texto processado e metadados
        """
        logger.info("Iniciando processamento completo")
        
        # 1. Limpeza
        texto_limpo = self.limpar_texto(texto)
        
        # 2. Tokenização de sentenças
        sentencas = self.tokenizar_sentencas(texto_limpo) if manter_sentencas else []
        
        # 3. Tokenização de palavras
        tokens = self.tokenizar_palavras(texto_limpo)
        
        # 4. Remover pontuação (opcional)
        if remover_pont:
            tokens = self.remover_pontuacao(tokens)
        
        # 5. Remover stopwords (opcional)
        if remover_stop:
            tokens = self.remover_stopwords(tokens)
        
        # 6. Stemming (opcional)
        if aplicar_stem:
            tokens = self.aplicar_stemming(tokens)
        
        resultado = {
            'texto_original': texto,
            'texto_limpo': texto_limpo,
            'tokens': tokens,
            'sentencas': sentencas,
            'num_tokens': len(tokens),
            'num_sentencas': len(sentencas),
            'num_caracteres': len(texto_limpo)
        }
        
        logger.success(f"Processamento concluído: {resultado['num_tokens']} tokens, {resultado['num_sentencas']} sentenças")
        return resultado
    
    def extrair_entidades(self, texto: str) -> List[dict]:
        """
        Extrai entidades nomeadas do texto usando spaCy
        
        Args:
            texto: Texto para extração
            
        Returns:
            Lista de entidades encontradas
        """
        if self.nlp is None:
            logger.warning("spaCy não disponível. Pulando extração de entidades.")
            return []
        
        doc = self.nlp(texto)
        entidades = [
            {
                'texto': ent.text,
                'tipo': ent.label_,
                'inicio': ent.start_char,
                'fim': ent.end_char
            }
            for ent in doc.ents
        ]
        
        logger.debug(f"Extraídas {len(entidades)} entidades")
        return entidades
