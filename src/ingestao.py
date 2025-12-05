"""
Módulo de Ingestão de Políticas de Privacidade
Responsável por carregar documentos de diferentes fontes (TXT, PDF, URL)
"""

import requests
from pathlib import Path
from typing import Union, Optional
from loguru import logger
import PyPDF2
from io import BytesIO
from bs4 import BeautifulSoup
import re
import time


class IngestorPoliticas:
    """Classe para ingestão de políticas de privacidade de múltiplas fontes"""
    
    def __init__(self):
        logger.info("Inicializando IngestorPoliticas")
    
    def carregar_txt(self, caminho: Union[str, Path]) -> str:
        """
        Carrega conteúdo de arquivo TXT
        
        Args:
            caminho: Caminho para o arquivo TXT
            
        Returns:
            Conteúdo do arquivo como string
        """
        try:
            with open(caminho, 'r', encoding='utf-8') as f:
                conteudo = f.read()
            logger.success(f"TXT carregado com sucesso: {caminho}")
            return conteudo
        except Exception as e:
            logger.error(f"Erro ao carregar TXT {caminho}: {e}")
            raise
    
    def carregar_pdf(self, caminho: Union[str, Path, BytesIO]) -> str:
        """
        Carrega conteúdo de arquivo PDF
        
        Args:
            caminho: Caminho para o arquivo PDF ou objeto BytesIO
            
        Returns:
            Conteúdo extraído do PDF como string
        """
        try:
            if isinstance(caminho, BytesIO):
                pdf_reader = PyPDF2.PdfReader(caminho)
            else:
                with open(caminho, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    texto = []
                    for pagina in pdf_reader.pages:
                        texto.append(pagina.extract_text())
                    return '\n'.join(texto)
            
            texto = []
            for pagina in pdf_reader.pages:
                texto.append(pagina.extract_text())
            
            logger.success(f"PDF carregado com sucesso: {len(pdf_reader.pages)} páginas")
            return '\n'.join(texto)
        except Exception as e:
            logger.error(f"Erro ao carregar PDF: {e}")
            raise
    
    def carregar_url(self, url: str, timeout: int = 10) -> str:
        """
        Carrega conteúdo de uma URL e extrai texto limpo
        
        Args:
            url: URL da política de privacidade
            timeout: Tempo máximo de espera em segundos
            
        Returns:
            Texto extraído e limpo da página
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            
            # Usar BeautifulSoup para extrair texto limpo
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remover elementos desnecessários (navegação, scripts, estilos, etc)
            for elemento in soup(["script", "style", "nav", "footer", "header", "aside", 
                                  "menu", "form", "button", "noscript", "iframe"]):
                elemento.decompose()
            
            # Tentar encontrar o conteúdo principal (artigos, main, divs com conteúdo)
            conteudo_principal = None
            
            # Priorizar tags semânticas de conteúdo
            for tag in ['article', 'main', '[role="main"]', '.content', '#content', '.main-content']:
                conteudo_principal = soup.select_one(tag)
                if conteudo_principal:
                    break
            
            # Se não encontrar, usar body inteiro
            if not conteudo_principal:
                conteudo_principal = soup.body if soup.body else soup
            
            # Extrair texto
            texto = conteudo_principal.get_text(separator='\n', strip=True)
            
            # Limpar e filtrar linhas - VERSÃO MENOS RESTRITIVA
            linhas = []
            linhas_vistas_exatas = set()  # Para duplicatas EXATAS apenas
            
            import re
            
            for linha in texto.split('\n'):
                linha = linha.strip()
                
                # Filtros MÍNIMOS de limpeza:
                
                # 1. Ignorar linhas MUITO curtas (< 3 caracteres) - apenas lixo
                if len(linha) < 3:
                    continue
                
                # 2. Ignorar linhas EXATAMENTE repetidas (não case-insensitive)
                if linha in linhas_vistas_exatas:
                    continue
                
                # 3. Ignorar apenas números isolados muito curtos
                if linha.isdigit() and len(linha) <= 2:
                    continue
                
                # 4. Ignorar linhas com caracteres especiais repetidos (mais de 5 vezes)
                if linha.count('|') > 5 or linha.count('>') > 5 or linha.count('=') > 5:
                    continue
                
                # 5. Ignorar apenas linhas que são numeração de seção vazia (ex: "1.", "2.1")
                if re.match(r'^\d+\.(\d+\.?)*\s*$', linha):
                    continue
                
                # 6. Ignorar cookies de navegação específicos
                palavras_cookies = ['aceitar cookies', 'aceitar todos os cookies', 
                                   'gerenciar cookies', 'política de cookies']
                if any(cookie in linha.lower() for cookie in palavras_cookies) and len(linha) < 50:
                    continue
                
                linhas_vistas_exatas.add(linha)
                linhas.append(linha)
            
            texto_limpo = '\n\n'.join(linhas)
            
            # Limpar espaços múltiplos e quebras excessivas
            texto_limpo = re.sub(r'\n{3,}', '\n\n', texto_limpo)  # Max 2 quebras
            texto_limpo = re.sub(r' {2,}', ' ', texto_limpo)  # Max 1 espaço
            
            logger.success(f"URL carregada e processada: {url} ({len(texto_limpo)} caracteres, {len(linhas)} linhas)")
            return texto_limpo
        except Exception as e:
            logger.error(f"Erro ao carregar URL {url}: {e}")
            raise
    
    def carregar(self, fonte: Union[str, Path, BytesIO], tipo: Optional[str] = None) -> str:
        """
        Método genérico para carregar conteúdo de qualquer fonte
        
        Args:
            fonte: Caminho do arquivo, URL ou objeto BytesIO
            tipo: Tipo da fonte ('txt', 'pdf', 'url'). Se None, tenta inferir
            
        Returns:
            Conteúdo carregado como string
        """
        if tipo is None:
            # Inferir tipo
            if isinstance(fonte, BytesIO):
                tipo = 'pdf'
            elif isinstance(fonte, (str, Path)):
                fonte_str = str(fonte)
                if fonte_str.startswith(('http://', 'https://')):
                    tipo = 'url'
                elif fonte_str.endswith('.pdf'):
                    tipo = 'pdf'
                elif fonte_str.endswith('.txt'):
                    tipo = 'txt'
                else:
                    tipo = 'txt'  # Default
        
        if tipo == 'txt':
            return self.carregar_txt(fonte)
        elif tipo == 'pdf':
            return self.carregar_pdf(fonte)
        elif tipo == 'url':
            return self.carregar_url(fonte)
        else:
            raise ValueError(f"Tipo de fonte não suportado: {tipo}")
