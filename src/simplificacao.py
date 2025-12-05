"""
Módulo de Simplificação Lexical
Reescreve textos complexos em linguagem simples para o público leigo
"""

import re
from typing import Dict, List, Optional
from loguru import logger
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# Dicionário de substituições de termos técnicos/jurídicos
SUBSTITUICOES_LGPD = {
    # Termos técnicos da LGPD
    'titular': 'pessoa dona dos dados',
    'controlador': 'empresa responsável pelos dados',
    'operador': 'empresa que processa os dados',
    'encarregado': 'responsável pela privacidade',
    'tratamento': 'uso',
    'dados pessoais': 'informações pessoais',
    'dados sensíveis': 'informações sensíveis',
    'consentimento': 'autorização',
    'finalidade': 'objetivo',
    'anonimização': 'remoção de identificação',
    'pseudonimização': 'substituição por código',
    'portabilidade': 'transferência dos dados',
    'revogação': 'cancelamento',
    'incidente de segurança': 'problema de segurança',
    'vazamento': 'exposição acidental',
    'transferência internacional': 'envio para outro país',
    'base legal': 'motivo legal',
    'legítimo interesse': 'interesse justificado',
    'cookies': 'arquivos de rastreamento',
    'dispositivo': 'aparelho',
    'geolocalização': 'localização',
    'log': 'registro',
    'criptografia': 'proteção por código',
    
    # Termos jurídicos comuns
    'mediante': 'por meio de',
    'conforme': 'de acordo com',
    'referente': 'sobre',
    'vigente': 'atual',
    'subsequente': 'seguinte',
    'em conformidade': 'de acordo',
    'no âmbito': 'no contexto',
    'por intermédio': 'através',
    'em virtude': 'por causa',
    'doravante': 'daqui em diante',
}


class SimplificadorTexto:
    """Classe para simplificação lexical de políticas de privacidade"""
    
    def __init__(self, usar_modelo: bool = False):
        """
        Inicializa o simplificador
        
        Args:
            usar_modelo: Se deve usar modelo de linguagem para simplificação
        """
        self.usar_modelo = usar_modelo
        self.pipeline = None
        
        logger.info(f"Inicializando SimplificadorTexto (modelo: {usar_modelo})")
        
        if usar_modelo:
            self._inicializar_modelo()
    
    def _inicializar_modelo(self):
        """Inicializa modelo de paráfrase/simplificação"""
        try:
            # Tentar carregar modelo de paráfrase
            dispositivo = 0 if torch.cuda.is_available() else -1
            self.pipeline = pipeline(
                "text2text-generation",
                model="t5-small",
                device=dispositivo
            )
            logger.success("Modelo de simplificação carregado")
        except Exception as e:
            logger.warning(f"Não foi possível carregar modelo: {e}")
            self.pipeline = None
    
    def substituir_termos_tecnicos(self, texto: str) -> str:
        """
        Substitui termos técnicos por equivalentes mais simples
        
        Args:
            texto: Texto original
            
        Returns:
            Texto com termos substituídos
        """
        texto_simplificado = texto
        
        # Ordenar por tamanho decrescente para evitar substituições parciais
        termos_ordenados = sorted(
            SUBSTITUICOES_LGPD.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        for termo_tecnico, termo_simples in termos_ordenados:
            # Substituição case-insensitive com preservação de capitalização
            pattern = re.compile(re.escape(termo_tecnico), re.IGNORECASE)
            
            def replacer(match):
                original = match.group(0)
                if original.isupper():
                    return termo_simples.upper()
                elif original[0].isupper():
                    return termo_simples.capitalize()
                else:
                    return termo_simples
            
            texto_simplificado = pattern.sub(replacer, texto_simplificado)
        
        return texto_simplificado
    
    def simplificar_sentenca(self, sentenca: str) -> str:
        """
        Simplifica uma sentença individual
        
        Args:
            sentenca: Sentença a simplificar
            
        Returns:
            Sentença simplificada
        """
        # 1. Substituir termos técnicos
        simplificada = self.substituir_termos_tecnicos(sentenca)
        
        # 2. Reduzir sentenças muito longas
        if len(simplificada.split()) > 30:
            # Quebrar em pontos de vírgula ou ponto-e-vírgula
            simplificada = simplificada.replace(';', '.')
            simplificada = simplificada.replace(', e ', '. ')
        
        # 3. Remover redundâncias comuns
        redundancias = [
            (r'\b(e|ou) também\b', r'\1'),
            (r'\b(informações e|e) dados\b', 'informações'),
            (r'\bprivacidade e proteção\b', 'privacidade'),
        ]
        
        for pattern, replacement in redundancias:
            simplificada = re.sub(pattern, replacement, simplificada, flags=re.IGNORECASE)
        
        return simplificada.strip()
    
    def simplificar_com_modelo(self, texto: str, max_length: int = 150) -> str:
        """
        Simplifica texto usando modelo de linguagem
        
        Args:
            texto: Texto original
            max_length: Comprimento máximo
            
        Returns:
            Texto simplificado
        """
        if self.pipeline is None:
            logger.warning("Modelo não disponível, usando simplificação por regras")
            return self.simplificar_sentenca(texto)
        
        try:
            prompt = f"simplify: {texto}"
            resultado = self.pipeline(
                prompt,
                max_length=max_length,
                do_sample=False
            )
            return resultado[0]['generated_text']
        except Exception as e:
            logger.error(f"Erro ao usar modelo: {e}")
            return self.simplificar_sentenca(texto)
    
    def simplificar(self, texto: str, usar_modelo: Optional[bool] = None) -> Dict:
        """
        Método principal de simplificação
        
        Args:
            texto: Texto completo
            usar_modelo: Override do uso de modelo (None = usa configuração da classe)
            
        Returns:
            Dicionário com texto simplificado e metadados
        """
        logger.info("Iniciando simplificação de texto")
        
        usar_ml = usar_modelo if usar_modelo is not None else self.usar_modelo
        
        # Dividir em sentenças
        from nltk.tokenize import sent_tokenize
        try:
            sentencas = sent_tokenize(texto, language='portuguese')
        except:
            sentencas = texto.split('.')
        
        # Simplificar cada sentença
        sentencas_simplificadas = []
        
        for sentenca in sentencas:
            if usar_ml and self.pipeline:
                simp = self.simplificar_com_modelo(sentenca)
            else:
                simp = self.simplificar_sentenca(sentenca)
            
            if simp.strip():
                sentencas_simplificadas.append(simp)
        
        texto_simplificado = ' '.join(sentencas_simplificadas)
        
        # Calcular métricas de simplificação
        palavras_original = texto.split()
        palavras_simplificado = texto_simplificado.split()
        
        resultado = {
            'texto_original': texto,
            'texto_simplificado': texto_simplificado,
            'num_sentencas': len(sentencas_simplificadas),
            'num_palavras_original': len(palavras_original),
            'num_palavras_simplificado': len(palavras_simplificado),
            'reducao_palavras': 1 - (len(palavras_simplificado) / len(palavras_original)) if len(palavras_original) > 0 else 0,
            'metodo': 'modelo' if usar_ml else 'regras'
        }
        
        logger.success(
            f"Simplificação concluída: {len(palavras_original)} → {len(palavras_simplificado)} palavras "
            f"({resultado['reducao_palavras']:.1%} de redução)"
        )
        
        return resultado
    
    def criar_glossario(self, texto: str) -> List[Dict]:
        """
        Identifica termos técnicos no texto e cria glossário
        
        Args:
            texto: Texto a analisar
            
        Returns:
            Lista de termos técnicos encontrados com explicações
        """
        texto_lower = texto.lower()
        glossario = []
        
        for termo_tecnico, termo_simples in SUBSTITUICOES_LGPD.items():
            if termo_tecnico in texto_lower:
                # Contar ocorrências
                ocorrencias = texto_lower.count(termo_tecnico)
                
                glossario.append({
                    'termo': termo_tecnico,
                    'explicacao': termo_simples,
                    'ocorrencias': ocorrencias
                })
        
        # Ordenar por número de ocorrências
        glossario.sort(key=lambda x: x['ocorrencias'], reverse=True)
        
        logger.info(f"Glossário criado com {len(glossario)} termos")
        return glossario
    
    def avaliar_complexidade(self, texto: str) -> Dict:
        """
        Avalia a complexidade lexical do texto
        
        Args:
            texto: Texto a avaliar
            
        Returns:
            Métricas de complexidade
        """
        palavras = texto.split()
        sentencas = texto.split('.')
        
        # Métricas básicas
        media_palavras_por_sentenca = len(palavras) / len(sentencas) if len(sentencas) > 0 else 0
        media_caracteres_por_palavra = sum(len(p) for p in palavras) / len(palavras) if len(palavras) > 0 else 0
        
        # Contar termos técnicos
        texto_lower = texto.lower()
        termos_tecnicos_encontrados = sum(
            1 for termo in SUBSTITUICOES_LGPD.keys()
            if termo in texto_lower
        )
        
        # Calcular índice de complexidade (simplificado)
        # Valores maiores = mais complexo
        indice_complexidade = (
            (media_palavras_por_sentenca / 15) * 0.4 +
            (media_caracteres_por_palavra / 6) * 0.3 +
            (termos_tecnicos_encontrados / 10) * 0.3
        )
        
        resultado = {
            'num_palavras': len(palavras),
            'num_sentencas': len(sentencas),
            'media_palavras_por_sentenca': media_palavras_por_sentenca,
            'media_caracteres_por_palavra': media_caracteres_por_palavra,
            'termos_tecnicos': termos_tecnicos_encontrados,
            'indice_complexidade': min(indice_complexidade, 1.0),  # Normalizar para [0, 1]
            'nivel': self._classificar_complexidade(indice_complexidade)
        }
        
        return resultado
    
    def _classificar_complexidade(self, indice: float) -> str:
        """Classifica o nível de complexidade"""
        if indice < 0.3:
            return 'Simples'
        elif indice < 0.6:
            return 'Moderado'
        elif indice < 0.8:
            return 'Complexo'
        else:
            return 'Muito Complexo'
