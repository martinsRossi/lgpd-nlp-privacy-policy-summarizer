"""
Arquivo __init__ para o pacote src
"""

__version__ = "1.0.0"
__author__ = "TCC - Anchieta"
__description__ = "Sistema de Sumarização Automática de Políticas de Privacidade LGPD"

from .ingestao import IngestorPoliticas
from .preprocessamento import PreprocessadorTexto
from .classificador_lgpd import ClassificadorLGPD, CATEGORIAS_LGPD
from .sumarizacao_extrativa import SumarizadorExtrativo
from .sumarizacao_abstrativa import SumarizadorAbstrativo
from .simplificacao import SimplificadorTexto
from .avaliacao import AvaliadorSumarizacao
from .treinamento import TreinadorModelo

__all__ = [
    'IngestorPoliticas',
    'PreprocessadorTexto',
    'ClassificadorLGPD',
    'CATEGORIAS_LGPD',
    'SumarizadorExtrativo',
    'SumarizadorAbstrativo',
    'SimplificadorTexto',
    'AvaliadorSumarizacao',
    'TreinadorModelo'
]
