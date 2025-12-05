"""
Script para contribuir exemplos rotulados de uma nova política ao dataset global

Este script permite adicionar novos exemplos manualmente rotulados ao dataset unificado,
seguindo a metodologia de treinamento global e incremental.

Uso:
    python scripts/contribuir_exemplos.py --empresa amazon --arquivo rotulos_amazon.csv
    
    Ou interativamente:
    python scripts/contribuir_exemplos.py
"""

import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path
from loguru import logger

# Categorias válidas
CATEGORIAS_VALIDAS = {
    'coleta_dados', 'finalidade', 'compartilhamento', 'armazenamento',
    'seguranca', 'direitos_titular', 'cookies', 'internacional',
    'criancas', 'contato', 'alteracoes', 'outros'
}

def validar_rotulos(df: pd.DataFrame) -> bool:
    """
    Valida que os rótulos estão no formato correto
    
    Args:
        df: DataFrame com colunas 'texto' e 'categoria'
        
    Returns:
        True se válido, False caso contrário
    """
    # Verificar colunas obrigatórias
    if 'texto' not in df.columns or 'categoria' not in df.columns:
        logger.error("DataFrame deve ter colunas 'texto' e 'categoria'")
        return False
    
    # Verificar categorias válidas
    categorias_invalidas = set(df['categoria'].unique()) - CATEGORIAS_VALIDAS
    if categorias_invalidas:
        logger.error(f"Categorias inválidas encontradas: {categorias_invalidas}")
        logger.info(f"Categorias válidas: {CATEGORIAS_VALIDAS}")
        return False
    
    # Verificar textos vazios
    textos_vazios = df['texto'].isna().sum()
    if textos_vazios > 0:
        logger.error(f"Encontrados {textos_vazios} textos vazios")
        return False
    
    logger.success("Rótulos validados com sucesso")
    return True


def adicionar_ao_dataset_global(
    df_novos: pd.DataFrame,
    empresa: str,
    versao_atual: str = "v1.0"
) -> bool:
    """
    Adiciona novos exemplos rotulados ao dataset global
    
    Args:
        df_novos: DataFrame com colunas 'texto' e 'categoria'
        empresa: Nome da empresa de origem (ex: 'amazon', 'google')
        versao_atual: Versão atual do modelo (ex: 'v1.0', 'v1.1')
        
    Returns:
        True se adicionado com sucesso
    """
    caminho_global = Path("data/lgpd_rotulado_global.csv")
    
    # Carregar dataset global existente
    if caminho_global.exists():
        df_global = pd.read_csv(caminho_global)
        logger.info(f"Dataset global atual: {len(df_global)} exemplos")
    else:
        logger.warning("Dataset global não existe, será criado")
        df_global = pd.DataFrame(columns=['texto', 'categoria', 'empresa_origem', 'data_contribuicao', 'versao_modelo'])
    
    # Adicionar metadados aos novos exemplos
    df_novos = df_novos.copy()
    df_novos['empresa_origem'] = empresa
    df_novos['data_contribuicao'] = datetime.now().isoformat()
    df_novos['versao_modelo'] = versao_atual
    
    # Concatenar
    df_atualizado = pd.concat([df_global, df_novos], ignore_index=True)
    
    # Salvar
    df_atualizado.to_csv(caminho_global, index=False)
    
    logger.success(f"Adicionados {len(df_novos)} exemplos de {empresa}")
    logger.info(f"Dataset global agora tem {len(df_atualizado)} exemplos")
    
    # Estatísticas
    logger.info(f"\nDistribuição por empresa:")
    print(df_atualizado['empresa_origem'].value_counts())
    
    logger.info(f"\nDistribuição por categoria:")
    print(df_atualizado['categoria'].value_counts())
    
    # Verificar threshold para retreinamento
    novos_desde_ultima_versao = df_atualizado[
        df_atualizado['versao_modelo'] == versao_atual
    ]
    
    if len(novos_desde_ultima_versao) >= 50:
        logger.warning(f"\nTHRESHOLD ATINGIDO: {len(novos_desde_ultima_versao)} novos exemplos desde {versao_atual}")
        logger.info("💡 Considere retreinar o modelo com uma nova versão")
        logger.info("   Comando: python scripts/treinar_classificador_global.py --versao v1.1")
    
    return True


def modo_interativo():
    """Modo interativo para adicionar exemplos"""
    print("=" * 70)
    print("📚 CONTRIBUIR EXEMPLOS AO DATASET GLOBAL LGPD")
    print("=" * 70)
    
    # Solicitar informações
    empresa = input("\nNome da empresa (ex: amazon, google): ").strip().lower()
    
    arquivo = input("Caminho para arquivo CSV com rótulos: ").strip()
    
    versao = input("Versão atual do modelo (ex: v1.0, v1.1) [v1.0]: ").strip() or "v1.0"
    
    # Carregar arquivo
    try:
        df_novos = pd.read_csv(arquivo)
        logger.info(f"Arquivo carregado: {len(df_novos)} exemplos")
    except Exception as e:
        logger.error(f"Erro ao carregar arquivo: {e}")
        return False
    
    # Validar
    if not validar_rotulos(df_novos):
        return False
    
    # Mostrar amostra
    print("\n🔍 Amostra dos dados:")
    print(df_novos.head(3).to_string())
    
    # Confirmar
    confirmacao = input("\nAdicionar estes exemplos ao dataset global? (s/n): ").strip().lower()
    
    if confirmacao != 's':
        logger.info("Operação cancelada")
        return False
    
    # Adicionar
    return adicionar_ao_dataset_global(df_novos, empresa, versao)


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Contribuir exemplos rotulados ao dataset global LGPD"
    )
    parser.add_argument('--empresa', type=str, help="Nome da empresa de origem")
    parser.add_argument('--arquivo', type=str, help="Caminho para arquivo CSV com rótulos")
    parser.add_argument('--versao', type=str, default="v1.0", help="Versão atual do modelo")
    
    args = parser.parse_args()
    
    # Modo linha de comando ou interativo
    if args.empresa and args.arquivo:
        logger.info(f"   Modo linha de comando")
        logger.info(f"   Empresa: {args.empresa}")
        logger.info(f"   Arquivo: {args.arquivo}")
        logger.info(f"   Versão: {args.versao}")
        
        try:
            df_novos = pd.read_csv(args.arquivo)
            logger.info(f"Arquivo carregado: {len(df_novos)} exemplos")
            
            if validar_rotulos(df_novos):
                adicionar_ao_dataset_global(df_novos, args.empresa, args.versao)
        except Exception as e:
            logger.error(f"Erro: {e}")
            return 1
    else:
        # Modo interativo
        if not modo_interativo():
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
