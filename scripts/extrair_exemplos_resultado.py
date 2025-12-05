"""
Script para extrair exemplos classificados de um resultado e preparar para contribuição

Este script extrai os textos classificados de uma análise completa e os prepara
para serem adicionados ao dataset global após revisão manual.

Uso:
    # Extrair de um resultado específico
    python scripts/extrair_exemplos_resultado.py --resultado results/nubank/nubank_extrativo_textrank_0_3_20251204_120000
    
    # Ou extrair automaticamente do último resultado da empresa
    python scripts/extrair_exemplos_resultado.py --empresa nubank
    
    # Especificar quantos exemplos por categoria (padrão: 5)
    python scripts/extrair_exemplos_resultado.py --empresa nubank --num-por-categoria 10
"""

import pandas as pd
import argparse
from pathlib import Path
from loguru import logger
import random


def encontrar_ultimo_resultado(empresa: str) -> Path:
    """Encontra o último resultado de uma empresa"""
    results_dir = Path("results") / empresa
    
    if not results_dir.exists():
        logger.error(f"Diretório não encontrado: {results_dir}")
        return None
    
    # Listar subdiretórios
    subdirs = sorted([d for d in results_dir.iterdir() if d.is_dir()], 
                     key=lambda x: x.name, reverse=True)
    
    if not subdirs:
        logger.error(f"Nenhum resultado encontrado em {results_dir}")
        return None
    
    return subdirs[0]


def extrair_exemplos_classificados(
    caminho_resultado: Path,
    num_por_categoria: int = 5
) -> pd.DataFrame:
    """
    Extrai exemplos classificados de um resultado
    
    Args:
        caminho_resultado: Caminho para o diretório do resultado
        num_por_categoria: Número de exemplos a extrair por categoria
        
    Returns:
        DataFrame com colunas 'texto' e 'categoria'
    """
    # Procurar arquivo de classificação
    arquivo_class = None
    
    # Tentar encontrar o arquivo
    for pattern in ["resultado_*/02_classificacao_lgpd.csv", "*/02_classificacao_lgpd.csv"]:
        arquivos = list(caminho_resultado.glob(pattern))
        if arquivos:
            arquivo_class = arquivos[0]
            break
    
    if not arquivo_class:
        logger.error(f"Arquivo de classificação não encontrado em {caminho_resultado}")
        return None
    
    logger.info(f"Lendo: {arquivo_class}")
    df = pd.read_csv(arquivo_class)
    
    # Verificar colunas
    if 'sentenca' not in df.columns or 'categoria_prevista' not in df.columns:
        logger.error(f"Colunas esperadas não encontradas. Colunas disponíveis: {df.columns.tolist()}")
        return None
    
    logger.success(f"Carregado: {len(df)} sentenças classificadas")
    
    # Renomear colunas
    df_exemplos = df.rename(columns={
        'sentenca': 'texto',
        'categoria_prevista': 'categoria'
    })[['texto', 'categoria']]
    
    # Mostrar distribuição
    logger.info("\nDistribuição por categoria:")
    distribuicao = df_exemplos['categoria'].value_counts()
    for cat, count in distribuicao.items():
        logger.info(f"   {cat}: {count}")
    
    # Selecionar amostra balanceada
    exemplos_selecionados = []
    
    for categoria in df_exemplos['categoria'].unique():
        df_cat = df_exemplos[df_exemplos['categoria'] == categoria]
        
        # Pegar amostra aleatória
        n = min(num_por_categoria, len(df_cat))
        amostra = df_cat.sample(n=n, random_state=42)
        exemplos_selecionados.append(amostra)
        
        logger.info(f"   ✓ Selecionados {n} exemplos de '{categoria}'")
    
    df_final = pd.concat(exemplos_selecionados, ignore_index=True)
    logger.success(f"\nTotal de exemplos selecionados: {len(df_final)}")
    
    return df_final


def salvar_para_revisao(df: pd.DataFrame, empresa: str, caminho_saida: Path):
    """Salva exemplos em CSV para revisão manual"""
    df.to_csv(caminho_saida, index=False, encoding='utf-8')
    logger.success(f"\nExemplos salvos em: {caminho_saida}")
    
    logger.info("\n" + "="*80)
    logger.info("PRÓXIMOS PASSOS:")
    logger.info("="*80)
    logger.info("1. Revise o arquivo CSV gerado")
    logger.info("2. Corrija categorias se necessário")
    logger.info("3. Remova exemplos ruins ou duplicados")
    logger.info("4. Execute:")
    logger.info(f"   python scripts/contribuir_exemplos.py --empresa {empresa} --arquivo {caminho_saida}")


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Extrair exemplos classificados para contribuição ao dataset"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--empresa', type=str, help="Nome da empresa (busca último resultado)")
    group.add_argument('--resultado', type=Path, help="Caminho completo para o resultado")
    
    parser.add_argument(
        '--num-por-categoria', 
        type=int, 
        default=5,
        help="Número de exemplos a extrair por categoria (padrão: 5)"
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help="Caminho para salvar os exemplos (padrão: data/temp/empresa_para_revisar.csv)"
    )
    
    args = parser.parse_args()
    
    # Determinar caminho do resultado
    if args.empresa:
        logger.info(f"Buscando último resultado de: {args.empresa}")
        caminho_resultado = encontrar_ultimo_resultado(args.empresa)
        empresa = args.empresa
    else:
        caminho_resultado = args.resultado
        empresa = caminho_resultado.parent.name
    
    if not caminho_resultado:
        return
    
    logger.info(f"Usando resultado: {caminho_resultado}")
    
    # Extrair exemplos
    df_exemplos = extrair_exemplos_classificados(
        caminho_resultado, 
        num_por_categoria=args.num_por_categoria
    )
    
    if df_exemplos is None or len(df_exemplos) == 0:
        logger.error("Nenhum exemplo extraído")
        return
    
    # Determinar caminho de saída
    if args.output:
        caminho_saida = args.output
    else:
        Path("data/temp").mkdir(parents=True, exist_ok=True)
        caminho_saida = Path(f"data/temp/{empresa}_para_revisar.csv")
    
    # Salvar
    salvar_para_revisao(df_exemplos, empresa, caminho_saida)
    
    logger.info("\n" + "="*80)
    logger.info("DICA: Para melhor qualidade do dataset:")
    logger.info("="*80)
    logger.info("• Revise se as categorias estão corretas")
    logger.info("• Remova textos muito curtos ou genéricos")
    logger.info("• Remova duplicatas ou textos muito similares")
    logger.info("• Priorize textos claros e representativos")
    logger.info("="*80)


if __name__ == "__main__":
    main()
