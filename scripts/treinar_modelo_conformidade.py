"""
Script de Treinamento do Modelo de Conformidade LGPD
Treina o modelo que avalia se uma política está conforme ou não com a LGPD
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.modelo_conformidade_lgpd import ModeloConformidadeLGPD
from src.dataset_conformidade import CarregadorDatasetConformidade, criar_dataset_exemplo
from src.treinamento import TreinadorModelo
from loguru import logger
import argparse


def treinar_modelo_conformidade(
    caminho_dataset: Path,
    caminho_modelo_saida: Path,
    tipo_modelo: str = 'gradient_boosting',
    num_epocas: int = 50,
    test_size: float = 0.2
):
    """
    Função principal de treinamento do modelo de conformidade
    
    Args:
        caminho_dataset: Caminho do CSV com dados de treinamento
        caminho_modelo_saida: Onde salvar o modelo treinado
        tipo_modelo: Tipo do modelo ML
        num_epocas: Número de épocas
        test_size: Proporção do conjunto de teste
    """
    logger.info("="*80)
    logger.info("TREINAMENTO DO MODELO DE CONFORMIDADE LGPD")
    logger.info("="*80)
    
    # 1. Carregar dataset
    logger.info(f"Carregando dataset de {caminho_dataset}")
    
    if not caminho_dataset.exists():
        logger.warning(f"Dataset não encontrado em {caminho_dataset}")
        logger.info("Gerando dataset de exemplo...")
        criar_dataset_exemplo(caminho_dataset)
    
    df = CarregadorDatasetConformidade.carregar_csv(caminho_dataset)
    
    # Estatísticas
    logger.info(f"\nEstatísticas do Dataset:")
    logger.info(f"  Total de exemplos: {len(df)}")
    logger.info(f"  Conformes: {(df['classe_binaria'] == 1).sum()} ({(df['classe_binaria'] == 1).sum()/len(df)*100:.1f}%)")
    logger.info(f"  Não conformes: {(df['classe_binaria'] == 0).sum()} ({(df['classe_binaria'] == 0).sum()/len(df)*100:.1f}%)")
    logger.info(f"  Score médio: {df['score_conformidade'].mean():.2f}")
    
    # 2. Dividir treino/teste
    df_treino, df_teste = CarregadorDatasetConformidade.dividir_treino_teste(
        df, test_size=test_size
    )
    
    # 3. Preparar dados
    X_train = df_treino['texto'].tolist()
    y_train = df_treino['score_conformidade'].tolist()
    X_test = df_teste['texto'].tolist()
    y_test = df_teste['score_conformidade'].tolist()
    
    # 4. Inicializar modelo
    logger.info(f"\nInicializando modelo: {tipo_modelo}")
    modelo = ModeloConformidadeLGPD(tipo_modelo=tipo_modelo)
    
    # 5. Treinar
    logger.info(f"\nIniciando treinamento com {num_epocas} épocas...")
    resultado_treino = modelo.treinar(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        num_epocas=num_epocas,
        verbose=True
    )
    
    # 6. Exibir resultados
    logger.info("\n" + "="*80)
    logger.success("TREINAMENTO CONCLUÍDO")
    logger.info("="*80)
    logger.info(f"Tempo total: {resultado_treino['tempo_total']:.2f}s")
    logger.info(f"Épocas executadas: {resultado_treino['num_epocas']}")
    logger.info(f"Acurácia final: {resultado_treino['acuracia_final']:.3f}")
    logger.info(f"Melhor F1-Score: {resultado_treino['melhor_f1']:.3f}")
    
    # Relatório detalhado
    relatorio = resultado_treino['relatorio_classificacao']
    logger.info("\nRelatório de Classificação:")
    logger.info(f"  Classe 'Não Conforme':")
    logger.info(f"    Precision: {relatorio['Não Conforme']['precision']:.3f}")
    logger.info(f"    Recall:    {relatorio['Não Conforme']['recall']:.3f}")
    logger.info(f"    F1-Score:  {relatorio['Não Conforme']['f1-score']:.3f}")
    logger.info(f"  Classe 'Conforme':")
    logger.info(f"    Precision: {relatorio['Conforme']['precision']:.3f}")
    logger.info(f"    Recall:    {relatorio['Conforme']['recall']:.3f}")
    logger.info(f"    F1-Score:  {relatorio['Conforme']['f1-score']:.3f}")
    
    # 7. Gerar gráficos
    logger.info("\nGerando gráficos de treinamento...")
    treinador = TreinadorModelo(Path("docs/figuras"))
    treinador.historico = resultado_treino['historico']
    treinador.plotar_curvas_treinamento(salvar=True, mostrar=False)
    
    # 8. Salvar modelo
    logger.info(f"\nSalvando modelo em {caminho_modelo_saida}")
    caminho_modelo_saida.parent.mkdir(parents=True, exist_ok=True)
    modelo.salvar_modelo(caminho_modelo_saida)
    
    # 9. Teste de inferência
    logger.info("\n" + "="*80)
    logger.info("TESTE DE INFERÊNCIA")
    logger.info("="*80)
    
    # Pegar um exemplo de teste
    exemplo_conforme = df_teste[df_teste['classe_binaria'] == 1].iloc[0]['texto']
    exemplo_nao_conforme = df_teste[df_teste['classe_binaria'] == 0].iloc[0]['texto']
    
    logger.info("\nTestando política CONFORME:")
    resultado = modelo.prever_conformidade(exemplo_conforme[:500])
    logger.info(f"  Score: {resultado.score_conformidade:.2f}/100")
    logger.info(f"  Decisão: {resultado.conformidade_binaria}")
    logger.info(f"  Recomendação: {resultado.recomendacao}")
    logger.info(f"  Requisitos atendidos: {len(resultado.requisitos_atendidos)}")
    
    logger.info("\nTestando política NÃO CONFORME:")
    resultado = modelo.prever_conformidade(exemplo_nao_conforme[:500])
    logger.info(f"  Score: {resultado.score_conformidade:.2f}/100")
    logger.info(f"  Decisão: {resultado.conformidade_binaria}")
    logger.info(f"  Recomendação: {resultado.recomendacao}")
    logger.info(f"  Requisitos atendidos: {len(resultado.requisitos_atendidos)}")
    
    logger.success("\n✓ Modelo de conformidade treinado e salvo com sucesso!")
    logger.info(f"Modelo salvo em: {caminho_modelo_saida}")
    logger.info(f"Gráficos salvos em: docs/figuras/")
    
    return modelo, resultado_treino


def main():
    """Função principal CLI"""
    parser = argparse.ArgumentParser(
        description="Treina modelo de avaliação de conformidade LGPD"
    )
    
    parser.add_argument(
        '--dataset',
        type=Path,
        default=Path('data/dataset_conformidade_exemplo.csv'),
        help='Caminho do dataset CSV'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('models/modelo_conformidade_lgpd.pkl'),
        help='Caminho para salvar o modelo'
    )
    
    parser.add_argument(
        '--modelo',
        type=str,
        default='gradient_boosting',
        choices=['logistic', 'gradient_boosting', 'random_forest', 'mlp'],
        help='Tipo de modelo'
    )
    
    parser.add_argument(
        '--epocas',
        type=int,
        default=50,
        help='Número de épocas de treinamento'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proporção do conjunto de teste (0.0 a 1.0)'
    )
    
    args = parser.parse_args()
    
    # Executar treinamento
    treinar_modelo_conformidade(
        caminho_dataset=args.dataset,
        caminho_modelo_saida=args.output,
        tipo_modelo=args.modelo,
        num_epocas=args.epocas,
        test_size=args.test_size
    )


if __name__ == "__main__":
    main()
