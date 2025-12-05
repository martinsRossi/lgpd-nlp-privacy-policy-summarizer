"""
Script para treinar nova versão do classificador LGPD global

Este script treina o classificador usando o dataset global unificado e salva
uma nova versão do modelo com metadados completos.

Uso:
    python scripts/treinar_classificador_global.py --versao v1.1
    
    Ou com parâmetros:
    python scripts/treinar_classificador_global.py --versao v1.1 --test-size 0.2 --cv 5
"""

import pandas as pd
import argparse
import json
from datetime import datetime
from pathlib import Path
from loguru import logger
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.classificador_lgpd import ClassificadorLGPD, CATEGORIAS_LGPD


def carregar_dataset_global():
    """Carrega o dataset global unificado"""
    caminho = Path("data/lgpd_rotulado_global.csv")
    
    if not caminho.exists():
        logger.error(f"Dataset global não encontrado: {caminho}")
        logger.info("Execute primeiro: python scripts/preparar_dataset_global.py")
        return None
    
    df = pd.read_csv(caminho)
    logger.success(f"Dataset global carregado: {len(df)} exemplos")
    
    # Estatísticas
    logger.info(f"\nDistribuição por categoria:")
    print(df['categoria'].value_counts())
    
    logger.info(f"\nDistribuição por empresa:")
    print(df['empresa_origem'].value_counts())
    
    logger.info(f"\nDistribuição por versão:")
    print(df['versao_modelo'].value_counts())
    
    return df


def treinar_nova_versao(
    versao: str,
    test_size: float = 0.2,
    cv_folds: int = 5,
    modelo_tipo: str = 'logistic'
):
    """
    Treina uma nova versão do classificador global
    
    Args:
        versao: Versão do modelo (ex: 'v1.1', 'v2.0')
        test_size: Proporção do conjunto de teste
        cv_folds: Número de folds para cross-validation
        modelo_tipo: Tipo do modelo ('logistic', 'random_forest', 'naive_bayes')
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"TREINANDO CLASSIFICADOR LGPD GLOBAL - VERSÃO {versao}")
    logger.info(f"{'='*70}\n")
    
    # 1. Carregar dataset
    df = carregar_dataset_global()
    if df is None:
        return False
    
    # 2. Preparar dados
    X = df['texto'].tolist()
    y = df['categoria'].tolist()
    
    logger.info(f"\nPreparando dados...")
    logger.info(f"   - Total de exemplos: {len(X)}")
    logger.info(f"   - Categorias únicas: {len(set(y))}")
    logger.info(f"   - Empresas únicas: {df['empresa_origem'].nunique()}")
    
    # 3. Inicializar classificador
    logger.info(f"\nInicializando classificador ({modelo_tipo})...")
    classificador = ClassificadorLGPD(modelo_tipo=modelo_tipo)
    
    # 4. Treinar com split
    logger.info(f"\nTreinando modelo...")
    logger.info(f"   - Test size: {test_size*100:.0f}%")
    logger.info(f"   - Random state: 42")
    
    resultado = classificador.treinar(
        textos=X,
        categorias=y,
        test_size=test_size,
        validacao=True
    )
    
    logger.success(f"\nTreinamento concluído!")
    logger.info(f"   - Acurácia treino: {resultado['acuracia_treino']:.3f}")
    logger.info(f"   - Acurácia teste: {resultado['acuracia_teste']:.3f}")
    
    # 5. Cross-validation
    logger.info(f"\nExecutando cross-validation ({cv_folds} folds)...")
    
    X_vec = classificador.vectorizer.transform(X)
    y_encoded = classificador.label_encoder.transform(y)
    
    cv_scores = cross_val_score(
        classificador.modelo,
        X_vec,
        y_encoded,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring='accuracy'
    )
    
    logger.success(f"   - CV Scores: {[f'{s:.3f}' for s in cv_scores]}")
    logger.success(f"   - CV Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    # 6. Relatório de classificação
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    y_pred = [classificador.classificar(texto)['categoria'] for texto in X_test]
    
    logger.info(f"\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # 7. Salvar modelo
    caminho_modelo = Path(f"models/classificador_lgpd_{versao}.pkl")
    caminho_modelo.parent.mkdir(exist_ok=True)
    
    joblib.dump({
        'modelo': classificador.modelo,
        'vectorizer': classificador.vectorizer,
        'label_encoder': classificador.label_encoder,
        'modelo_tipo': classificador.modelo_tipo
    }, caminho_modelo)
    
    logger.success(f"\nModelo salvo: {caminho_modelo}")
    
    # 8. Salvar metadados
    metadados = {
        'versao': versao,
        'data_treinamento': datetime.now().isoformat(),
        'num_exemplos_total': len(X),
        'num_empresas': df['empresa_origem'].nunique(),
        'empresas': list(df['empresa_origem'].unique()),
        'num_categorias': len(set(y)),
        'categorias': list(CATEGORIAS_LGPD.keys()),
        'acuracia_treino': float(resultado['acuracia_treino']),
        'acuracia_teste': float(resultado['acuracia_teste']),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'cv_scores': [float(s) for s in cv_scores],
        'test_size': test_size,
        'cv_folds': cv_folds,
        'algoritmo': modelo_tipo,
        'hyperparametros': {
            'max_features_tfidf': 5000,
            'ngram_range': '(1,2)',
            'random_state': 42
        },
        'distribuicao_categorias': df['categoria'].value_counts().to_dict(),
        'distribuicao_empresas': df['empresa_origem'].value_counts().to_dict()
    }
    
    caminho_metadata = Path(f"models/metadata/classificador_{versao}_metadata.json")
    caminho_metadata.parent.mkdir(exist_ok=True, parents=True)
    
    with open(caminho_metadata, 'w', encoding='utf-8') as f:
        json.dump(metadados, f, indent=2, ensure_ascii=False)
    
    logger.success(f"Metadados salvos: {caminho_metadata}")
    
    # 9. Resumo final
    logger.info(f"\n{'='*70}")
    logger.info(f"VERSÃO {versao} TREINADA COM SUCESSO!")
    logger.info(f"{'='*70}")
    logger.info(f"Estatísticas:")
    logger.info(f"   - Exemplos: {len(X)}")
    logger.info(f"   - Empresas: {df['empresa_origem'].nunique()}")
    logger.info(f"   - Acurácia: {resultado['acuracia_teste']:.1%}")
    logger.info(f"   - CV Score: {cv_scores.mean():.1%} ±{cv_scores.std()*2:.1%}")
    logger.info(f"\nArquivos gerados:")
    logger.info(f"   - {caminho_modelo}")
    logger.info(f"   - {caminho_metadata}")
    logger.info(f"\nPróximos passos:")
    logger.info(f"   1. Revisar métricas e relatório de classificação")
    logger.info(f"   2. Testar o modelo em políticas novas")
    logger.info(f"   3. Se performance for satisfatória, atualizar produção")
    logger.info(f"   4. Continuar coletando exemplos para versão futura")
    
    return True


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Treinar nova versão do classificador LGPD global"
    )
    parser.add_argument('--versao', type=str, required=True, help="Versão do modelo (ex: v1.1)")
    parser.add_argument('--test-size', type=float, default=0.2, help="Proporção de teste (0.1-0.4)")
    parser.add_argument('--cv', type=int, default=5, help="Folds de cross-validation")
    parser.add_argument('--modelo', type=str, default='logistic', 
                       choices=['logistic', 'random_forest', 'naive_bayes'],
                       help="Tipo de modelo")
    
    args = parser.parse_args()
    
    # Validar versão
    if not args.versao.startswith('v'):
        logger.error("Versão deve começar com 'v' (ex: v1.1, v2.0)")
        return 1
    
    # Treinar
    sucesso = treinar_nova_versao(
        versao=args.versao,
        test_size=args.test_size,
        cv_folds=args.cv,
        modelo_tipo=args.modelo
    )
    
    return 0 if sucesso else 1


if __name__ == "__main__":
    exit(main())
