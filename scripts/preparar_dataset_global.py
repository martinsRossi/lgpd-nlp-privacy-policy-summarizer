"""
Script para preparar o dataset global unificado com metadados
Converte lgpd_rotulado.csv → lgpd_rotulado_global.csv
"""

import pandas as pd
from datetime import datetime
from pathlib import Path

def preparar_dataset_global():
    """Prepara dataset global com colunas de metadados"""
    
    # Carregar dataset atual
    df = pd.read_csv("data/lgpd_rotulado.csv")
    
    print(f"Dataset original: {len(df)} exemplos")
    print(f"Colunas atuais: {list(df.columns)}")
    
    # Adicionar colunas de metadados
    df['empresa_origem'] = 'diversos'  # Exemplos genéricos da versão inicial
    df['data_contribuicao'] = datetime(2025, 1, 1).isoformat()  # Data base
    df['versao_modelo'] = 'v1.0'  # Versão inicial
    
    # Salvar dataset global
    df.to_csv("data/lgpd_rotulado_global.csv", index=False)
    
    print(f"\nDataset global criado: data/lgpd_rotulado_global.csv")
    print(f"Novas colunas: {list(df.columns)}")
    print(f"Total de exemplos: {len(df)}")
    
    # Mostrar amostra
    print("\nAmostra do dataset global:")
    print(df.head(3).to_string())
    
    # Estatísticas por categoria
    print("\nDistribuição por categoria:")
    print(df['categoria'].value_counts())

if __name__ == "__main__":
    preparar_dataset_global()
