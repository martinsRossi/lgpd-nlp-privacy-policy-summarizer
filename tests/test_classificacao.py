#!/usr/bin/env python3
"""Teste rápido de classificação para verificar o bug de 'países'"""

import joblib
from pathlib import Path

# Carregar modelo
modelo_path = Path("models/classificador_lgpd.pkl")
print(f"Carregando modelo: {modelo_path}")
print(f"Arquivo existe: {modelo_path.exists()}")
print(f"Tamanho: {modelo_path.stat().st_size if modelo_path.exists() else 0} bytes")
print()

modelo_data = joblib.load(modelo_path)
print(f"✅ Modelo carregado com sucesso!")
print(f"Tipo de modelo: {modelo_data.get('modelo_tipo', 'N/A')}")
print()

# Criar instância do classificador
from src.classificador_lgpd import ClassificadorLGPD
classificador = ClassificadorLGPD()
classificador.modelo = modelo_data['modelo']
classificador.vectorizer = modelo_data['vectorizer']
classificador.label_encoder = modelo_data['label_encoder']

# Texto de teste que estava sendo classificado errado
texto_teste = "Se você não deseja que suas informações sejam processadas em outros países entre em contato."

print("=" * 70)
print("TESTE DE CLASSIFICAÇÃO")
print("=" * 70)
print(f"Texto: {texto_teste}")
print()

# Classificar
resultado = classificador.classificar(texto_teste)

print("Classificação:")
print(f"  Categoria: {resultado['categoria']}")
print(f"  Nome: {resultado['categoria_nome']}")
print(f"  Confiança: {resultado['confianca']:.2%}")
print(f"  Método: {resultado['metodo']}")
print()

# Verificar se 'criancas' aparece
if resultado['categoria'] == 'criancas':
    print("❌ ERRO: Categoria 'criancas' detectada!")
    print(f"   Confiança: {resultado['confianca']:.2%}")
else:
    print("✅ SUCESSO: Categoria 'criancas' NÃO foi detectada!")

# Verificar se 'internacional' aparece
if resultado['categoria'] == 'internacional':
    print(f"✅ CORRETO: Categoria 'internacional' detectada com {resultado['confianca']:.2%}")
else:
    print(f"⚠️  Categoria '{resultado['categoria']}' detectada (esperado: 'internacional')")
