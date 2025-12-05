#!/bin/bash
# Script para atualizar o modelo do classificador LGPD
# Execute este script após fazer git pull para obter a versão mais recente

echo "🔄 Atualizando modelo do classificador LGPD..."
echo ""

# Treinar nova versão
python -m scripts.treinar_classificador_global --versao v2.2_fix_dataset

# Copiar para o arquivo padrão
echo ""
echo "📦 Copiando modelo para produção..."

if [ -f "models/classificador_lgpd_v2.2_fix_dataset.pkl" ]; then
    cp models/classificador_lgpd_v2.2_fix_dataset.pkl models/classificador_lgpd.pkl
    
    if [ -f "models/classificador_lgpd.pkl" ]; then
        echo ""
        echo "✅ Modelo atualizado com sucesso!"
        echo "   Arquivo: models/classificador_lgpd.pkl"
    else
        echo ""
        echo "❌ ERRO: Falha ao copiar o modelo!"
        exit 1
    fi
else
    echo ""
    echo "❌ ERRO: Modelo treinado não encontrado!"
    echo "   Esperado: models/classificador_lgpd_v2.2_fix_dataset.pkl"
    exit 1
fi
echo ""
echo "📋 Correções nesta versão:"
echo "   - Corrigido rótulo incorreto no dataset de treinamento"
echo "   - Adicionados exemplos negativos para 'países' (internacional)"
echo "   - Removida ambiguidade 'pais' vs 'país' na categoria crianças"
echo ""
echo "▶️  Execute 'streamlit run app.py' para usar a nova versão"
