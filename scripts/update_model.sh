#!/bin/bash
# Script para atualizar o modelo do classificador LGPD
# Execute este script após fazer git pull para obter a versão mais recente

echo "🔄 Atualizando modelo do classificador LGPD..."
echo ""

# Treinar nova versão
python -m scripts.treinar_classificador_global --versao v2.1_fix_criancas

# Copiar para o arquivo padrão
echo ""
echo "📦 Copiando modelo treinado..."
cp models/classificador_lgpd_v2.1_fix_criancas.pkl models/classificador_lgpd.pkl

echo ""
echo "✅ Modelo atualizado com sucesso!"
echo ""
echo "📋 Correções nesta versão:"
echo "   - Removida ambiguidade 'pais' vs 'país' na categoria crianças"
echo "   - Melhor detecção de finalidades e compartilhamentos"
echo ""
echo "▶️  Execute 'streamlit run app.py' para usar a nova versão"
