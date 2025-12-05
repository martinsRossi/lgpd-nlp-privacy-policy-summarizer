# Script para atualizar o modelo do classificador LGPD
# Execute este script após fazer git pull para obter a versão mais recente

Write-Host "🔄 Atualizando modelo do classificador LGPD..." -ForegroundColor Cyan
Write-Host ""

# Treinar nova versão
python -m scripts.treinar_classificador_global --versao v2.2_fix_dataset

# Copiar para o arquivo padrão
Write-Host ""
Write-Host "📦 Copiando modelo treinado..." -ForegroundColor Yellow

if (Test-Path "models\classificador_lgpd_v2.2_fix_dataset.pkl") {
    Copy-Item models\classificador_lgpd_v2.2_fix_dataset.pkl models\classificador_lgpd.pkl -Force
    
    if (Test-Path "models\classificador_lgpd.pkl") {
        Write-Host ""
        Write-Host "✅ Modelo atualizado com sucesso!" -ForegroundColor Green
        Write-Host "   Arquivo: models\classificador_lgpd.pkl" -ForegroundColor Gray
    } else {
        Write-Host ""
        Write-Host "❌ ERRO: Falha ao copiar o modelo!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host ""
    Write-Host "❌ ERRO: Modelo treinado não encontrado!" -ForegroundColor Red
    Write-Host "   Esperado: models\classificador_lgpd_v2.2_fix_dataset.pkl" -ForegroundColor Gray
    exit 1
}
Write-Host ""
Write-Host "📋 Correções nesta versão:" -ForegroundColor White
Write-Host "   - Corrigido rótulo incorreto no dataset de treinamento"
Write-Host "   - Adicionados exemplos negativos para 'países' (internacional)"
Write-Host "   - Removida ambiguidade 'pais' vs 'país' na categoria crianças"
Write-Host ""
Write-Host "▶️  Execute 'streamlit run app.py' para usar a nova versão" -ForegroundColor Cyan
