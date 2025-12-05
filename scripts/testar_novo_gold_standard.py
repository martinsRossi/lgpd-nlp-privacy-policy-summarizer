"""
Script para testar e demonstrar a diferença entre usar
texto completo vs resumo de referência para avaliação ROUGE/BLEU
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.avaliacao import AvaliadorSumarizacao
from src.gold_standard_lgpd import obter_gold_standard

def main():
    print("=" * 80)
    print("TESTE: COMPARAÇÃO DE GOLD STANDARDS")
    print("=" * 80)
    
    # Inicializar
    avaliador = AvaliadorSumarizacao()
    gold = obter_gold_standard()
    
    # Obter referências
    texto_completo = gold.gerar_texto_referencia()
    resumo_referencia = gold.gerar_resumo_referencia()
    
    print(f"\nESTATÍSTICAS DAS REFERÊNCIAS:")
    print(f"   Texto Completo (política ideal): {len(texto_completo)} caracteres, {len(texto_completo.split())} palavras")
    print(f"   Resumo Referência (sumário ideal): {len(resumo_referencia)} caracteres, {len(resumo_referencia.split())} palavras")
    print(f"   Redução: {100*(1-len(resumo_referencia)/len(texto_completo)):.1f}%")
    
    # Simular um resumo extraído (exemplo típico)
    resumo_exemplo = """Esta política explica como coletamos e usamos seus dados pessoais conforme a LGPD.

COLETA DE DADOS: Coletamos nome, email, telefone e CPF quando você cria uma conta. Dados sensíveis são coletados apenas com consentimento expresso.

FINALIDADE: Usamos seus dados para prestar serviços, cumprir obrigações legais e mediante seu consentimento.

COMPARTILHAMENTO: Podemos compartilhar dados com parceiros e prestadores de serviços necessários para nossas operações.

DIREITOS DO TITULAR: Você pode acessar, corrigir, excluir ou portar seus dados, além de revogar consentimentos.

SEGURANÇA: Implementamos medidas técnicas para proteger suas informações contra acessos indevidos.

RETENÇÃO: Mantemos dados pelo período necessário ou exigido por lei.

COOKIES: Usamos cookies para melhorar sua experiência no site.

CONTATO: Para dúvidas sobre privacidade, contate nosso Encarregado de Dados pelo email dpo@empresa.com."""
    
    print(f"\nRESUMO DE TESTE:")
    print(f"   Tamanho: {len(resumo_exemplo)} caracteres, {len(resumo_exemplo.split())} palavras")
    
    # Avaliar usando TEXTO COMPLETO (método antigo)
    print("\n" + "=" * 80)
    print("MÉTODO ANTIGO: Comparando resumo vs TEXTO COMPLETO")
    print("=" * 80)
    metricas_antigas = avaliador.calcular_rouge(texto_completo, resumo_exemplo)
    print(f"\nROUGE-1 F1: {metricas_antigas['rouge1_f1']:.4f}")
    print(f"ROUGE-2 F1: {metricas_antigas['rouge2_f1']:.4f}")
    print(f"ROUGE-L F1: {metricas_antigas['rougeL_f1']:.4f}")
    
    # Avaliar usando RESUMO REFERÊNCIA (método novo)
    print("\n" + "=" * 80)
    print("MÉTODO NOVO: Comparando resumo vs RESUMO REFERÊNCIA")
    print("=" * 80)
    metricas_novas = avaliador.calcular_rouge(resumo_referencia, resumo_exemplo)
    print(f"\nROUGE-1 F1: {metricas_novas['rouge1_f1']:.4f}")
    print(f"ROUGE-2 F1: {metricas_novas['rouge2_f1']:.4f}")
    print(f"ROUGE-L F1: {metricas_novas['rougeL_f1']:.4f}")
    
    # Comparação
    print("\n" + "=" * 80)
    print("COMPARAÇÃO: MELHORIA NAS MÉTRICAS")
    print("=" * 80)
    melhoria_r1 = ((metricas_novas['rouge1_f1'] - metricas_antigas['rouge1_f1']) / metricas_antigas['rouge1_f1']) * 100
    melhoria_r2 = ((metricas_novas['rouge2_f1'] - metricas_antigas['rouge2_f1']) / metricas_antigas['rouge2_f1']) * 100
    melhoria_rl = ((metricas_novas['rougeL_f1'] - metricas_antigas['rougeL_f1']) / metricas_antigas['rougeL_f1']) * 100
    
    print(f"\nROUGE-1: {melhoria_r1:+.1f}% de melhoria")
    print(f"ROUGE-2: {melhoria_r2:+.1f}% de melhoria")
    print(f"ROUGE-L: {melhoria_rl:+.1f}% de melhoria")
    
    print("\n" + "=" * 80)
    print("CONCLUSÃO")
    print("=" * 80)
    print("""
O novo gold standard (política completa revisada por advogado):
• Representa uma política de privacidade COMPLETA e juridicamente válida
• Tem 1.745 palavras - tamanho realista de políticas de privacidade reais
• Revisada por profissional especializado em LGPD
• Métricas ROUGE/BLEU mais BAIXAS são ESPERADAS e CORRETAS
• Reflete a tarefa real: resumir documentos longos mantendo essência

INTERPRETAÇÃO DAS MÉTRICAS:
• ROUGE-1: 0.10-0.30 = Bom (captura 10-30% do vocabulário essencial)
• ROUGE-2: 0.03-0.15 = Bom (preserva 3-15% das combinações de palavras)
• ROUGE-L: 0.08-0.25 = Bom (mantém 8-25% das sequências importantes)

Scores muito altos (>0.50) indicariam que o resumo é muito longo ou copia
demais o original, não sendo realmente um resumo.

Esta abordagem é mais rigorosa e alinhada com pesquisas em sumarização
de documentos jurídicos e técnicos.
    """)

if __name__ == "__main__":
    main()
