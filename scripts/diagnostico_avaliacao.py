"""
Script de Diagnóstico para Avaliar Métricas Baixas
Compara sumário gerado com gold standard e identifica problemas
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.avaliacao import AvaliadorSumarizacao

def diagnosticar_avaliacao():
    """Diagnóstico completo de avaliação"""
    
    print("=" * 80)
    print(" DIAGNÓSTICO DE AVALIAÇÃO ROUGE/BLEU")
    print("=" * 80)
    print()
    
    # Carregar gold standard
    gold_path = Path("data/gold_standard_extrativo_shopee.txt")
    if not gold_path.exists():
        print(" Gold standard não encontrado!")
        return
    
    with open(gold_path, 'r', encoding='utf-8') as f:
        gold_standard = f.read()
    
    # Extrair apenas o conteúdo (antes dos comentários)
    if "" in gold_standard:
        gold_standard = gold_standard.split("")[0].strip()
    
    print(" GOLD STANDARD CARREGADO")
    print(f"   Caracteres: {len(gold_standard)}")
    print(f"   Palavras: {len(gold_standard.split())}")
    print(f"   Sentenças: {gold_standard.count('. ')}")
    print()
    
    # Solicitar sumário do usuário
    print(" COLE SEU SUMÁRIO GERADO ABAIXO (pressione CTRL+D ou CTRL+Z+Enter quando terminar):")
    print("-" * 80)
    
    try:
        import sys
        if sys.platform == 'win32':
            print("(Windows: Cole o texto e pressione CTRL+Z seguido de Enter)")
        else:
            print("(Linux/Mac: Cole o texto e pressione CTRL+D)")
        print()
        
        linhas = []
        while True:
            try:
                linha = input()
                linhas.append(linha)
            except EOFError:
                break
        
        sumario_usuario = '\n'.join(linhas).strip()
        
    except KeyboardInterrupt:
        print("\n\n Diagnóstico cancelado!")
        return
    
    if not sumario_usuario:
        print("\n Nenhum sumário fornecido!")
        return
    
    print()
    print("-" * 80)
    print(" SUMÁRIO DO USUÁRIO RECEBIDO")
    print(f"   Caracteres: {len(sumario_usuario)}")
    print(f"   Palavras: {len(sumario_usuario.split())}")
    print(f"   Sentenças: {sumario_usuario.count('. ')}")
    print()
    
    # Avaliar
    print(" CALCULANDO MÉTRICAS...")
    avaliador = AvaliadorSumarizacao()
    metricas = avaliador.avaliar_sumarizacao(gold_standard, sumario_usuario)
    
    print()
    print("=" * 80)
    print(" RESULTADOS DA AVALIAÇÃO")
    print("=" * 80)
    print()
    
    print(" MÉTRICAS ROUGE")
    print(f"   ROUGE-1: P={metricas['rouge1_precision']:.3f} | R={metricas['rouge1_recall']:.3f} | F1={metricas['rouge1_f1']:.3f}")
    print(f"   ROUGE-2: P={metricas['rouge2_precision']:.3f} | R={metricas['rouge2_recall']:.3f} | F1={metricas['rouge2_f1']:.3f}")
    print(f"   ROUGE-L: P={metricas['rougeL_precision']:.3f} | R={metricas['rougeL_recall']:.3f} | F1={metricas['rougeL_f1']:.3f}")
    print()
    
    print(" MÉTRICAS BLEU")
    print(f"   BLEU-1: {metricas['bleu1']:.3f}")
    print(f"   BLEU-2: {metricas['bleu2']:.3f}")
    print(f"   BLEU-3: {metricas['bleu3']:.3f}")
    print(f"   BLEU-4: {metricas['bleu4']:.3f}")
    print()
    
    # Análise
    print("=" * 80)
    print(" ANÁLISE E DIAGNÓSTICO")
    print("=" * 80)
    print()
    
    palavras_gold = len(gold_standard.split())
    palavras_usuario = len(sumario_usuario.split())
    taxa_compressao = palavras_usuario / palavras_gold if palavras_gold > 0 else 0
    
    print(f" Taxa de Compressão: {taxa_compressao:.1%}")
    print(f"   Gold Standard: {palavras_gold} palavras")
    print(f"   Seu Sumário: {palavras_usuario} palavras")
    print()
    
    # Diagnóstico baseado em ROUGE-2
    rouge2 = metricas['rouge2_f1']
    
    if rouge2 >= 0.35:
        print(" EXCELENTE! Métricas dentro do esperado (ROUGE-2 ≥ 0.35)")
        print("   Seu sumário está muito próximo do gold standard.")
    elif rouge2 >= 0.25:
        print(" MODERADO! Métricas razoáveis mas abaixo do ideal (ROUGE-2: 0.25-0.35)")
        print("   Possíveis causas:")
        print("   - Método diferente do usado no gold standard (TextRank)")
        print("   - Taxa de compressão diferente")
        print("   - Algumas sentenças diferentes foram selecionadas")
    elif rouge2 >= 0.15:
        print(" BAIXO! Métricas significativamente abaixo do esperado (ROUGE-2: 0.15-0.25)")
        print("   Possíveis causas:")
        print("   1. Você colou o TEXTO ORIGINAL ao invés do sumário gerado")
        print("   2. Usou método muito diferente (ex: abstrativo vs extrativo)")
        print("   3. Taxa de compressão muito diferente (ex: 0.1 vs 0.3)")
        print("   4. Colou apenas PARTE do sumário")
    else:
        print(" MUITO BAIXO! Métricas críticas (ROUGE-2 < 0.15)")
        print("   PROBLEMA DETECTADO - Possíveis causas:")
        print("   1.  Você colou o TEXTO ORIGINAL COMPLETO ao invés do sumário")
        print("   2.  Colou texto de OUTRA empresa (não Shopee)")
        print("   3.  O sumário gerado está VAZIO ou muito curto")
        print("   4.  Erro no processamento do texto")
    print()
    
    # Análise de sobreposição
    tokens_gold = set(gold_standard.lower().split())
    tokens_usuario = set(sumario_usuario.lower().split())
    intersecao = tokens_gold.intersection(tokens_usuario)
    sobreposicao = len(intersecao) / len(tokens_gold) if tokens_gold else 0
    
    print(f" Sobreposição de Palavras: {sobreposicao:.1%}")
    print(f"   Palavras únicas no Gold: {len(tokens_gold)}")
    print(f"   Palavras únicas no Seu: {len(tokens_usuario)}")
    print(f"   Palavras em comum: {len(intersecao)}")
    print()
    
    if sobreposicao < 0.3:
        print("    Sobreposição muito baixa! Textos parecem muito diferentes.")
        print("   Verifique se você está comparando os textos corretos.")
    elif sobreposicao < 0.5:
        print("    Sobreposição moderada. Textos têm alguma similaridade.")
    else:
        print("    Boa sobreposição! Textos são similares.")
    print()
    
    # Primeiras palavras
    print(" COMPARAÇÃO VISUAL (primeiras 100 palavras)")
    print()
    print("GOLD STANDARD:")
    print(' '.join(gold_standard.split()[:100]) + "...")
    print()
    print("SEU SUMÁRIO:")
    print(' '.join(sumario_usuario.split()[:100]) + "...")
    print()
    
    print("=" * 80)
    print(" RECOMENDAÇÕES")
    print("=" * 80)
    print()
    
    if rouge2 < 0.25:
        print("1.  Verifique se você colou o SUMÁRIO GERADO (não o texto original)")
        print("2.  Use o método TextRank com taxa de redução 0.3")
        print("3.  Compare o gold standard com seu sumário visualmente")
        print("4.  Certifique-se de ter carregado a política da SHOPEE")
    else:
        print("1.  Continue testando outros métodos (LexRank, LSA)")
        print("2.  Compare as diferenças entre os métodos")
        print("3.  Documente os resultados para o TCC")
    print()
    
    print("=" * 80)

if __name__ == "__main__":
    diagnosticar_avaliacao()
