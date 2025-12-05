"""
Script para exportar o Gold Standard LGPD em formato legível

Este script gera um documento formatado com todos os 23 requisitos
obrigatórios da LGPD que compõem o Gold Standard Universal usado
para avaliação de conformidade.

Uso:
    python scripts/exportar_gold_standard_lgpd.py
    
Saída:
    - docs/gold_standard_lgpd.txt (formato texto)
    - docs/gold_standard_lgpd.md (formato Markdown)
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.gold_standard_lgpd import GoldStandardLGPD
from datetime import datetime


def exportar_gold_standard():
    """Exporta o Gold Standard LGPD em formatos legíveis"""
    
    # Inicializar Gold Standard
    gold = GoldStandardLGPD()
    
    # Obter requisitos obrigatórios
    requisitos = gold.obter_requisitos_obrigatorios()
    categorias = gold.obter_categorias()
    
    print(f"Gold Standard LGPD")
    print(f"   Total de requisitos obrigatórios: {len(requisitos)}")
    print(f"   Categorias: {len(categorias)}")
    print()
    
    # Criar diretório de saída
    output_dir = Path("docs")
    output_dir.mkdir(exist_ok=True)
    
    # ==================== EXPORTAR COMO TEXTO ====================
    txt_path = output_dir / "gold_standard_lgpd.txt"
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GOLD STANDARD UNIVERSAL - LGPD (Lei nº 13.709/2018)\n")
        f.write("Sistema de Avaliação de Conformidade de Políticas de Privacidade\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Data de geração: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}\n")
        f.write(f"Total de requisitos obrigatórios: {len(requisitos)}\n")
        f.write(f"Peso total: {gold.calcular_peso_total()}\n\n")
        
        f.write("DESCRIÇÃO:\n")
        f.write("-" * 80 + "\n")
        f.write("Este Gold Standard Universal contém todos os requisitos obrigatórios que uma\n")
        f.write("política de privacidade deve atender para estar em conformidade com a LGPD.\n")
        f.write("Cada requisito possui:\n")
        f.write("  • ID único de identificação\n")
        f.write("  • Categoria LGPD correspondente\n")
        f.write("  • Descrição do requisito legal\n")
        f.write("  • Palavras-chave para detecção semântica\n")
        f.write("  • Peso relativo (importância do requisito)\n")
        f.write("  • Fundamentação legal (artigos da LGPD)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("REQUISITOS OBRIGATÓRIOS POR CATEGORIA\n")
        f.write("=" * 80 + "\n\n")
        
        # Agrupar por categoria
        for categoria in sorted(categorias):
            reqs_cat = [r for r in requisitos if r.categoria == categoria]
            
            if not reqs_cat:
                continue
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"CATEGORIA: {categoria.upper()}\n")
            f.write(f"Total de requisitos: {len(reqs_cat)}\n")
            f.write("=" * 80 + "\n\n")
            
            for req in reqs_cat:
                f.write(f"[{req.id}] {req.descricao}\n")
                f.write(f"      Peso: {req.peso}\n")
                f.write(f"      Base Legal: {req.artigo_lei}\n")
                f.write(f"      Palavras-chave: {', '.join(req.palavras_chave[:5])}...\n")
                f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("TEXTO COMPLETO DO GOLD STANDARD\n")
        f.write("=" * 80 + "\n\n")
        f.write("Abaixo está o texto completo da política modelo que atende todos os\n")
        f.write("requisitos acima. Este texto é usado como referência nas métricas ROUGE/BLEU.\n\n")
        f.write("-" * 80 + "\n\n")
        
        texto_completo = gold.gerar_texto_referencia()
        f.write(texto_completo)
        f.write("\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("FIM DO GOLD STANDARD LGPD\n")
        f.write("=" * 80 + "\n")
    
    print(f"Arquivo TXT gerado: {txt_path}")
    
    # ==================== EXPORTAR COMO MARKDOWN ====================
    md_path = output_dir / "gold_standard_lgpd.md"
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Gold Standard Universal - LGPD\n\n")
        f.write("**Sistema de Avaliação de Conformidade de Políticas de Privacidade**\n\n")
        f.write(f"**Lei nº 13.709/2018** - Lei Geral de Proteção de Dados Pessoais\n\n")
        f.write("---\n\n")
        
        f.write(f"**Data de geração:** {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}\n\n")
        f.write(f"**Total de requisitos obrigatórios:** {len(requisitos)}\n\n")
        f.write(f"**Peso total:** {gold.calcular_peso_total()}\n\n")
        
        f.write("##Descrição\n\n")
        f.write("Este Gold Standard Universal contém todos os requisitos obrigatórios que uma ")
        f.write("política de privacidade deve atender para estar em conformidade com a LGPD.\n\n")
        f.write("Cada requisito possui:\n\n")
        f.write("- ID único de identificação\n")
        f.write("- Categoria LGPD correspondente\n")
        f.write("- Descrição do requisito legal\n")
        f.write("- Palavras-chave para detecção semântica\n")
        f.write("- Peso relativo (importância do requisito)\n")
        f.write("- Fundamentação legal (artigos da LGPD)\n\n")
        
        f.write("---\n\n")
        f.write("## Requisitos Obrigatórios por Categoria\n\n")
        
        # Agrupar por categoria
        for categoria in sorted(categorias):
            reqs_cat = [r for r in requisitos if r.categoria == categoria]
            
            if not reqs_cat:
                continue
            
            f.write(f"### {categoria.replace('_', ' ').title()}\n\n")
            f.write(f"**Total de requisitos:** {len(reqs_cat)}\n\n")
            
            for i, req in enumerate(reqs_cat, 1):
                f.write(f"#### {i}. [{req.id}] {req.descricao}\n\n")
                f.write(f"- **Peso:** {req.peso}\n")
                f.write(f"- **Base Legal:** {req.artigo_lei}\n")
                f.write(f"- **Palavras-chave:** {', '.join(req.palavras_chave[:8])}")
                if len(req.palavras_chave) > 8:
                    f.write(f" (e mais {len(req.palavras_chave) - 8})")
                f.write("\n\n")
        
        f.write("---\n\n")
        f.write("## Texto Completo do Gold Standard\n\n")
        f.write("Abaixo está o texto completo da **política modelo** que atende todos os ")
        f.write("requisitos acima. Este texto é usado como referência nas métricas ROUGE/BLEU.\n\n")
        f.write("```\n")
        
        texto_completo = gold.gerar_texto_referencia()
        f.write(texto_completo)
        
        f.write("\n```\n\n")
        f.write("---\n\n")
        f.write("**Nota:** Este Gold Standard é utilizado pelo sistema automatizado para avaliar ")
        f.write("a conformidade de políticas de privacidade reais com os requisitos da LGPD.\n")
    
    print(f"Arquivo MD gerado: {md_path}")
    
    # ==================== ESTATÍSTICAS ====================
    print(f"\nEstatísticas:")
    print(f"   • Requisitos por categoria:")
    for categoria in sorted(categorias):
        count = len([r for r in requisitos if r.categoria == categoria])
        print(f"     - {categoria}: {count}")
    
    print(f"\n   • Peso por categoria:")
    for categoria in sorted(categorias):
        peso = sum(r.peso for r in requisitos if r.categoria == categoria)
        print(f"     - {categoria}: {peso}")
    
    print(f"\nDocumentos exportados com sucesso!")
    print(f"   {txt_path}")
    print(f"   {md_path}")


if __name__ == "__main__":
    exportar_gold_standard()
