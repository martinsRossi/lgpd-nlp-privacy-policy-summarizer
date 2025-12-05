"""Script rápido para gerar dataset de conformidade sem imports pesados"""
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Importar apenas o necessário
from gold_standard_lgpd import obter_gold_standard
import random
from pathlib import Path

def gerar_dataset_rapido():
    """Gera dataset de exemplo rapidamente"""
    print("🚀 Gerando dataset de conformidade...")
    
    gold_standard = obter_gold_standard()
    requisitos = gold_standard.obter_requisitos_obrigatorios()
    
    dataset = []
    
    # Políticas CONFORMES (score 85-100)
    print("  ✓ Gerando políticas conformes...")
    for i in range(50):
        # Incluir a maioria dos requisitos
        num_requisitos = random.randint(18, len(requisitos))
        reqs_selecionados = random.sample(requisitos, num_requisitos)
        
        texto = f"Política de Privacidade Conforme #{i+1}\\n\\n"
        for req in reqs_selecionados:
            texto += f"{req.titulo}: {req.descricao}\\n\\n"
        
        score = random.randint(85, 100)
        dataset.append(f'"{texto.replace(chr(10), " ").replace(chr(34), chr(39))}",{score},1')
    
    # Políticas NÃO CONFORMES (score 10-40)
    print("  ✗ Gerando políticas não conformes...")
    for i in range(50):
        # Incluir poucos requisitos
        num_requisitos = random.randint(2, 8)
        reqs_selecionados = random.sample(requisitos, num_requisitos)
        
        texto = f"Política Vaga #{i+1}\\n\\n"
        texto += "Coletamos seus dados. Podemos compartilhar com terceiros. "
        texto += "Entre em contato para dúvidas. "
        
        for req in reqs_selecionados[:3]:  # Apenas mencionar vagamente alguns
            texto += f"{req.titulo}. "
        
        score = random.randint(10, 40)
        dataset.append(f'"{texto.replace(chr(10), " ").replace(chr(34), chr(39))}",{score},0')
    
    # Políticas PARCIALMENTE CONFORMES (score 50-70)
    print("  ⚠ Gerando políticas parcialmente conformes...")
    for i in range(50):
        # Incluir metade dos requisitos
        num_requisitos = random.randint(12, 18)
        reqs_selecionados = random.sample(requisitos, num_requisitos)
        
        texto = f"Política Parcial #{i+1}\\n\\n"
        for req in reqs_selecionados:
            texto += f"{req.titulo}: {req.descricao}\\n"
        
        score = random.randint(50, 70)
        dataset.append(f'"{texto.replace(chr(10), " ").replace(chr(34), chr(39))}",{score},0')
    
    # Embaralhar
    random.shuffle(dataset)
    
    # Salvar
    output_path = Path("data/dataset_conformidade_exemplo.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("texto,score_conformidade,classe_binaria\\n")
        for linha in dataset:
            f.write(linha + "\\n")
    
    print(f"\\nDataset gerado: {output_path}")
    print(f"   Total: {len(dataset)} exemplos")
    print(f"   Conformes: 50")
    print(f"   Parciais: 50")
    print(f"   Não conformes: 50")

if __name__ == "__main__":
    gerar_dataset_rapido()
