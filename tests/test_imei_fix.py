"""
Teste para verificar que 'imei' não é detectado em 'primeira'
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.analisador_consumidor import AnalisadorConsumidor

def test_imei_primeira():
    """Testa que 'imei' não é detectado incorretamente em 'primeira'"""
    
    analisador = AnalisadorConsumidor()
    
    # Texto com 'primeira' mas SEM 'imei'
    texto_primeira = """
    Esta é a primeira vez que coletamos seus dados.
    Na primeira compra, solicitamos seu nome e email.
    O primeiro acesso requer cadastro.
    """
    
    # Texto com 'imei' DE VERDADE
    texto_imei = """
    Coletamos o IMEI do seu dispositivo móvel.
    O número IMEI é usado para identificação.
    Armazenamos dados como IMEI e número de série.
    """
    
    # Teste 1: 'primeira' NÃO deve detectar 'imei'
    dados1, sensiveis1 = analisador.extrair_dados_coletados(texto_primeira, {})
    dados1_str = " ".join(dados1).lower()
    
    print("=" * 70)
    print("TESTE 1: Texto com 'primeira' (NÃO deve detectar IMEI)")
    print("=" * 70)
    print(f"Texto: {texto_primeira[:100]}...")
    print(f"\nDados detectados: {dados1}")
    print(f"\n✅ PASSOU" if 'imei' not in dados1_str else f"\n❌ FALHOU - IMEI detectado incorretamente!")
    
    # Teste 2: 'imei' DE VERDADE deve ser detectado
    dados2, sensiveis2 = analisador.extrair_dados_coletados(texto_imei, {})
    dados2_str = " ".join(dados2).lower()
    
    print("\n" + "=" * 70)
    print("TESTE 2: Texto com 'IMEI' real (DEVE detectar IMEI)")
    print("=" * 70)
    print(f"Texto: {texto_imei[:100]}...")
    print(f"\nDados detectados: {dados2}")
    print(f"\n✅ PASSOU" if 'imei' in dados2_str else f"\n❌ FALHOU - IMEI NÃO detectado!")
    
    # Resultado final
    print("\n" + "=" * 70)
    teste1_ok = 'imei' not in dados1_str
    teste2_ok = 'imei' in dados2_str
    
    if teste1_ok and teste2_ok:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("✅ 'primeira' não detecta IMEI incorretamente")
        print("✅ 'IMEI' real é detectado corretamente")
        return True
    else:
        print("❌ ALGUNS TESTES FALHARAM")
        if not teste1_ok:
            print("   - 'primeira' ainda detecta IMEI (BUG)")
        if not teste2_ok:
            print("   - 'IMEI' real não está sendo detectado")
        return False

if __name__ == "__main__":
    sucesso = test_imei_primeira()
    exit(0 if sucesso else 1)
