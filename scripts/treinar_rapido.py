"""Script rápido para treinar o modelo sem dataset completo - teste inicial"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.modelo_conformidade_lgpd import ModeloConformidadeLGPD
import numpy as np

print(">> Treinamento rapido do modelo de conformidade...")
print()

# Criar dados sintéticos mínimos
print(">> Gerando dados de treinamento sinteticos...")
textos_treino = [
    "Esta política descreve como coletamos e usamos seus dados pessoais de acordo com a LGPD. Você tem direito de acessar, corrigir e excluir seus dados. Nomeamos um Encarregado de Dados (DPO). Informamos as bases legais. Coletamos apenas dados necessários. Mantemos os dados seguros. Você pode revogar seu consentimento.",
    "Esta política descreve como coletamos e usamos seus dados pessoais de acordo com a LGPD. Você tem direito de acessar, corrigir e excluir seus dados. Nomeamos um Encarregado de Dados (DPO). Informamos as bases legais. Coletamos apenas dados necessários. Mantemos os dados seguros.",
    "Coletamos seus dados. Podemos compartilhar. Entre em contato.",
    "Política vaga sem detalhes.",
    "Esta política abrange coleta de dados, finalidades, bases legais, direitos dos titulares (acesso, correção, exclusão), segurança, DPO, compartilhamento, prazo de retenção.",
] * 30  # Repetir para ter 150 exemplos

scores_treino = [95, 85, 20, 10, 75] * 30

print(f">> Gerados {len(textos_treino)} exemplos de treinamento")
print()

# Criar e treinar modelo
print(">> Iniciando treinamento do modelo...")
modelo = ModeloConformidadeLGPD(tipo_modelo='logistic')

resultado = modelo.treinar(
    X_train=textos_treino,
    y_train=np.array(scores_treino),
    num_epocas=20
)

print()
print("="*80)
print(">> RESULTADO DO TREINAMENTO")
print("="*80)
print(f"Epocas executadas: {resultado.get('num_epocas', resultado.get('epocas_executadas', 'N/A'))}")
print(f"Acuracia final: {resultado.get('acuracia_validacao', resultado.get('acuracia_final', 0)):.2%}")
print()

# Salvar modelo
output_path = Path("models/modelo_conformidade_lgpd.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)
modelo.salvar_modelo(output_path)
print(f">> Modelo salvo em: {output_path}")
print()

# Teste
print("="*80)
print(">> TESTE DE INFERENCIA")
print("="*80)
print()

teste_conforme = "Esta política de privacidade descreve como tratamos seus dados pessoais em conformidade com a LGPD. Coletamos apenas dados necessários. Você tem direito de acessar, corrigir e excluir seus dados. Temos um Encarregado de Proteção de Dados. Mantemos medidas de segurança."
teste_nao_conforme = "Coletamos seus dados. Podemos usar como quisermos."

print(">> Teste 1 - Política CONFORME:")
resultado1 = modelo.prever_conformidade(teste_conforme)
print(f"   Score: {resultado1.score_conformidade:.0f}/100")
print(f"   Decisao: {resultado1.conformidade_binaria.upper()}")
print(f"   Recomendacao: {resultado1.recomendacao.upper()}")
print()

print(">> Teste 2 - Política NAO CONFORME:")
resultado2 = modelo.prever_conformidade(teste_nao_conforme)
print(f"   Score: {resultado2.score_conformidade:.0f}/100")
print(f"   Decisao: {resultado2.conformidade_binaria.upper()}")
print(f"   Recomendacao: {resultado2.recomendacao.upper()}")
print()

print("="*80)
print(">> TREINAMENTO CONCLUIDO!")
print("="*80)
print()
print(">> Agora voce pode usar o modelo no Streamlit app:")
print("   1. Abra o app: http://localhost:8501")
print("   2. Carregue uma politica")
print("   3. Clique em 'Conformidade LGPD'")
print("   4. Avalie a conformidade!")
