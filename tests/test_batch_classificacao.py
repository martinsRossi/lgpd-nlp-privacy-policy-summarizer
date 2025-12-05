#!/usr/bin/env python3
"""Teste de classificação em lote para identificar o problema"""

import joblib
from pathlib import Path
from src.classificador_lgpd import ClassificadorLGPD

# Carregar modelo
modelo_path = Path("models/classificador_lgpd.pkl")
modelo_data = joblib.load(modelo_path)

classificador = ClassificadorLGPD()
classificador.modelo = modelo_data['modelo']
classificador.vectorizer = modelo_data['vectorizer']
classificador.label_encoder = modelo_data['label_encoder']
classificador.treinado = True  # IMPORTANT: Mark as trained!

# Textos que estão sendo classificados incorretamente como 'criancas'
textos_problema = [
    "Se você não deseja que suas informações sejam processadas de acordo com esta Política de Privacidade em geral ou em qualquer parte dela, você não deve usar nosso Serviço.",
    "Usamos essas informações para operar, manter, melhorar e fornecer os recursos e a funcionalidade do Serviço para você, para corresponder a você e para resolver quaisquer problemas que você levantar sobre o Serviço.",
    "Os Identificadores de Dispositivos usados pelo Canva incluem o ID de Publicidade do Android e o Identificador de Publicidade do iOS.",
    "Isso ajuda o Canva a localizar e personalizar o conteúdo, cumprir as leis locais, realizar análises agregadas, entender se seus usuários usam o Canva para uso doméstico, empresarial ou educacional, melhorar a eficiência da publicidade e estimar a responsabilidade fiscal do Canva.",
    "Para melhoria do Serviço (incluindo análise e aprendizado de máquina): Podemos analisar sua atividade, conteúdo, envios de mídia e dados relacionados em sua conta para fornecer e personalizar o Serviço e treinar nossos algoritmos, modelos e produtos e serviços de IA usando aprendizado de máquina para desenvolver, melhorar e fornecer nosso Serviço.",
    "Você pode gerenciar o uso dos seus dados para treinar a IA e melhorar o nosso Serviço na página de configurações de privacidade nas suas configurações de privacidade.",
    "Essas atividades incluem, mas não estão limitadas a: rotular e detectar componentes em imagens (por exemplo, fundo, olhos) para fornecer e aprimorar ferramentas de edição de fotos, como remoção de fundo, correção de manchas e olhos vermelhos e apagamento de componentes; rotular dados individuais brutos; traduzir trilhas sonoras de áudio; prever as ofertas de assinatura ou produto mais relevantes para um usuário para personalizar comunicações e publicidade; e termos de pesquisa e dados de interação de pesquisa correspondentes para fornecer o resultado de design mais relevante.",
    "A subsidiária do Canva nos Estados Unidos adere à conformidade com o Quadro de Privacidade de Dados UE-EUA, o Quadro de Privacidade de Dados Suíça-EUA e a Extensão do Reino Unido para o Quadro de Privacidade de Dados UE-EUA.",
    "Você pode gerenciar suas preferências relacionadas a como o Canva e os provedores de serviços terceirizados com os quais colaboramos para habilitar funcionalidades no Serviço podem usar seus dados para treinamento de IA, o que inclui tecnologias de aprendizado de máquina.",
    "Para proteger a privacidade e a segurança das crianças, tomaremos medidas razoáveis para ajudar a verificar a identidade da instituição de ensino antes de conceder acesso a qualquer informação pessoal.",
    "Para proteger a privacidade e a segurança das crianças, tomaremos medidas razoáveis para ajudar a verificar a identidade do distrito e o relacionamento com a Equipe Escolar antes de conceder acesso a qualquer informação pessoal.",
    "Necessidade contratual: Precisamos disso para fornecer o Serviço a você e cumprir nossas obrigações com você nos termos de nossos Termos de Uso."
]

print("=" * 80)
print("TESTE DE CLASSIFICAÇÃO EM LOTE")
print("=" * 80)
print()

erros_criancas = []
corretos_criancas = []

for i, texto in enumerate(textos_problema, 1):
    resultado = classificador.classificar(texto)
    
    # Verificar se tem palavra relacionada a crianças
    palavras_criancas = ['criança', 'crianças', 'menor', 'menores', 'adolescente']
    tem_palavra_crianca = any(palavra in texto.lower() for palavra in palavras_criancas)
    
    print(f"{i}. {texto[:80]}...")
    print(f"   Classificado como: {resultado['categoria']} ({resultado['confianca']:.2%})")
    print(f"   Contém palavra de crianças: {tem_palavra_crianca}")
    
    if resultado['categoria'] == 'criancas':
        if tem_palavra_crianca:
            print(f"   ✅ CORRETO - Texto menciona crianças")
            corretos_criancas.append((texto, resultado['confianca']))
        else:
            print(f"   ❌ ERRO - Classificado como criancas mas não menciona crianças!")
            erros_criancas.append((texto, resultado['confianca']))
    
    print()

print("=" * 80)
print("RESUMO")
print("=" * 80)
print(f"Total de textos testados: {len(textos_problema)}")
print(f"Classificados como 'criancas' corretamente: {len(corretos_criancas)}")
print(f"Classificados como 'criancas' INCORRETAMENTE: {len(erros_criancas)}")
print()

if erros_criancas:
    print("❌ TEXTOS COM ERRO (classificados como criancas sem mencionar crianças):")
    for texto, conf in erros_criancas:
        print(f"  - {texto[:100]}... ({conf:.2%})")
    print()
    
if corretos_criancas:
    print("✅ TEXTOS CORRETOS (classificados como criancas e mencionam crianças):")
    for texto, conf in corretos_criancas:
        print(f"  - {texto[:100]}... ({conf:.2%})")
