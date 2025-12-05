"""
Script para aumentar o dataset LGPD de 78 para 150+ exemplos

Estratégias:
1. Parafrasear exemplos existentes (sinônimos)
2. Criar variações de sentenças
3. Adicionar exemplos de outras políticas conhecidas

Meta: 12-15 exemplos por categoria (150-180 exemplos total)
"""

import pandas as pd
import random

# Carregar dataset atual
df_original = pd.read_csv('data/lgpd_rotulado.csv')
print(f"Dataset original: {len(df_original)} exemplos")

# Dicionários de sinônimos para data augmentation
SINONIMOS = {
    # Verbos de coleta
    'coletamos': ['obtemos', 'capturamos', 'registramos', 'reunimos', 'recolhemos'],
    'utilizamos': ['usamos', 'empregamos', 'aplicamos', 'processamos'],
    'compartilhamos': ['divulgamos', 'transmitimos', 'fornecemos', 'repassamos'],
    'armazenamos': ['guardamos', 'mantemos', 'conservamos', 'retemos'],
    'protegemos': ['salvaguardamos', 'resguardamos', 'preservamos'],
    
    # Substantivos
    'dados': ['informações', 'dados pessoais', 'informações pessoais'],
    'dados pessoais': ['informações pessoais', 'dados', 'informações de identificação'],
    'usuário': ['titular', 'cliente', 'usuário da plataforma'],
    'usuários': ['titulares', 'clientes', 'usuários da plataforma'],
    'serviços': ['produtos e serviços', 'funcionalidades', 'recursos'],
    'plataforma': ['site', 'aplicativo', 'sistema'],
    'empresa': ['organização', 'companhia', 'sociedade'],
    
    # Adjetivos
    'pessoais': ['individuais', 'privados', 'de identificação'],
    
    # Frases
    'entre em contato': ['comunique-se', 'contacte', 'envie mensagem'],
    'tem o direito': ['possui o direito', 'pode solicitar', 'está autorizado'],
}

def parafrasear(texto, categoria):
    """Gera variação parafraseada do texto"""
    texto_novo = texto
    
    # Aplicar substituições de sinônimos aleatoriamente
    for palavra, sinonimos in SINONIMOS.items():
        if palavra.lower() in texto_novo.lower():
            if random.random() > 0.5:  # 50% chance de substituir
                sinonimo = random.choice(sinonimos)
                # Manter capitalização
                if palavra[0].isupper():
                    sinonimo = sinonimo.capitalize()
                texto_novo = texto_novo.replace(palavra, sinonimo)
    
    return texto_novo

# Novos exemplos específicos por categoria (baseados em outras políticas BR)
NOVOS_EXEMPLOS = {
    'alteracoes': [
        "Reservamo-nos o direito de modificar esta política a qualquer momento.",
        "Atualizações desta política serão publicadas nesta página.",
        "Notificaremos você sobre mudanças significativas em nossa política.",
        "Esta política está sujeita a alterações sem aviso prévio.",
        "Revisões desta política entram em vigor imediatamente após publicação.",
        "Recomendamos verificar periodicamente esta política para atualizações.",
    ],
    
    'armazenamento': [
        "Seus dados são mantidos por período determinado pela legislação.",
        "Retemos informações pelo tempo necessário para cumprir finalidades declaradas.",
        "Dados são armazenados em servidores seguros localizados no Brasil.",
        "O período de retenção varia conforme o tipo de dado coletado.",
        "Informações são mantidas enquanto sua conta estiver ativa.",
        "Após exclusão da conta, dados são removidos em até 90 dias.",
    ],
    
    'coleta_dados': [
        "Registramos informações quando você cria uma conta em nossa plataforma.",
        "Capturamos dados fornecidos voluntariamente durante o cadastro.",
        "Obtemos informações automaticamente através de cookies e tecnologias similares.",
        "Coletamos endereço IP e dados de navegação para análise de uso.",
        "Seus dados de localização podem ser coletados com seu consentimento.",
        "Informações de dispositivo são registradas automaticamente ao acessar nosso site.",
        "Dados de transações são coletados para processamento de pedidos.",
    ],
    
    'compartilhamento': [
        "Seus dados podem ser compartilhados com prestadores de serviço contratados.",
        "Informações são transmitidas a parceiros apenas quando necessário.",
        "Podemos divulgar dados em resposta a ordens judiciais.",
        "Compartilhamos informações agregadas e anonimizadas para fins estatísticos.",
        "Dados podem ser repassados em caso de fusão ou aquisição da empresa.",
        "Não vendemos suas informações pessoais a terceiros.",
    ],
    
    'contato': [
        "Para dúvidas sobre privacidade, contate nosso DPO através do email.",
        "Envie suas solicitações para privacidade@empresa.com.br.",
        "O encarregado de dados pode ser contatado pelo telefone indicado.",
        "Dúvidas podem ser enviadas através do formulário de contato.",
        "Nossa equipe de privacidade responderá em até 15 dias úteis.",
    ],
    
    'cookies': [
        "Utilizamos cookies essenciais para funcionamento do site.",
        "Cookies de análise nos ajudam a entender como você usa a plataforma.",
        "Você pode gerenciar preferências de cookies nas configurações do navegador.",
        "Cookies de terceiros podem ser utilizados para publicidade direcionada.",
        "Pixels e web beacons são empregados para análise de campanhas.",
        "Tecnologias de rastreamento melhoram sua experiência de navegação.",
    ],
    
    'criancas': [
        "Não coletamos intencionalmente dados de menores de 13 anos.",
        "Pais ou responsáveis devem autorizar o uso por menores de idade.",
        "Consentimento parental é obrigatório para usuários menores de 18 anos.",
        "Verificamos idade durante o cadastro para proteção de menores.",
        "Dados de crianças são tratados com cuidado especial conforme legislação.",
    ],
    
    'direitos_titular': [
        "Você pode solicitar acesso aos seus dados pessoais a qualquer momento.",
        "Possui o direito de corrigir informações incompletas ou desatualizadas.",
        "Pode requerer a exclusão de seus dados, salvo obrigações legais.",
        "Tem direito à portabilidade de dados para outro fornecedor de serviço.",
        "Pode revogar consentimentos concedidos anteriormente.",
        "Solicitações de direitos do titular são respondidas em até 15 dias.",
        "Você pode se opor ao tratamento de dados para finalidades específicas.",
    ],
    
    'finalidade': [
        "Seus dados são usados para melhorar nossos produtos e serviços.",
        "Processamos informações para personalizar sua experiência.",
        "Dados são tratados para cumprimento de obrigações contratuais.",
        "Utilizamos informações para prevenir fraudes e garantir segurança.",
        "O tratamento visa enviar comunicações sobre produtos de seu interesse.",
        "Dados são processados para fins estatísticos e analíticos.",
    ],
    
    'internacional': [
        "Dados podem ser transferidos para servidores em outros países.",
        "Transferências internacionais seguem cláusulas contratuais padrão.",
        "Garantimos proteção adequada em transferências para fora do Brasil.",
        "Alguns prestadores de serviço estão localizados nos Estados Unidos.",
        "Transferências são realizadas apenas com salvaguardas apropriadas.",
    ],
    
    'outros': [
        "Esta política complementa nossos Termos de Uso.",
        "Mantemos a confidencialidade de suas informações.",
        "Auditamos regularmente nossas práticas de privacidade.",
        "Estamos comprometidos com a transparência no tratamento de dados.",
    ],
    
    'seguranca': [
        "Implementamos medidas técnicas e organizacionais de segurança.",
        "Utilizamos criptografia para proteger dados em trânsito.",
        "Acesso a informações é restrito a colaboradores autorizados.",
        "Realizamos auditorias de segurança regularmente.",
        "Sistemas são monitorados continuamente contra ameaças.",
        "Empregamos firewalls e sistemas de detecção de intrusão.",
    ],
}

# Criar lista de novos exemplos
novos_exemplos = []

# 1. Adicionar exemplos específicos novos
for categoria, exemplos in NOVOS_EXEMPLOS.items():
    for exemplo in exemplos:
        novos_exemplos.append({
            'texto': exemplo,
            'categoria': categoria
        })

print(f"Adicionados {len(novos_exemplos)} exemplos novos específicos")

# 2. Parafrasear exemplos existentes (2x cada)
parafraseados = []
for _, row in df_original.iterrows():
    # Gerar 2 paráfrases de cada exemplo
    for i in range(2):
        texto_parafraseado = parafrasear(row['texto'], row['categoria'])
        # Só adicionar se realmente mudou
        if texto_parafraseado != row['texto']:
            parafraseados.append({
                'texto': texto_parafraseado,
                'categoria': row['categoria']
            })

print(f"Gerados {len(parafraseados)} exemplos parafraseados")

# 3. Combinar tudo
df_novos = pd.DataFrame(novos_exemplos + parafraseados)
df_expandido = pd.concat([df_original, df_novos], ignore_index=True)

# Remover duplicatas exatas
df_expandido = df_expandido.drop_duplicates(subset=['texto'], keep='first')

# Distribuição final
print(f"\nDataset expandido: {len(df_expandido)} exemplos")
print(f"Aumento: +{len(df_expandido) - len(df_original)} exemplos")
print(f"\nDistribuição final por categoria:")
print(df_expandido['categoria'].value_counts().sort_index())

# Salvar
df_expandido.to_csv('data/lgpd_rotulado_expandido.csv', index=False)
print(f"\nSalvo em: data/lgpd_rotulado_expandido.csv")

# Estatísticas
print(f"\nEstatísticas:")
print(f"- Mínimo por categoria: {df_expandido['categoria'].value_counts().min()}")
print(f"- Máximo por categoria: {df_expandido['categoria'].value_counts().max()}")
print(f"- Média por categoria: {len(df_expandido) / df_expandido['categoria'].nunique():.1f}")
