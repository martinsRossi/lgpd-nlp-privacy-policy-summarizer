"""
Módulo de Gold Standard Universal LGPD
Define o padrão ideal de conformidade com a Lei Geral de Proteção de Dados

⚠️ IMPORTANTE - DISTINÇÃO METODOLÓGICA:

Este módulo define um GOLD STANDARD FIXO DE AVALIAÇÃO, NÃO um modelo treinado.

GOLD STANDARD LGPD:
- ❌ NÃO envolve treinamento com épocas ou otimização
- ❌ NÃO é machine learning
- ✅ É um BENCHMARK FIXO baseado nos artigos da Lei 13.709/2018
- ✅ Contém 23 requisitos obrigatórios extraídos da legislação
- ✅ É usado APENAS para AVALIAÇÃO de conformidade de políticas
- ✅ Avaliação feita por COBERTURA SEMÂNTICA (keyword matching)

vs.

MODELO DE CONFORMIDADE (modelo_conformidade_lgpd.py):
- ✅ É um MODELO TREINADO com épocas e otimização
- ✅ Aprende a prever se uma política é conforme (score 0-100)
- ✅ Usa dataset sintético de 200 exemplos
- ✅ Treinado com Logistic Regression + SGDClassifier

Para metodologia completa, veja: docs/METODOLOGIA_CORRIGIDA.md
"""

from typing import Dict, List, Set
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class RequisitoLGPD:
    """Representa um requisito específico da LGPD"""
    id: str
    categoria: str
    titulo: str
    descricao: str
    artigo_lei: str
    palavras_chave: List[str]
    obrigatorio: bool = True
    peso: float = 1.0  # Peso para cálculo de score


class GoldStandardLGPD:
    """
    Gold Standard Universal representando uma política de privacidade
    ideal em conformidade com a LGPD (Lei nº 13.709/2018)
    """
    
    def __init__(self):
        """Inicializa o Gold Standard com todos os requisitos da LGPD"""
        logger.info("Inicializando Gold Standard Universal LGPD")
        
        self.requisitos: Dict[str, RequisitoLGPD] = {}
        self._carregar_requisitos()
    
    def _carregar_requisitos(self):
        """Carrega todos os requisitos obrigatórios da LGPD"""
        
        # PRINCÍPIOS FUNDAMENTAIS (Art. 6º)
        self._adicionar_requisito(RequisitoLGPD(
            id="PRINC_001",
            categoria="principios",
            titulo="Princípio da Finalidade",
            descricao="A política deve especificar claramente as finalidades legítimas, específicas, explícitas e informadas ao titular para o tratamento de dados.",
            artigo_lei="Art. 6º, I",
            palavras_chave=["finalidade", "propósito", "objetivo", "para que", "utilizamos"],
            obrigatorio=True,
            peso=2.0
        ))
        
        self._adicionar_requisito(RequisitoLGPD(
            id="PRINC_002",
            categoria="principios",
            titulo="Princípio da Adequação",
            descricao="O tratamento deve ser compatível com as finalidades informadas ao titular.",
            artigo_lei="Art. 6º, II",
            palavras_chave=["adequado", "compatível", "pertinente", "necessário"],
            obrigatorio=True,
            peso=1.5
        ))
        
        self._adicionar_requisito(RequisitoLGPD(
            id="PRINC_003",
            categoria="principios",
            titulo="Princípio da Necessidade",
            descricao="Limitação do tratamento ao mínimo necessário para a realização das finalidades.",
            artigo_lei="Art. 6º, III",
            palavras_chave=["mínimo necessário", "estritamente necessário", "indispensável"],
            obrigatorio=True,
            peso=1.5
        ))
        
        self._adicionar_requisito(RequisitoLGPD(
            id="PRINC_004",
            categoria="principios",
            titulo="Princípio da Transparência",
            descricao="Garantia de informações claras, precisas e facilmente acessíveis sobre o tratamento.",
            artigo_lei="Art. 6º, VI",
            palavras_chave=["transparente", "claro", "acessível", "informações claras"],
            obrigatorio=True,
            peso=2.0
        ))
        
        self._adicionar_requisito(RequisitoLGPD(
            id="PRINC_005",
            categoria="principios",
            titulo="Princípio da Segurança",
            descricao="Utilização de medidas técnicas e administrativas para proteger os dados de acessos não autorizados.",
            artigo_lei="Art. 6º, VII",
            palavras_chave=["segurança", "proteção", "medidas técnicas", "criptografia"],
            obrigatorio=True,
            peso=2.0
        ))
        
        # COLETA E TIPOS DE DADOS (Art. 5º e 7º)
        self._adicionar_requisito(RequisitoLGPD(
            id="COLETA_001",
            categoria="coleta_dados",
            titulo="Especificação dos Dados Coletados",
            descricao="A política deve listar claramente quais dados pessoais são coletados.",
            artigo_lei="Art. 9º, I",
            palavras_chave=["coletamos", "dados coletados", "informações coletadas", "dados pessoais"],
            obrigatorio=True,
            peso=2.0
        ))
        
        self._adicionar_requisito(RequisitoLGPD(
            id="COLETA_002",
            categoria="coleta_dados",
            titulo="Dados Sensíveis - Consentimento Específico",
            descricao="Se houver coleta de dados sensíveis (raça, religião, saúde, etc.), deve haver consentimento específico e destacado.",
            artigo_lei="Art. 5º, II e Art. 11",
            palavras_chave=["dados sensíveis", "origem racial", "religião", "saúde", "biométricos", "genéticos"],
            obrigatorio=True,
            peso=3.0
        ))
        
        self._adicionar_requisito(RequisitoLGPD(
            id="COLETA_003",
            categoria="coleta_dados",
            titulo="Dados de Crianças e Adolescentes",
            descricao="Tratamento de dados de crianças deve ser realizado com consentimento específico dos pais/responsáveis.",
            artigo_lei="Art. 14",
            palavras_chave=["crianças", "menores", "adolescentes", "responsável legal", "pais"],
            obrigatorio=False,
            peso=2.0
        ))
        
        # BASES LEGAIS (Art. 7º)
        self._adicionar_requisito(RequisitoLGPD(
            id="BASE_001",
            categoria="bases_legais",
            titulo="Base Legal para Tratamento",
            descricao="A política deve indicar a base legal que fundamenta o tratamento de dados.",
            artigo_lei="Art. 7º",
            palavras_chave=["base legal", "consentimento", "cumprimento de obrigação legal", "legítimo interesse"],
            obrigatorio=True,
            peso=2.0
        ))
        
        self._adicionar_requisito(RequisitoLGPD(
            id="BASE_002",
            categoria="bases_legais",
            titulo="Consentimento Livre e Informado",
            descricao="Quando o consentimento for a base legal, deve ser livre, informado e inequívoco.",
            artigo_lei="Art. 8º",
            palavras_chave=["consentimento", "concordo", "aceito", "autorizo", "livre e informado"],
            obrigatorio=True,
            peso=2.0
        ))
        
        # DIREITOS DO TITULAR (Art. 18)
        self._adicionar_requisito(RequisitoLGPD(
            id="DIREITO_001",
            categoria="direitos_titular",
            titulo="Direito de Acesso",
            descricao="O titular tem direito de confirmar a existência de tratamento e acessar seus dados.",
            artigo_lei="Art. 18, I e II",
            palavras_chave=["acesso aos dados", "consultar dados", "acessar informações"],
            obrigatorio=True,
            peso=2.0
        ))
        
        self._adicionar_requisito(RequisitoLGPD(
            id="DIREITO_002",
            categoria="direitos_titular",
            titulo="Direito de Correção",
            descricao="O titular pode solicitar a correção de dados incompletos, inexatos ou desatualizados.",
            artigo_lei="Art. 18, III",
            palavras_chave=["corrigir", "retificar", "atualizar dados"],
            obrigatorio=True,
            peso=1.5
        ))
        
        self._adicionar_requisito(RequisitoLGPD(
            id="DIREITO_003",
            categoria="direitos_titular",
            titulo="Direito de Exclusão",
            descricao="O titular pode solicitar a eliminação dos dados tratados com base no consentimento.",
            artigo_lei="Art. 18, VI",
            palavras_chave=["excluir", "eliminar", "apagar dados", "direito ao esquecimento"],
            obrigatorio=True,
            peso=2.0
        ))
        
        self._adicionar_requisito(RequisitoLGPD(
            id="DIREITO_004",
            categoria="direitos_titular",
            titulo="Direito de Portabilidade",
            descricao="O titular pode solicitar a portabilidade dos dados a outro fornecedor.",
            artigo_lei="Art. 18, V",
            palavras_chave=["portabilidade", "transferir dados", "exportar dados"],
            obrigatorio=True,
            peso=1.5
        ))
        
        self._adicionar_requisito(RequisitoLGPD(
            id="DIREITO_005",
            categoria="direitos_titular",
            titulo="Direito de Revogação do Consentimento",
            descricao="O titular pode revogar o consentimento a qualquer momento.",
            artigo_lei="Art. 8º, § 5º e Art. 18, IX",
            palavras_chave=["revogar", "retirar consentimento", "cancelar autorização"],
            obrigatorio=True,
            peso=2.0
        ))
        
        # COMPARTILHAMENTO E TRANSFERÊNCIA (Art. 33)
        self._adicionar_requisito(RequisitoLGPD(
            id="COMPART_001",
            categoria="compartilhamento",
            titulo="Informação sobre Compartilhamento",
            descricao="A política deve informar se os dados são compartilhados com terceiros e quem são esses terceiros.",
            artigo_lei="Art. 9º, III",
            palavras_chave=["compartilhar", "terceiros", "parceiros", "fornecedores"],
            obrigatorio=True,
            peso=2.0
        ))
        
        self._adicionar_requisito(RequisitoLGPD(
            id="COMPART_002",
            categoria="compartilhamento",
            titulo="Transferência Internacional",
            descricao="Se houver transferência internacional, deve indicar o país e as garantias de proteção.",
            artigo_lei="Art. 33",
            palavras_chave=["transferência internacional", "outros países", "exterior"],
            obrigatorio=False,
            peso=1.5
        ))
        
        # SEGURANÇA E RETENÇÃO (Art. 46 e 47)
        self._adicionar_requisito(RequisitoLGPD(
            id="SEGUR_001",
            categoria="seguranca",
            titulo="Medidas de Segurança",
            descricao="A política deve descrever as medidas técnicas e administrativas de segurança.",
            artigo_lei="Art. 46",
            palavras_chave=["medidas de segurança", "proteção", "criptografia", "controle de acesso"],
            obrigatorio=True,
            peso=2.0
        ))
        
        self._adicionar_requisito(RequisitoLGPD(
            id="SEGUR_002",
            categoria="seguranca",
            titulo="Comunicação de Incidentes",
            descricao="Deve haver menção sobre como serão comunicados incidentes de segurança.",
            artigo_lei="Art. 48",
            palavras_chave=["incidente", "vazamento", "violação", "notificação de incidente"],
            obrigatorio=True,
            peso=1.5
        ))
        
        self._adicionar_requisito(RequisitoLGPD(
            id="RETENC_001",
            categoria="retencao",
            titulo="Prazo de Retenção",
            descricao="A política deve indicar por quanto tempo os dados serão mantidos.",
            artigo_lei="Art. 15 e 16",
            palavras_chave=["prazo", "período", "retenção", "conservação", "armazenamento"],
            obrigatorio=True,
            peso=2.0
        ))
        
        # ENCARREGADO/DPO (Art. 41)
        self._adicionar_requisito(RequisitoLGPD(
            id="DPO_001",
            categoria="encarregado",
            titulo="Identificação do Encarregado (DPO)",
            descricao="A política deve identificar o encarregado pelo tratamento de dados e como contatá-lo.",
            artigo_lei="Art. 41",
            palavras_chave=["encarregado", "dpo", "proteção de dados", "contato"],
            obrigatorio=True,
            peso=2.0
        ))
        
        # COOKIES E RASTREAMENTO (Lei do Marco Civil da Internet)
        self._adicionar_requisito(RequisitoLGPD(
            id="COOKIE_001",
            categoria="cookies",
            titulo="Informação sobre Cookies",
            descricao="A política deve informar sobre o uso de cookies e tecnologias de rastreamento.",
            artigo_lei="Marco Civil da Internet",
            palavras_chave=["cookies", "rastreamento", "pixels", "web beacon"],
            obrigatorio=True,
            peso=1.0
        ))
        
        # ALTERAÇÕES DA POLÍTICA (Transparência)
        self._adicionar_requisito(RequisitoLGPD(
            id="ALTER_001",
            categoria="alteracoes",
            titulo="Alterações da Política",
            descricao="A política deve indicar como o titular será informado sobre alterações.",
            artigo_lei="Art. 9º (transparência)",
            palavras_chave=["alterações", "atualização", "modificações", "revisão"],
            obrigatorio=True,
            peso=1.0
        ))
        
        logger.success(f"Carregados {len(self.requisitos)} requisitos LGPD")
    
    def _adicionar_requisito(self, requisito: RequisitoLGPD):
        """Adiciona um requisito ao gold standard"""
        self.requisitos[requisito.id] = requisito
    
    def obter_requisitos_obrigatorios(self) -> List[RequisitoLGPD]:
        """Retorna apenas os requisitos obrigatórios"""
        return [r for r in self.requisitos.values() if r.obrigatorio]
    
    def obter_requisitos_por_categoria(self, categoria: str) -> List[RequisitoLGPD]:
        """Retorna requisitos de uma categoria específica"""
        return [r for r in self.requisitos.values() if r.categoria == categoria]
    
    def obter_categorias(self) -> Set[str]:
        """Retorna todas as categorias de requisitos"""
        return {r.categoria for r in self.requisitos.values()}
    
    def calcular_peso_total(self) -> float:
        """Calcula o peso total de todos os requisitos obrigatórios"""
        return sum(r.peso for r in self.obter_requisitos_obrigatorios())
    
    def gerar_texto_referencia(self) -> str:
        """
        Gera um texto de referência representando uma política ideal
        Este texto será usado para comparação em métricas ROUGE/BLEU
        """
        linhas = [
            "POLÍTICA DE PRIVACIDADE IDEAL CONFORME LGPD\n",
            "Esta é uma política de privacidade modelo que atende todos os requisitos da Lei Geral de Proteção de Dados (Lei nº 13.709/2018).\n"
        ]
        
        # Agrupar por categoria
        categorias = self.obter_categorias()
        
        for categoria in sorted(categorias):
            requisitos_cat = self.obter_requisitos_por_categoria(categoria)
            
            # Título da seção
            linhas.append(f"\n{categoria.upper().replace('_', ' ')}\n")
            
            # Descrição de cada requisito
            for req in requisitos_cat:
                linhas.append(f"{req.titulo}: {req.descricao}\n")
        
        texto_completo = '\n'.join(linhas)
        logger.info(f"Texto de referência gerado: {len(texto_completo)} caracteres")
        return texto_completo
    
    def gerar_resumo_referencia(self) -> str:
        """
        Gera um resumo de referência conciso e bem estruturado
        Este texto representa como um HUMANO (advogado especialista) escreveria um resumo ideal de uma política LGPD
        Usado como gold standard para métricas ROUGE/BLEU em sumarizações
        Baseado em modelo revisado por profissional jurídico
        """
        resumo_ideal = """POLÍTICA DE PRIVACIDADE – MODELO ALINHADO À LGPD

Esta Política de Privacidade tem por objetivo informar de forma clara, transparente e objetiva como [NOME DA EMPRESA] realiza o tratamento de dados pessoais, em conformidade com a Lei nº 13.709/2018 – Lei Geral de Proteção de Dados Pessoais ("LGPD"). Ao utilizar nossos serviços, aplicações, sites ou canais de atendimento, o titular declara estar ciente de que seus dados pessoais poderão ser tratados nos termos aqui descritos.

1. Conceitos básicos e fundamentos

Para fins desta Política, consideram-se:

Dado pessoal: informação relacionada a pessoa natural identificada ou identificável.

Dado pessoal sensível: dado sobre origem racial ou étnica, convicção religiosa, opinião política, filiação a sindicato ou organização de caráter religioso, filosófico ou político, dado referente à saúde ou à vida sexual, dado genético ou biométrico, quando vinculado a uma pessoa natural.

Controlador: [NOME DA EMPRESA], responsável pelas decisões referentes ao tratamento de dados pessoais.

Operador: pessoa natural ou jurídica que realiza o tratamento de dados pessoais em nome do controlador.

Titular: pessoa natural a quem se referem os dados pessoais objeto de tratamento.

Tratamento: toda operação realizada com dados pessoais, como coleta, produção, recepção, classificação, utilização, acesso, reprodução, transmissão, distribuição, processamento, arquivamento, armazenamento, eliminação, avaliação, controle, modificação, comunicação, transferência, difusão ou extração.

O tratamento de dados pessoais realizado por [NOME DA EMPRESA] observa os princípios da finalidade, adequação, necessidade, livre acesso, qualidade dos dados, transparência, segurança, prevenção, não discriminação, responsabilização e prestação de contas, nos termos do art. 6º da LGPD, bem como os fundamentos previstos no art. 2º, tais como o respeito à privacidade, a autodeterminação informativa e o livre desenvolvimento da personalidade.

2. Quais dados pessoais coletamos

[Nome da Empresa] poderá coletar e tratar, de forma direta ou indireta, os seguintes dados pessoais, de acordo com o contexto da relação com o titular:

Dados de identificação: nome completo, CPF, RG ou documento equivalente, data de nascimento, nacionalidade, estado civil.

Dados de contato: endereço residencial ou comercial, e-mail, número de telefone fixo e celular.

Dados de navegação e uso de serviços digitais: endereço IP, identificadores de dispositivo, logs de acesso, data e hora de ações realizadas nos nossos canais, informações de cookies e tecnologias semelhantes, páginas acessadas e interações efetuadas.

Dados de pagamento e transações: dados bancários tokenizados, histórico de pagamentos, valores transacionados, datas de operações e forma de pagamento utilizada.

Dados profissionais e de relacionamento: cargo, empresa, histórico de interação com nossos serviços, registros de atendimentos e comunicações.

Dados sensíveis, quando estritamente necessários: informações de saúde, biometria ou outras categorias sensíveis, sempre em hipóteses específicas previstas em lei e com proteção reforçada.

Quando estritamente necessário para a prestação de serviços, poderemos ainda tratar dados pessoais de terceiros fornecidos pelo titular (como dependentes, representantes legais ou procuradores), presumindo que o titular possui autorização para compartilhá-los.

3. Finalidades e bases legais do tratamento

Os dados pessoais são tratados para finalidades legítimas, específicas e informadas ao titular, com base nas hipóteses legais previstas nos arts. 7º e 11 da LGPD. Entre as principais finalidades, destacam-se:

Execução de contrato ou de procedimentos preliminares: cadastro de usuários, prestação de serviços contratados, atendimento a solicitações, suporte técnico, faturamento, cobrança e gestão de relacionamento.

Cumprimento de obrigação legal ou regulatória: guarda de registros de acesso, emissão de notas fiscais, atendimento a demandas de autoridades administrativas, fiscais ou judiciais.

Legítimo interesse do controlador: melhoria de produtos e serviços, aperfeiçoamento da experiência do usuário, prevenção à fraude, segurança de rede e de informações, desde que respeitados os direitos e liberdades fundamentais do titular.

Consentimento do titular: envio de comunicações de marketing, newsletters, ofertas personalizadas e utilização de determinados cookies não estritamente necessários, quando exigido pela legislação.

Proteção da vida ou da incolumidade física do titular ou de terceiro, tutela da saúde e exercício regular de direitos: em situações específicas, nos termos da LGPD, especialmente quando for indispensável garantir a segurança, a integridade ou a defesa de direitos em processos administrativos, judiciais ou arbitrais.

Sempre que o tratamento se basear em consentimento, este será coletado de forma livre, informada e inequívoca, para finalidades determinadas, sendo facultado ao titular revogá-lo a qualquer tempo, mediante canais adequados.

4. Tratamento de dados pessoais sensíveis

O tratamento de dados pessoais sensíveis será realizado apenas quando estritamente necessário e nas hipóteses autorizadas pela LGPD, tais como cumprimento de obrigação legal ou regulatória, exercício regular de direitos, proteção da vida ou da incolumidade física do titular ou de terceiro, tutela da saúde ou, quando aplicável, mediante consentimento específico e destacado. Em todos os casos, adotaremos medidas técnicas e organizacionais reforçadas para proteção desses dados, limitando o acesso somente a profissionais autorizados e devidamente instruídos.

5. Tratamento de dados de crianças e adolescentes

Quando houver tratamento de dados pessoais de crianças e adolescentes, este será conduzido sempre no melhor interesse do titular, em conformidade com o art. 14 da LGPD e legislação correlata. Nos casos que envolverem crianças, o tratamento dependerá de consentimento específico e em destaque de pelo menos um dos pais ou responsável legal, salvo hipóteses legais de dispensa. As informações sobre o tratamento serão apresentadas em linguagem clara, acessível e adequada, de modo a permitir a compreensão pelos responsáveis e, quando possível, pelos próprios menores.

6. Compartilhamento de dados pessoais

Os dados pessoais poderão ser compartilhados com terceiros apenas quando necessário e observado o princípio da necessidade, nas seguintes situações:

Com operadores e prestadores de serviço que atuem em nome de [NOME DA EMPRESA], como provedores de infraestrutura tecnológica, meios de pagamento, serviços de atendimento, consultorias e empresas de auditoria, sempre mediante obrigações contratuais de confidencialidade e proteção de dados.

Com autoridades públicas, órgãos reguladores ou entidades governamentais, para cumprimento de obrigações legais, regulatórias ou ordens judiciais.

Com parceiros de negócio, em situações específicas e legítimas, devidamente informadas ao titular, respeitando as bases legais aplicáveis e, quando necessário, obtendo consentimento prévio.

Em casos de operações societárias, como fusão, aquisição ou incorporação, hipótese em que a continuidade do tratamento seguirá os termos desta Política ou outra que a substitua, desde que igualmente compatível com a LGPD.

Não realizaremos a comercialização de dados pessoais em desacordo com a legislação vigente, nem compartilharemos dados para finalidades incompatíveis com aquelas informadas ao titular.

7. Transferência internacional de dados

Quando houver transferência internacional de dados pessoais, [NOME DA EMPRESA] observará os requisitos previstos na LGPD, garantindo que o país de destino ou a organização internacional proporcione grau de proteção de dados pessoais adequado ou que existam garantias suficientes de cumprimento dos princípios, direitos do titular e regime de proteção previstos na lei, por meio de cláusulas contratuais, normas corporativas globais ou outros mecanismos reconhecidos pela autoridade competente.

8. Prazo de armazenamento e término do tratamento

Os dados pessoais serão armazenados apenas pelo tempo necessário ao cumprimento das finalidades para as quais foram coletados, ao cumprimento de obrigações legais ou regulatórias, ao exercício regular de direitos em processos administrativos, judiciais ou arbitrais ou conforme outras hipóteses autorizadas pela LGPD. Ao término do tratamento, os dados serão eliminados ou anonimizados, salvo nas hipóteses em que a conservação for permitida ou exigida por lei, conforme o art. 16 da LGPD.

9. Direitos do titular de dados

O titular de dados pessoais possui os direitos previstos nos arts. 17 a 20 da LGPD, podendo exercê-los a qualquer momento, mediante solicitação dirigida aos canais de atendimento indicados por [NOME DA EMPRESA]. Entre esses direitos, destacam-se:

confirmação da existência de tratamento;

acesso aos dados pessoais;

correção de dados incompletos, inexatos ou desatualizados;

anonimização, bloqueio ou eliminação de dados desnecessários, excessivos ou tratados em desconformidade com a LGPD;

portabilidade dos dados a outro fornecedor de serviço ou produto, observada a regulamentação aplicável;

eliminação dos dados pessoais tratados com base no consentimento, ressalvadas as hipóteses de guarda autorizadas por lei;

informação sobre as entidades públicas e privadas com as quais realizamos uso compartilhado de dados;

informação sobre a possibilidade de não fornecer consentimento e sobre as consequências desta negativa;

revogação do consentimento, quando esta for a base legal aplicável;

revisão de decisões tomadas unicamente com base em tratamento automatizado de dados pessoais que afetem seus interesses, quando cabível.

As solicitações serão analisadas e respondidas dentro de prazos razoáveis, em conformidade com a legislação vigente e considerando eventuais limitações técnicas ou legais.

10. Cookies e tecnologias de rastreamento

Poderemos utilizar cookies e tecnologias semelhantes para melhorar a experiência do usuário, personalizar conteúdo, analisar estatísticas de uso e viabilizar funcionalidades essenciais do serviço. Cookies estritamente necessários são utilizados para o funcionamento adequado do site ou aplicação, enquanto cookies de desempenho, funcionalidade ou marketing poderão ser utilizados mediante informação prévia e, quando exigido, consentimento do titular. O usuário poderá gerenciar as preferências de cookies por meio das configurações do navegador ou de ferramentas específicas disponibilizadas em nossos canais.

11. Segurança da informação

[Nome da Empresa] adota medidas técnicas e administrativas aptas a proteger os dados pessoais contra acessos não autorizados e contra situações acidentais ou ilícitas de destruição, perda, alteração, comunicação ou qualquer forma de tratamento inadequado ou ilícito. Tais medidas incluem, entre outras, controles de acesso, criptografia quando aplicável, políticas internas de segurança, gestão de perfis de usuários, registros de logs e programas de capacitação de colaboradores. Embora empreguemos nossos melhores esforços para proteger as informações, nenhum sistema é totalmente imune a incidentes, motivo pelo qual mantemos procedimentos para resposta e mitigação de riscos, incluindo comunicação aos titulares e à autoridade competente, quando exigido pela LGPD.

12. Encarregado pelo tratamento de dados pessoais (DPO)

Para fins de comunicação sobre qualquer assunto relacionado a esta Política ou ao tratamento de dados pessoais, [NOME DA EMPRESA] indica como encarregado de proteção de dados (Data Protection Officer – DPO) a pessoa ou área de contato cujo e-mail e demais informações serão disponibilizados de forma clara e destacada em nossos canais oficiais. O encarregado é o canal de comunicação entre o controlador, os titulares de dados e a Autoridade Nacional de Proteção de Dados (ANPD).

13. Atualizações desta Política de Privacidade

Esta Política de Privacidade poderá ser alterada a qualquer tempo, para refletir mudanças legislativas, regulatórias, tecnológicas ou relativas à prestação de nossos serviços. Sempre que houver atualizações relevantes, envidaremos esforços para informar os titulares por meio dos nossos canais oficiais, destacando as principais alterações. A versão vigente estará sempre disponível em nosso site ou aplicação, com indicação da data de última atualização.

Esta Política de Privacidade deve ser lida em conjunto com os demais termos e condições de uso dos serviços oferecidos por [NOME DA EMPRESA], não excluindo outras obrigações de proteção de dados pessoais previstas na legislação brasileira."""

        logger.info(f"Resumo de referência gerado: {len(resumo_ideal)} caracteres")
        return resumo_ideal
    
    def gerar_checklist(self) -> Dict[str, Dict]:
        """
        Gera checklist estruturado para avaliação de conformidade
        
        Returns:
            Dicionário com estrutura de checklist por categoria
        """
        checklist = {}
        
        for categoria in self.obter_categorias():
            requisitos_cat = self.obter_requisitos_por_categoria(categoria)
            
            checklist[categoria] = {
                'titulo': categoria.replace('_', ' ').title(),
                'requisitos': [
                    {
                        'id': req.id,
                        'titulo': req.titulo,
                        'descricao': req.descricao,
                        'obrigatorio': req.obrigatorio,
                        'peso': req.peso,
                        'palavras_chave': req.palavras_chave,
                        'atendido': False  # Será preenchido na avaliação
                    }
                    for req in requisitos_cat
                ]
            }
        
        return checklist
    
    def obter_requisito(self, id_requisito: str) -> RequisitoLGPD:
        """Obtém um requisito específico por ID"""
        return self.requisitos.get(id_requisito)
    
    def exportar_json(self) -> Dict:
        """Exporta o gold standard para formato JSON"""
        return {
            'versao': '1.0',
            'lei': 'Lei nº 13.709/2018 - LGPD',
            'data_atualizacao': '2025-01-01',
            'requisitos': [
                {
                    'id': req.id,
                    'categoria': req.categoria,
                    'titulo': req.titulo,
                    'descricao': req.descricao,
                    'artigo_lei': req.artigo_lei,
                    'palavras_chave': req.palavras_chave,
                    'obrigatorio': req.obrigatorio,
                    'peso': req.peso
                }
                for req in self.requisitos.values()
            ]
        }
    
    def __len__(self) -> int:
        """Retorna número total de requisitos"""
        return len(self.requisitos)
    
    def __repr__(self) -> str:
        return f"GoldStandardLGPD(requisitos={len(self.requisitos)})"


# Instância global (singleton) do Gold Standard
_gold_standard_instance = None

def obter_gold_standard() -> GoldStandardLGPD:
    """Obtém a instância global do Gold Standard LGPD"""
    global _gold_standard_instance
    if _gold_standard_instance is None:
        _gold_standard_instance = GoldStandardLGPD()
    return _gold_standard_instance
