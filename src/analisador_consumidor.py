"""
Módulo de Análise Automática para Consumidor Final
Funcionalidades:
1. Extração automática de dados coletados
2. Score de conformidade LGPD (0-100)
3. Recomendação de aceitação (aceitar/revisar/rejeitar)
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class AnaliseConsumidor:
    """Resultado da análise para o consumidor final"""
    dados_coletados: List[str]
    dados_sensiveis: List[str]
    finalidades: List[str]
    compartilhamentos: List[str]
    score_lgpd: float
    nivel_risco: str  # 'baixo', 'médio', 'alto'
    recomendacao: str  # 'aceitar', 'revisar', 'rejeitar'
    justificativa: List[str]
    alertas: List[str]


class AnalisadorConsumidor:
    """
    Analisa políticas de privacidade do ponto de vista do consumidor
    Gera recomendações automáticas baseadas em LGPD
    """
    
    # Categorias de dados pessoais
    DADOS_BASICOS = [
        'nome', 'e-mail', 'telefone', 'endereço', 'cpf', 'rg',
        'data de nascimento', 'idade', 'sexo', 'gênero', 'profissão',
        'ocupação', 'cargo', 'empresa', 'escolaridade', 'formação'
    ]
    
    DADOS_FINANCEIROS = [
        'cartão de crédito', 'conta bancária', 'dados bancários',
        'informações de pagamento', 'transações financeiras',
        'histórico de compras', 'renda'
    ]
    
    DADOS_SENSIVEIS_LGPD = [
        'origem racial', 'origem étnica', 'convicção religiosa',
        'opinião política', 'filiação sindical', 'saúde',
        'vida sexual', 'dados genéticos', 'dados biométricos'
    ]
    
    DADOS_LOCALIZACAO = [
        'localização', 'geolocalização', 'gps', 'endereço ip',
        'coordenadas geográficas'
    ]
    
    DADOS_DISPOSITIVO = [
        'dispositivo', 'navegador', 'sistema operacional',
        'cookies', 'identificador único', 'udid', 'imei'
    ]
    
    DADOS_COMPORTAMENTAIS = [
        'histórico de navegação', 'pesquisas', 'interações',
        'preferências', 'comportamento', 'hábitos'
    ]
    
    # Red flags para score LGPD
    RED_FLAGS = [
        'vendemos seus dados',
        'compartilhamos com qualquer terceiro',
        'não podemos garantir',
        'não nos responsabilizamos',
        'impossível excluir',
        'dados permanentes',
        'irrevogável',
        'imodificável'
    ]
    
    # Green flags (boas práticas)
    GREEN_FLAGS = [
        'você pode solicitar exclusão',
        'direito de acesso',
        'direito de correção',
        'direito de portabilidade',
        'consentimento',
        'opt-out',
        'criptografia',
        'anonimização',
        'dpo',
        'encarregado'
    ]
    
    def __init__(self):
        """Inicializa o analisador"""
        pass
    
    def _buscar_palavra_completa(self, palavra: str, texto: str) -> bool:
        """
        Busca por palavra completa usando word boundaries para evitar matches parciais
        Exemplo: 'imei' NÃO deve dar match em 'primeira'
        
        Args:
            palavra: Palavra a buscar (ex: 'imei', 'email')
            texto: Texto onde buscar (já em lowercase)
            
        Returns:
            True se a palavra completa foi encontrada
        """
        # Usa word boundary \b para garantir match de palavra completa
        # \b garante que não há caractere alfanumérico antes ou depois
        pattern = r'\b' + re.escape(palavra) + r'\b'
        return bool(re.search(pattern, texto))
    
    def extrair_dados_coletados(self, texto: str, texto_classificado: Dict) -> Tuple[List[str], List[str]]:
        """
        Extrai automaticamente quais dados são coletados
        
        Returns:
            (dados_coletados, dados_sensiveis)
        """
        texto_lower = texto.lower()
        dados_encontrados = []
        dados_sensiveis_encontrados = []
        
        # Busca na seção "coleta_dados" se disponível
        texto_coleta = ""
        if 'coleta_dados' in texto_classificado:
            texto_coleta = " ".join(texto_classificado['coleta_dados']).lower()
        else:
            texto_coleta = texto_lower
        
        # Detecta dados básicos
        for dado in self.DADOS_BASICOS:
            if self._buscar_palavra_completa(dado, texto_coleta):
                dados_encontrados.append(f"{dado.title()}")
        
        # Detecta dados financeiros
        for dado in self.DADOS_FINANCEIROS:
            if self._buscar_palavra_completa(dado, texto_coleta):
                dados_encontrados.append(f"{dado.title()}")
        
        # Detecta dados sensíveis (ATENÇÃO!)
        for dado in self.DADOS_SENSIVEIS_LGPD:
            if self._buscar_palavra_completa(dado, texto_coleta):
                dados_sensiveis_encontrados.append(f"{dado.title()}")
        
        # Detecta dados de localização
        for dado in self.DADOS_LOCALIZACAO:
            if self._buscar_palavra_completa(dado, texto_coleta):
                dados_encontrados.append(f"{dado.title()}")
        
        # Detecta dados de dispositivo
        for dado in self.DADOS_DISPOSITIVO:
            if self._buscar_palavra_completa(dado, texto_coleta):
                dados_encontrados.append(f"{dado.title()}")
        
        # Detecta dados comportamentais
        for dado in self.DADOS_COMPORTAMENTAIS:
            if self._buscar_palavra_completa(dado, texto_coleta):
                dados_encontrados.append(f"{dado.title()}")
        
        return list(set(dados_encontrados)), list(set(dados_sensiveis_encontrados))
    
    def extrair_finalidades(self, texto_classificado: Dict) -> List[str]:
        """Extrai as finalidades do tratamento"""
        finalidades = []
        
        # Buscar em TODAS as categorias disponíveis
        texto_completo = ""
        for categoria, trechos in texto_classificado.items():
            if isinstance(trechos, list):
                texto_completo += " " + " ".join(trechos)
        
        texto_finalidade = texto_completo.lower()
        
        # Se não há texto classificado, retornar não especificado
        if not texto_finalidade.strip():
            return ["Não especificado claramente"]
        
        # Finalidades comuns - busca expandida com mais variações
        if any(palavra in texto_finalidade for palavra in ['marketing', 'publicidade', 'promoç', 'anúnc', 'divulgaç', 'propaganda']):
            finalidades.append("Marketing e publicidade")
        
        if any(palavra in texto_finalidade for palavra in ['melhorar', 'aprimorar', 'desenvolv', 'aperfeiço', 'otimiz', 'qualidade']):
            finalidades.append("Melhoria de serviços")
        
        if any(palavra in texto_finalidade for palavra in ['transação', 'transaç', 'pagamento', 'compra', 'venda', 'comerci', 'financeiro']):
            finalidades.append("Processamento de transações")
        
        # Expandido para capturar mais variações de prestação de serviços
        if any(palavra in texto_finalidade for palavra in ['prestaç', 'execuç', 'fornec', 'entreg', 'ofere', 'disponibiliz', 'proporcionar', 'realizar', 'execut', 'operar']):
            finalidades.append("Prestação de serviços")
        
        if any(palavra in texto_finalidade for palavra in ['segurança', 'seguran', 'fraude', 'proteção', 'proteç', 'risco']):
            finalidades.append("Segurança e prevenção de fraudes")
        
        if any(palavra in texto_finalidade for palavra in ['legal', 'obrigaç', 'lei', 'cumprimento', 'ordem judicial', 'regulamentar', 'normativ']):
            finalidades.append("Cumprimento de obrigações legais")
        
        if any(palavra in texto_finalidade for palavra in ['contato', 'comunicaç', 'notific', 'informar', 'avisar', 'responder']):
            finalidades.append("Comunicação com usuários")
        
        if any(palavra in texto_finalidade for palavra in ['personaliz', 'customiz', 'adapt', 'preferência']):
            finalidades.append("Personalização de experiência")
        
        if any(palavra in texto_finalidade for palavra in ['análise', 'estatística', 'métric', 'monitorar', 'avaliar', 'estudar']):
            finalidades.append("Análises e estatísticas")
        
        # Adicionar mais finalidades comuns
        if any(palavra in texto_finalidade for palavra in ['autenticar', 'autenticaç', 'identificar', 'identificaç', 'login', 'acesso']):
            finalidades.append("Autenticação e controle de acesso")
        
        if any(palavra in texto_finalidade for palavra in ['suporte', 'atendimento', 'assistência', 'ajuda', 'apoio']):
            finalidades.append("Suporte ao cliente")
        
        if any(palavra in texto_finalidade for palavra in ['contrat', 'acordo', 'termo', 'vínculo']):
            finalidades.append("Execução de contrato")
        
        return finalidades if finalidades else ["Não especificado claramente"]
    
    def extrair_compartilhamentos(self, texto_classificado: Dict) -> List[str]:
        """Extrai com quem os dados são compartilhados"""
        compartilhamentos = []
        
        # Buscar em TODAS as categorias disponíveis
        texto_completo = ""
        for categoria, trechos in texto_classificado.items():
            if isinstance(trechos, list):
                texto_completo += " " + " ".join(trechos)
        
        texto_comp = texto_completo.lower()
        
        # Se não há texto classificado, retornar não especificado
        if not texto_comp.strip():
            return ["Não especificado"]
        
        # Busca expandida com stem matching e contexto de compartilhamento
        # Verificar se o texto menciona compartilhamento/divulgação
        menciona_compartilhamento = any(palavra in texto_comp for palavra in [
            'compartilh', 'divulg', 'transmit', 'transfer', 'repassa', 'fornec', 'revela'
        ])
        
        if menciona_compartilhamento or 'parceiro' in texto_comp or 'terceiro' in texto_comp:
            if any(palavra in texto_comp for palavra in ['terceiros', 'terceiro']):
                compartilhamentos.append("Terceiros (não especificados)")
            
            if any(palavra in texto_comp for palavra in ['parceiros', 'parceiro', 'parceir']):
                compartilhamentos.append("Parceiros comerciais")
            
            if any(palavra in texto_comp for palavra in ['autoridades', 'autoridade', 'governament', 'judicial', 'ordem judicial', 'administrativ']):
                compartilhamentos.append("Autoridades governamentais")
            
            if any(palavra in texto_comp for palavra in ['fornecedores', 'fornecedor', 'prestador', 'prestaç']):
                compartilhamentos.append("Prestadores de serviços")
            
            if any(palavra in texto_comp for palavra in ['subsidiárias', 'subsidiária', 'afiliadas', 'afiliada', 'grupo', 'coligada', 'controlada']):
                compartilhamentos.append("Empresas do grupo")
        
        if any(palavra in texto_comp for palavra in ['não compartilhamos', 'não vendemos', 'não vende', 'não divulg']):
            compartilhamentos.append("Empresa declara não vender dados")
        
        return compartilhamentos if compartilhamentos else ["Não especificado"]
    
    def calcular_score_lgpd(self, texto: str, texto_classificado: Dict, 
                           dados_sensiveis: List[str]) -> Tuple[float, List[str]]:
        """
        Calcula score de conformidade LGPD (0-100)
        
        Returns:
            (score, justificativas)
        """
        score = 100.0
        justificativas = []
        texto_lower = texto.lower()
        
        # PENALIZAÇÕES
        
        # 1. Dados sensíveis (-20 pontos)
        if dados_sensiveis:
            score -= 20
            justificativas.append(f" Coleta {len(dados_sensiveis)} tipo(s) de dados sensíveis (-20 pts)")
        
        # 2. Red flags (-10 pontos cada, máx -40)
        red_flags_encontrados = 0
        for red_flag in self.RED_FLAGS:
            if red_flag in texto_lower:
                red_flags_encontrados += 1
                score -= 10
                justificativas.append(f" Contém '{red_flag}' (-10 pts)")
                if red_flags_encontrados >= 4:
                    break
        
        # 3. Falta de seções importantes (-5 pontos cada)
        secoes_importantes = ['direitos_titular', 'seguranca', 'contato']
        for secao in secoes_importantes:
            if secao not in texto_classificado or not texto_classificado[secao]:
                score -= 5
                justificativas.append(f" Seção '{secao}' ausente ou vazia (-5 pts)")
        
        # 4. Compartilhamento vago com terceiros (-10 pontos)
        if 'compartilhamento' in texto_classificado:
            texto_comp = " ".join(texto_classificado['compartilhamento']).lower()
            if 'terceiros' in texto_comp and 'quais' not in texto_comp:
                score -= 10
                justificativas.append(" Compartilhamento com terceiros não detalhado (-10 pts)")
        
        # BONIFICAÇÕES
        
        # 1. Green flags (+5 pontos cada, máx +30)
        green_flags_encontrados = 0
        for green_flag in self.GREEN_FLAGS:
            if green_flag in texto_lower:
                green_flags_encontrados += 1
                score += 5
                if green_flags_encontrados <= 6:  # Mostra apenas primeiros 6
                    justificativas.append(f" Menciona '{green_flag}' (+5 pts)")
                if green_flags_encontrados >= 6:
                    break
        
        # 2. DPO claramente identificado (+10 pontos)
        if 'dpo' in texto_lower or 'encarregado' in texto_lower:
            if '@' in texto:  # Tem e-mail do DPO
                score += 10
                justificativas.append(" DPO identificado com contato (+10 pts)")
        
        # Limita score entre 0 e 100
        score = max(0.0, min(100.0, score))
        
        return score, justificativas
    
    def gerar_recomendacao(self, score: float, dados_sensiveis: List[str],
                          compartilhamentos: List[str]) -> Tuple[str, str, List[str]]:
        """
        Gera recomendação automática: aceitar, revisar ou rejeitar
        
        Returns:
            (recomendacao, nivel_risco, alertas)
        """
        alertas = []
        
        # Análise de risco
        if score >= 80:
            nivel_risco = "baixo"
            recomendacao = "aceitar"
            mensagem = " **ACEITAR** - Política parece estar em conformidade com LGPD"
        elif score >= 60:
            nivel_risco = "médio"
            recomendacao = "revisar"
            mensagem = " **REVISAR COM ATENÇÃO** - Alguns pontos de preocupação identificados"
            alertas.append(" Leia especialmente as seções sobre compartilhamento de dados")
        else:
            nivel_risco = "alto"
            recomendacao = "rejeitar"
            mensagem = " **CONSIDERAR REJEITAR** - Múltiplos problemas identificados"
            alertas.append(" Esta política apresenta sérios problemas de conformidade")
        
        # Alertas adicionais
        if dados_sensiveis:
            alertas.append(f" ATENÇÃO: Coleta dados sensíveis ({len(dados_sensiveis)} tipo(s))")
            alertas.append("   Verifique se há justificativa legal e consentimento explícito")
        
        if any('terceiros' in c.lower() for c in compartilhamentos):
            alertas.append(" Seus dados serão compartilhados com terceiros")
            alertas.append("   Verifique quem são esses terceiros")
        
        if score < 50:
            alertas.append(" Score muito baixo - considere usar serviço alternativo")
        
        return recomendacao, nivel_risco, [mensagem] + alertas
    
    def analisar(self, texto_original: str, texto_classificado: Dict) -> AnaliseConsumidor:
        """
        Análise completa para o consumidor final
        
        Args:
            texto_original: texto completo da política
            texto_classificado: dicionário com texto por categoria LGPD
        
        Returns:
            AnaliseConsumidor com todas as informações
        """
        # 1. Extrai dados coletados
        dados_coletados, dados_sensiveis = self.extrair_dados_coletados(
            texto_original, texto_classificado
        )
        
        # 2. Extrai finalidades
        finalidades = self.extrair_finalidades(texto_classificado)
        
        # 3. Extrai compartilhamentos
        compartilhamentos = self.extrair_compartilhamentos(texto_classificado)
        
        # 4. Calcula score LGPD
        score_lgpd, justificativas_score = self.calcular_score_lgpd(
            texto_original, texto_classificado, dados_sensiveis
        )
        
        # 5. Gera recomendação
        recomendacao, nivel_risco, alertas = self.gerar_recomendacao(
            score_lgpd, dados_sensiveis, compartilhamentos
        )
        
        return AnaliseConsumidor(
            dados_coletados=dados_coletados,
            dados_sensiveis=dados_sensiveis,
            finalidades=finalidades,
            compartilhamentos=compartilhamentos,
            score_lgpd=score_lgpd,
            nivel_risco=nivel_risco,
            recomendacao=recomendacao,
            justificativa=justificativas_score,
            alertas=alertas
        )
    
    def gerar_relatorio_consumidor(self, analise: AnaliseConsumidor, empresa: str) -> str:
        """
        Gera relatório formatado em markdown para o consumidor
        """
        # Emoji do score
        if analise.score_lgpd >= 80:
            emoji_score = ""
        elif analise.score_lgpd >= 60:
            emoji_score = ""
        else:
            emoji_score = ""
        
        relatorio = f"""
#  Análise da Política de Privacidade - {empresa}

## {emoji_score} Score de Conformidade LGPD: {analise.score_lgpd:.0f}/100

**Nível de Risco:** {analise.nivel_risco.upper()}

---

##  O QUE A EMPRESA VAI COLETAR DE VOCÊ

### Dados Pessoais:
"""
        
        if analise.dados_coletados:
            for dado in analise.dados_coletados:
                relatorio += f"- {dado}\n"
        else:
            relatorio += "- ℹ Não especificado claramente\n"
        
        if analise.dados_sensiveis:
            relatorio += f"\n###  DADOS SENSÍVEIS (LGPD Art. 5º):\n"
            for dado in analise.dados_sensiveis:
                relatorio += f"- {dado}\n"
            relatorio += "\n> **ATENÇÃO:** Dados sensíveis requerem consentimento explícito!\n"
        
        relatorio += f"""
---

##  PARA QUE VÃO USAR SEUS DADOS

"""
        for finalidade in analise.finalidades:
            relatorio += f"- {finalidade}\n"
        
        relatorio += f"""
---

##  COM QUEM VÃO COMPARTILHAR

"""
        for compartilhamento in analise.compartilhamentos:
            relatorio += f"- {compartilhamento}\n"
        
        relatorio += f"""
---

##  RECOMENDAÇÃO AUTOMÁTICA

"""
        for alerta in analise.alertas:
            relatorio += f"{alerta}\n\n"
        
        relatorio += f"""
---

##  JUSTIFICATIVA DO SCORE

"""
        for justificativa in analise.justificativa:
            relatorio += f"- {justificativa}\n"
        
        relatorio += f"""
---

## ℹ SEUS DIREITOS (LGPD)

Independente do score, você SEMPRE tem direito a:

 **Acesso** - Ver quais dados a empresa tem sobre você
 **Correção** - Corrigir dados incorretos
 **Exclusão** - Solicitar remoção dos seus dados
 **Portabilidade** - Transferir seus dados para outra empresa
 **Revogação** - Retirar seu consentimento a qualquer momento

 Contate o DPO (Encarregado de Proteção de Dados) da empresa para exercer seus direitos.

---

** Esta análise foi gerada automaticamente e serve como orientação inicial. 
Para decisões importantes, consulte um advogado especializado em proteção de dados.**
"""
        
        return relatorio


# Exemplo de uso
if __name__ == "__main__":
    # Teste rápido
    analisador = AnalisadorConsumidor()
    
    texto_exemplo = """
    Coletamos seus dados pessoais como nome, e-mail, CPF, dados biométricos,
    histórico de navegação e localização. Utilizamos para marketing, melhorar
    nossos serviços e prevenir fraudes. Compartilhamos com parceiros comerciais
    e terceiros. Você pode solicitar exclusão dos seus dados através do DPO
    em dpo@empresa.com.
    """
    
    texto_classificado = {
        'coleta_dados': [texto_exemplo],
        'finalidade': [texto_exemplo],
        'compartilhamento': [texto_exemplo],
        'direitos_titular': [texto_exemplo],
        'contato': [texto_exemplo]
    }
    
    analise = analisador.analisar(texto_exemplo, texto_classificado)
    relatorio = analisador.gerar_relatorio_consumidor(analise, "Empresa Exemplo")
    
    print(relatorio)
    print(f"\n Score: {analise.score_lgpd:.0f}/100")
    print(f" Recomendação: {analise.recomendacao.upper()}")
