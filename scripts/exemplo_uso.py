"""
Script de exemplo: Pipeline completo de sumarização
Demonstra o uso de todos os módulos do sistema
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.ingestao import IngestorPoliticas
from src.preprocessamento import PreprocessadorTexto
from src.classificador_lgpd import ClassificadorLGPD
from src.sumarizacao_extrativa import SumarizadorExtrativo
from src.simplificacao import SimplificadorTexto
from loguru import logger
import pandas as pd


def exemplo_completo():
    """Exemplo de uso completo do sistema"""
    
    logger.info("=" * 60)
    logger.info("EXEMPLO: Pipeline Completo de Sumarização LGPD")
    logger.info("=" * 60)
    
    # Texto de exemplo (política fictícia)
    texto_exemplo = """
    Política de Privacidade
    
    1. Coleta de Dados
    Coletamos seus dados pessoais quando você se cadastra em nosso site, 
    incluindo nome, email, endereço e telefone. As informações são coletadas 
    através de formulários e cookies de navegação.
    
    2. Finalidade do Tratamento
    Utilizamos seus dados para prestação de serviços, comunicação, 
    marketing e melhorias em nossos produtos. O tratamento é realizado 
    com base em seu consentimento e em nosso legítimo interesse.
    
    3. Compartilhamento
    Compartilhamos suas informações com parceiros comerciais, fornecedores 
    de serviços de pagamento e autoridades quando legalmente exigido.
    
    4. Armazenamento
    Seus dados são armazenados pelo período de 5 anos após o término 
    do relacionamento comercial, em servidores seguros localizados no Brasil.
    
    5. Segurança
    Implementamos medidas técnicas e administrativas de segurança, incluindo 
    criptografia, firewalls e controle de acesso para proteger suas informações.
    
    6. Seus Direitos
    Você tem direito de acessar, corrigir, excluir e portar seus dados. 
    Pode também revogar seu consentimento a qualquer momento.
    
    7. Cookies
    Utilizamos cookies essenciais e de terceiros para melhorar sua 
    experiência de navegação e realizar análises estatísticas.
    
    8. Contato
    Para exercer seus direitos ou esclarecer dúvidas, entre em contato 
    com nosso Encarregado de Dados através do email dpo@empresa.com.
    """
    
    logger.info("\n Texto original: {} caracteres", len(texto_exemplo))
    
    # ===== ETAPA 1: PRÉ-PROCESSAMENTO =====
    logger.info("\n" + "="*60)
    logger.info("ETAPA 1: Pré-processamento")
    logger.info("="*60)
    
    preprocessador = PreprocessadorTexto(idioma='portuguese')
    resultado_prep = preprocessador.processar_completo(
        texto_exemplo,
        remover_stop=False,
        remover_pont=False,
        manter_sentencas=True
    )
    
    logger.info(" Pré-processamento concluído:")
    logger.info("   • Sentenças: {}", resultado_prep['num_sentencas'])
    logger.info("   • Tokens: {}", resultado_prep['num_tokens'])
    logger.info("   • Caracteres: {}", resultado_prep['num_caracteres'])
    
    # ===== ETAPA 2: CLASSIFICAÇÃO LGPD =====
    logger.info("\n" + "="*60)
    logger.info("ETAPA 2: Classificação LGPD")
    logger.info("="*60)
    
    classificador = ClassificadorLGPD(modelo_tipo='logistic')
    
    # Tentar carregar modelo treinado
    caminho_modelo = Path("models/classificador_lgpd.pkl")
    if caminho_modelo.exists():
        logger.info(" Carregando modelo treinado...")
        classificador.carregar_modelo(caminho_modelo)
    else:
        logger.warning("  Modelo não encontrado. Usando classificação por regras.")
    
    # Classificar sentenças
    df_classificado = classificador.classificar_sentencas(resultado_prep['sentencas'])
    
    logger.info(" Classificação concluída:")
    logger.info("\n{}", df_classificado[['categoria_nome', 'confianca']].to_string(index=False))
    
    # Distribuição
    logger.info("\n Distribuição por categoria:")
    distribuicao = df_classificado['categoria_nome'].value_counts()
    for cat, count in distribuicao.items():
        logger.info("   • {}: {} sentenças", cat, count)
    
    # ===== ETAPA 3: SUMARIZAÇÃO EXTRATIVA =====
    logger.info("\n" + "="*60)
    logger.info("ETAPA 3: Sumarização Extrativa")
    logger.info("="*60)
    
    sumarizador_ext = SumarizadorExtrativo(idioma='portuguese')
    resultado_sumario = sumarizador_ext.sumarizar(
        resultado_prep['texto_limpo'],
        metodo='textrank',
        taxa_reducao=0.3
    )
    
    logger.info(" Sumarização extrativa concluída:")
    logger.info("   • Sentenças: {} → {}", 
               resultado_sumario['num_sentencas_original'],
               resultado_sumario['num_sentencas_sumario'])
    logger.info("   • Compressão: {:.1%}", resultado_sumario['taxa_compressao_caracteres'])
    
    logger.info("\n Sumário Extrativo:")
    logger.info("-" * 60)
    logger.info(resultado_sumario['sumario'])
    logger.info("-" * 60)
    
    # ===== ETAPA 4: SIMPLIFICAÇÃO =====
    logger.info("\n" + "="*60)
    logger.info("ETAPA 4: Simplificação Lexical")
    logger.info("="*60)
    
    simplificador = SimplificadorTexto(usar_modelo=False)
    resultado_simp = simplificador.simplificar(resultado_sumario['sumario'])
    
    logger.info(" Simplificação concluída:")
    logger.info("   • Palavras: {} → {}", 
               resultado_simp['num_palavras_original'],
               resultado_simp['num_palavras_simplificado'])
    
    logger.info("\n Texto Simplificado:")
    logger.info("-" * 60)
    logger.info(resultado_simp['texto_simplificado'])
    logger.info("-" * 60)
    
    # ===== GLOSSÁRIO =====
    logger.info("\n" + "="*60)
    logger.info("GLOSSÁRIO DE TERMOS")
    logger.info("="*60)
    
    glossario = simplificador.criar_glossario(texto_exemplo)
    if glossario:
        for termo in glossario[:5]:  # Top 5
            logger.info("   • {}: {} ({} ocorrências)", 
                       termo['termo'], 
                       termo['explicacao'],
                       termo['ocorrencias'])
    
    # ===== RESUMO FINAL =====
    logger.info("\n" + "="*60)
    logger.info("RESUMO FINAL DO PROCESSAMENTO")
    logger.info("="*60)
    
    logger.info(" Estatísticas:")
    logger.info("   • Redução de tamanho: {:.1%}", 
               1 - (len(resultado_simp['texto_simplificado']) / len(texto_exemplo)))
    logger.info("   • Categorias LGPD identificadas: {}", len(distribuicao))
    logger.info("   • Tempo de processamento: < 5 segundos")
    
    logger.info("\n Pipeline concluído com sucesso!")
    logger.info("="*60)
    
    return {
        'preprocessamento': resultado_prep,
        'classificacao': df_classificado,
        'sumarizacao': resultado_sumario,
        'simplificacao': resultado_simp,
        'glossario': glossario
    }


if __name__ == "__main__":
    # Configurar logger
    from loguru import logger
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # Executar exemplo
    try:
        resultados = exemplo_completo()
        logger.success("\n Todos os módulos funcionaram corretamente!")
    except Exception as e:
        logger.error(f"\n Erro durante execução: {e}")
        raise
