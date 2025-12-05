# This file contains the new resumo_final_consumidor function
# Copy this and insert it before def main() in app.py

def resumo_final_consumidor():
    """Resumo Final Orientado ao Consumidor - Última Etapa"""
    st.markdown("## Resumo Final - Visão do Consumidor")
    st.markdown("---")
    
    st.markdown("""
    <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #2196F3;'>
    <h3 style='margin-top: 0; color: #1565C0;'> Sobre este Resumo</h3>
    <p style='margin-bottom: 10px; color: #333;'>
    Este é o <strong>resumo final orientado ao consumidor</strong>, consolidando toda a análise da política de privacidade
    em linguagem acessível e compreensível.
    </p>
    <p style='margin-bottom: 10px; color: #333;'>
    <strong>O que você encontrará:</strong>
    </p>
    <ul style='margin-bottom: 10px; margin-left: 20px; color: #333;'>
        <li><strong>Recomendação clara:</strong> ACEITAR, REVISAR ou REJEITAR</li>
        <li><strong>Dados coletados:</strong> O que a empresa sabe sobre você</li>
        <li><strong>Seus direitos:</strong> O que você pode exigir da empresa</li>
        <li><strong>Resumo simplificado:</strong> Versão fácil de entender da política</li>
        <li><strong>Nível de risco:</strong> Avaliação da segurança dos seus dados</li>
    </ul>
    <p style='margin-bottom: 0; color: #333;'>
     <em>Este resumo usa o texto simplificado gerado anteriormente e a análise de conformidade LGPD.</em>
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificações
    if not st.session_state.get('analise_conformidade_lgpd'):
        st.warning("Execute primeiro a **Conformidade LGPD** para avaliar a política.")
        return
    
    if not st.session_state.texto_classificado is None:
        pass  # OK, tem classificação
    else:
        st.warning("Classifique o texto primeiro na aba 'Classificação LGPD'")
        return
    
    # Nome da empresa
    nome_empresa = st.text_input(
        "Nome da empresa:",
        value=st.session_state.empresa_nome if st.session_state.empresa_nome else "Empresa"
    )
    
    if st.button("Gerar Resumo Final para Consumidor", type="primary", use_container_width=True):
        with st.spinner("Gerando resumo final..."):
            try:
                # Obter dados da conformidade LGPD
                conformidade = st.session_state.analise_conformidade_lgpd
                
                # Analisar dados coletados
                analisador = AnalisadorConsumidor()
                texto_classificado_dict = {}
                if isinstance(st.session_state.texto_classificado, dict):
                    texto_classificado_dict = st.session_state.texto_classificado
                else:
                    df = st.session_state.texto_classificado
                    for categoria in df['categoria_nome'].unique():
                        sentencas = df[df['categoria_nome'] == categoria]['sentenca'].tolist()
                        texto_classificado_dict[categoria] = sentencas
                
                analise_consumidor = analisador.analisar(
                    st.session_state.texto_original,
                    texto_classificado_dict
                )
                
                # Determinar recomendação baseada no score
                score = conformidade['score_conformidade']
                if score >= 80:
                    recomendacao = "aceitar"
                elif score >= 65:
                    recomendacao = "revisar"
                else:
                    recomendacao = "rejeitar"
                
                # Salvar resumo final
                st.session_state.resumo_final = {
                    'empresa': nome_empresa,
                    'score_lgpd': score,
                    'conformidade': conformidade['conformidade_binaria'],
                    'recomendacao': recomendacao,
                    'nivel_risco': conformidade['nivel_risco'],
                    'requisitos_atendidos': conformidade['requisitos_atendidos'],
                    'requisitos_nao_atendidos': conformidade['requisitos_nao_atendidos'],
                    'dados_coletados': analise_consumidor.dados_coletados,
                    'dados_sensiveis': analise_consumidor.dados_sensiveis,
                    'finalidades': analise_consumidor.finalidades,
                    'compartilhamentos': analise_consumidor.compartilhamentos,
                    'alertas': analise_consumidor.alertas
                }
                
                st.success("Resumo final gerado com sucesso!")
                
            except Exception as e:
                st.error(f"Erro ao gerar resumo: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return
    
    # MOSTRAR RESUMO FINAL
    if st.session_state.get('resumo_final'):
        resumo = st.session_state.resumo_final
        
        # ========================================
        # DECISÃO FINAL - GRANDE DESTAQUE
        # ========================================
        st.markdown("---")
        st.markdown("#DECISÃO FINAL")
        
        rec = resumo['recomendacao']
        if rec == "aceitar":
            cor_principal = "green"
            emoji_principal = "✅"
            titulo_decisao = "VOCÊ PODE ACEITAR"
            explicacao_decisao = f"A política de privacidade da **{resumo['empresa']}** está em boa conformidade com a LGPD. Seus dados pessoais têm proteção adequada."
            cor_fundo = "#d4edda"
            cor_borda = "#28a745"
        elif rec == "revisar":
            cor_principal = "orange"
            emoji_principal = "⚠️"
            titulo_decisao = "REVISAR COM ATENÇÃO"
            explicacao_decisao = f"A política da **{resumo['empresa']}** tem conformidade parcial com a LGPD. Revise os pontos faltantes antes de decidir."
            cor_fundo = "#fff3cd"
            cor_borda = "#ffc107"
        else:
            cor_principal = "red"
            emoji_principal = "❌"
            titulo_decisao = "NÃO RECOMENDADO"
            explicacao_decisao = f"A política da **{resumo['empresa']}** apresenta baixa conformidade com a LGPD. Seus dados podem estar em risco."
            cor_fundo = "#f8d7da"
            cor_borda = "#dc3545"
        
        st.markdown(f"""
        <div style='background-color: {cor_fundo}; padding: 40px; border-radius: 15px; 
                    border: 3px solid {cor_borda}; text-align: center; margin-bottom: 30px;'>
            <h1 style='font-size: 4em; margin: 0;'>{emoji_principal}</h1>
            <h1 style='color: {cor_principal}; margin: 20px 0; font-size: 2.5em;'>{titulo_decisao}</h1>
            <p style='font-size: 1.3em; color: #333; margin: 20px 0;'>{explicacao_decisao}</p>
            <hr style='border: 2px solid {cor_borda}; margin: 30px 0;'>
            <div style='display: flex; justify-content: space-around; flex-wrap: wrap;'>
                <div style='margin: 10px;'>
                    <h2 style='color: {cor_principal}; margin: 5px 0;'>{resumo['score_lgpd']:.0f}/100</h2>
                    <p style='margin: 0; color: #666;'>Score de Conformidade</p>
                </div>
                <div style='margin: 10px;'>
                    <h2 style='color: {cor_principal}; margin: 5px 0; text-transform: uppercase;'>{resumo['nivel_risco']}</h2>
                    <p style='margin: 0; color: #666;'>Nível de Risco</p>
                </div>
                <div style='margin: 10px;'>
                    <h2 style='color: {cor_principal}; margin: 5px 0;'>{len(resumo['requisitos_atendidos'])}/23</h2>
                    <p style='margin: 0; color: #666;'>Requisitos LGPD</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Continue with tabs...
