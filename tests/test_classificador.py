"""
Testes para o módulo de classificação LGPD
"""

import pytest
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from classificador_lgpd import ClassificadorLGPD, CATEGORIAS_LGPD


class TestClassificadorLGPD:
    
    def test_inicializacao(self):
        """Testa inicialização do classificador"""
        clf = ClassificadorLGPD()
        assert clf is not None
        assert clf.modelo_tipo == 'logistic'
        assert not clf.treinado
    
    def test_categorias_lgpd(self):
        """Testa se todas as categorias estão definidas"""
        assert len(CATEGORIAS_LGPD) == 12
        assert 'coleta_dados' in CATEGORIAS_LGPD
        assert 'seguranca' in CATEGORIAS_LGPD
        assert 'direitos_titular' in CATEGORIAS_LGPD
    
    def test_classificacao_por_regras(self):
        """Testa classificação baseada em regras"""
        clf = ClassificadorLGPD()
        
        # Texto sobre coleta de dados
        texto1 = "Coletamos seus dados pessoais quando você se cadastra no site."
        resultado1 = clf.classificar_por_regras(texto1)
        assert resultado1 == 'coleta_dados'
        
        # Texto sobre segurança
        texto2 = "Utilizamos criptografia para proteger suas informações."
        resultado2 = clf.classificar_por_regras(texto2)
        assert resultado2 == 'seguranca'
        
        # Texto sobre direitos
        texto3 = "Você tem direito de solicitar a exclusão de seus dados."
        resultado3 = clf.classificar_por_regras(texto3)
        assert resultado3 == 'direitos_titular'
    
    def test_classificar_sem_treinamento(self):
        """Testa classificação sem modelo treinado (usa regras)"""
        clf = ClassificadorLGPD()
        texto = "Armazenamos seus dados pelo prazo de 5 anos."
        resultado = clf.classificar(texto)
        
        assert 'categoria' in resultado
        assert 'confianca' in resultado
        assert resultado['metodo'] == 'regras'
    
    def test_treinar_modelo(self):
        """Testa treinamento básico do modelo"""
        clf = ClassificadorLGPD()
        
        # Dados de treino mínimos
        textos = [
            "Coletamos seus dados pessoais",
            "Utilizamos criptografia",
            "Você tem direito de acesso"
        ]
        categorias = ['coleta_dados', 'seguranca', 'direitos_titular']
        
        resultado = clf.treinar(textos, categorias, test_size=0.3)
        
        assert clf.treinado
        assert 'acuracia_treino' in resultado
        assert 'acuracia_teste' in resultado
        assert resultado['num_exemplos'] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
