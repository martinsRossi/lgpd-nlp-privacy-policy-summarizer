"""
Módulo de Avaliação de Sumarizações e Conformidade LGPD
Calcula métricas automáticas (ROUGE, BLEU) e avalia conformidade com Gold Standard Universal LGPD
"""

import json
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pandas as pd

from .gold_standard_lgpd import obter_gold_standard


class AvaliadorSumarizacao:
    """Classe para avaliação automática de sumarizações e conformidade LGPD"""
    
    def __init__(self):
        """Inicializa o avaliador"""
        logger.info("Inicializando AvaliadorSumarizacao com Gold Standard LGPD")
        
        # Inicializar scorer ROUGE
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        # Smoothing para BLEU
        self.smoothing = SmoothingFunction()
        
        # Gold Standard Universal LGPD
        self.gold_standard_lgpd = obter_gold_standard()
        self._texto_referencia_lgpd = None
        self._resumo_referencia_lgpd = None
    
    def calcular_rouge(
        self,
        referencia: str,
        candidato: str
    ) -> Dict[str, float]:
        """
        Calcula métricas ROUGE
        
        Args:
            referencia: Texto de referência (ground truth)
            candidato: Texto candidato (gerado)
            
        Returns:
            Dicionário com scores ROUGE
        """
        scores = self.rouge_scorer.score(referencia, candidato)
        
        resultado = {
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rouge2_precision': scores['rouge2'].precision,
            'rouge2_recall': scores['rouge2'].recall,
            'rouge2_f1': scores['rouge2'].fmeasure,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall,
            'rougeL_f1': scores['rougeL'].fmeasure,
        }
        
        logger.debug(f"ROUGE-1 F1: {resultado['rouge1_f1']:.3f}, ROUGE-2 F1: {resultado['rouge2_f1']:.3f}")
        return resultado
    
    def calcular_bleu(
        self,
        referencia: str,
        candidato: str,
        max_ngram: int = 4
    ) -> Dict[str, float]:
        """
        Calcula métrica BLEU
        
        Args:
            referencia: Texto de referência
            candidato: Texto candidato
            max_ngram: N-grama máximo (padrão: 4)
            
        Returns:
            Dicionário com scores BLEU
        """
        # Tokenizar
        ref_tokens = referencia.split()
        cand_tokens = candidato.split()
        
        # BLEU com smoothing
        try:
            # BLEU-1 a BLEU-4
            weights_list = [
                (1.0, 0, 0, 0),      # BLEU-1
                (0.5, 0.5, 0, 0),    # BLEU-2
                (0.33, 0.33, 0.33, 0),  # BLEU-3
                (0.25, 0.25, 0.25, 0.25)  # BLEU-4
            ]
            
            scores = {}
            for i, weights in enumerate(weights_list[:max_ngram], 1):
                score = sentence_bleu(
                    [ref_tokens],
                    cand_tokens,
                    weights=weights,
                    smoothing_function=self.smoothing.method1
                )
                scores[f'bleu{i}'] = score
            
            # BLEU médio
            scores['bleu_avg'] = np.mean(list(scores.values()))
            
            logger.debug(f"BLEU-4: {scores.get('bleu4', 0):.3f}")
            return scores
        
        except Exception as e:
            logger.error(f"Erro ao calcular BLEU: {e}")
            return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0, 'bleu_avg': 0.0}
    
    def avaliar_sumarizacao(
        self,
        referencia: str,
        candidato: str,
        incluir_bleu: bool = True
    ) -> Dict:
        """
        Avaliação completa de uma sumarização
        
        Args:
            referencia: Texto de referência
            candidato: Texto candidato
            incluir_bleu: Se deve calcular BLEU
            
        Returns:
            Dicionário com todas as métricas
        """
        logger.info("Avaliando sumarização")
        
        # ROUGE
        rouge_scores = self.calcular_rouge(referencia, candidato)
        
        # BLEU
        if incluir_bleu:
            bleu_scores = self.calcular_bleu(referencia, candidato)
        else:
            bleu_scores = {}
        
        # Métricas adicionais
        len_ref = len(referencia.split())
        len_cand = len(candidato.split())
        
        resultado = {
            **rouge_scores,
            **bleu_scores,
            'comprimento_referencia': len_ref,
            'comprimento_candidato': len_cand,
            'taxa_compressao': len_cand / len_ref if len_ref > 0 else 0,
            'diferenca_comprimento': abs(len_ref - len_cand)
        }
        
        logger.success("Avaliação concluída")
        return resultado
    
    def avaliar_lote(
        self,
        referencias: List[str],
        candidatos: List[str]
    ) -> pd.DataFrame:
        """
        Avalia múltiplas sumarizações
        
        Args:
            referencias: Lista de textos de referência
            candidatos: Lista de textos candidatos
            
        Returns:
            DataFrame com métricas para cada par
        """
        if len(referencias) != len(candidatos):
            raise ValueError("Número de referências e candidatos deve ser igual")
        
        resultados = []
        
        for i, (ref, cand) in enumerate(zip(referencias, candidatos)):
            metricas = self.avaliar_sumarizacao(ref, cand)
            metricas['id'] = i
            resultados.append(metricas)
        
        df = pd.DataFrame(resultados)
        
        logger.info(f"Avaliadas {len(df)} sumarizações")
        return df
    
    def calcular_metricas_agregadas(self, df_avaliacoes: pd.DataFrame) -> Dict:
        """
        Calcula métricas agregadas de múltiplas avaliações
        
        Args:
            df_avaliacoes: DataFrame com avaliações
            
        Returns:
            Dicionário com médias e desvios padrão
        """
        metricas_numericas = df_avaliacoes.select_dtypes(include=[np.number])
        
        agregadas = {
            'media': metricas_numericas.mean().to_dict(),
            'desvio_padrao': metricas_numericas.std().to_dict(),
            'mediana': metricas_numericas.median().to_dict(),
            'min': metricas_numericas.min().to_dict(),
            'max': metricas_numericas.max().to_dict()
        }
        
        return agregadas
    
    def avaliar_classificacao(
        self,
        y_true: List,
        y_pred: List,
        labels: Optional[List] = None
    ) -> Dict:
        """
        Avalia desempenho de classificador LGPD
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            labels: Lista de labels para relatório
            
        Returns:
            Dicionário com métricas de classificação
        """
        logger.info("Avaliando classificação")
        
        # Acurácia
        acuracia = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Por classe
        precision_classe, recall_classe, f1_classe, support_classe = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0, labels=labels
        )
        
        # Relatório completo
        if labels:
            relatorio = classification_report(
                y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
            )
        else:
            relatorio = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
        
        resultado = {
            'acuracia': acuracia,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_por_classe': precision_classe.tolist() if hasattr(precision_classe, 'tolist') else [],
            'recall_por_classe': recall_classe.tolist() if hasattr(recall_classe, 'tolist') else [],
            'f1_por_classe': f1_classe.tolist() if hasattr(f1_classe, 'tolist') else [],
            'support_por_classe': support_classe.tolist() if hasattr(support_classe, 'tolist') else [],
            'relatorio_completo': relatorio
        }
        
        logger.success(f"Acurácia: {acuracia:.3f}, F1 (weighted): {f1:.3f}")
        return resultado
    
    def gerar_relatorio(
        self,
        metricas: Dict,
        caminho_saida: Path,
        formato: str = 'json'
    ):
        """
        Gera relatório de avaliação em arquivo
        
        Args:
            metricas: Dicionário com métricas
            caminho_saida: Caminho do arquivo de saída
            formato: Formato ('json', 'txt')
        """
        caminho_saida = Path(caminho_saida)
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        
        if formato == 'json':
            with open(caminho_saida, 'w', encoding='utf-8') as f:
                json.dump(metricas, f, indent=2, ensure_ascii=False)
        
        elif formato == 'txt':
            with open(caminho_saida, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("RELATÓRIO DE AVALIAÇÃO\n")
                f.write("=" * 60 + "\n\n")
                
                for chave, valor in metricas.items():
                    if isinstance(valor, (int, float)):
                        f.write(f"{chave}: {valor:.4f}\n")
                    else:
                        f.write(f"{chave}: {valor}\n")
        
        logger.success(f"Relatório salvo em {caminho_saida}")
    
    def comparar_metodos(
        self,
        referencias: List[str],
        candidatos_dict: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Compara diferentes métodos de sumarização
        
        Args:
            referencias: Lista de textos de referência
            candidatos_dict: Dicionário {nome_metodo: lista_candidatos}
            
        Returns:
            DataFrame comparando os métodos
        """
        resultados = []
        
        for metodo, candidatos in candidatos_dict.items():
            df_avaliacoes = self.avaliar_lote(referencias, candidatos)
            agregadas = self.calcular_metricas_agregadas(df_avaliacoes)
            
            linha = {'metodo': metodo}
            linha.update(agregadas['media'])
            resultados.append(linha)
        
        df_comparacao = pd.DataFrame(resultados)
        
        logger.info(f"Comparados {len(candidatos_dict)} métodos")
        return df_comparacao
    
    def obter_texto_referencia_lgpd(self) -> str:
        """
        Obtém texto de referência do Gold Standard Universal LGPD
        
        Returns:
            Texto de referência representando política ideal (completa)
        """
        if self._texto_referencia_lgpd is None:
            self._texto_referencia_lgpd = self.gold_standard_lgpd.gerar_texto_referencia()
            logger.info("Texto de referência LGPD gerado")
        
        return self._texto_referencia_lgpd
    
    def obter_resumo_referencia_lgpd(self) -> str:
        """
        Obtém resumo de referência do Gold Standard Universal LGPD
        Este é um resumo conciso e bem estruturado, como um humano escreveria
        Usado para métricas ROUGE/BLEU em sumarizações
        
        Returns:
            Resumo de referência representando sumário ideal
        """
        if self._resumo_referencia_lgpd is None:
            self._resumo_referencia_lgpd = self.gold_standard_lgpd.gerar_resumo_referencia()
            logger.info("Resumo de referência LGPD gerado")
        
        return self._resumo_referencia_lgpd
    
    def avaliar_contra_gold_standard_lgpd(
        self,
        candidato: str,
        incluir_bleu: bool = True
    ) -> Dict:
        """
        Avalia um sumário/política contra o Gold Standard Universal LGPD
        
        Args:
            candidato: Texto a ser avaliado
            incluir_bleu: Se deve calcular BLEU
            
        Returns:
            Dicionário com métricas
        """
        logger.info("Avaliando contra Gold Standard Universal LGPD")
        
        # Usar RESUMO de referência para métricas ROUGE/BLEU (não o texto completo)
        # Isso permite comparação justa entre sumários
        referencia = self.obter_resumo_referencia_lgpd()
        
        # Avaliar normalmente
        metricas = self.avaliar_sumarizacao(referencia, candidato, incluir_bleu)
        
        # Adicionar informação sobre uso do Gold Standard
        metricas['tipo_referencia'] = 'gold_standard_resumo_lgpd'
        metricas['num_requisitos_lgpd'] = len(self.gold_standard_lgpd.requisitos)
        
        logger.success("Avaliação contra Gold Standard concluída")
        return metricas
    
    def avaliar_cobertura_requisitos_lgpd(
        self,
        texto_politica: str
    ) -> Dict:
        """
        Avalia a cobertura dos requisitos LGPD em um texto
        
        Args:
            texto_politica: Texto da política a avaliar
            
        Returns:
            Dicionário com análise de cobertura
        """
        logger.info("Avaliando cobertura de requisitos LGPD")
        
        texto_lower = texto_politica.lower()
        
        requisitos_atendidos = []
        requisitos_nao_atendidos = []
        cobertura_por_categoria = {}
        
        for categoria in self.gold_standard_lgpd.obter_categorias():
            requisitos_cat = self.gold_standard_lgpd.obter_requisitos_por_categoria(categoria)
            
            atendidos_cat = 0
            total_cat = len(requisitos_cat)
            
            for req in requisitos_cat:
                # Verificar presença de palavras-chave (com variações morfológicas)
                palavras_encontradas = 0
                for palavra in req.palavras_chave:
                    palavra_lower = palavra.lower()
                    
                    # Matching exato
                    if palavra_lower in texto_lower:
                        palavras_encontradas += 1
                    # Matching por raiz (primeiros 5 caracteres) para pegar variações
                    # Ex: "excluir" -> "exclu" matches "exclusão"
                    elif len(palavra_lower) >= 5:
                        raiz = palavra_lower[:5]
                        if raiz in texto_lower:
                            palavras_encontradas += 1
                
                # Considerar atendido se encontrar pelo menos 20% das palavras (mais flexível)
                threshold = max(1, len(req.palavras_chave) * 0.2)
                
                if palavras_encontradas >= threshold:
                    requisitos_atendidos.append(req.id)
                    atendidos_cat += 1
                else:
                    requisitos_nao_atendidos.append(req.id)
            
            # Percentual de cobertura da categoria
            percentual = (atendidos_cat / total_cat * 100) if total_cat > 0 else 0
            
            cobertura_por_categoria[categoria] = {
                'total': total_cat,
                'atendidos': atendidos_cat,
                'percentual': percentual
            }
        
        # Cobertura geral
        total_requisitos = len(self.gold_standard_lgpd.obter_requisitos_obrigatorios())
        cobertura_geral = (len(requisitos_atendidos) / total_requisitos * 100) if total_requisitos > 0 else 0
        
        resultado = {
            'cobertura_geral_percentual': cobertura_geral,
            'requisitos_atendidos': requisitos_atendidos,
            'requisitos_nao_atendidos': requisitos_nao_atendidos,
            'num_atendidos': len(requisitos_atendidos),
            'num_nao_atendidos': len(requisitos_nao_atendidos),
            'cobertura_por_categoria': cobertura_por_categoria
        }
        
        logger.success(f"Cobertura LGPD: {cobertura_geral:.1f}%")
        return resultado
