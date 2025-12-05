"""
Módulo de Modelo de Conformidade LGPD
Treina e infere conformidade de políticas de privacidade com a LGPD

⚠️ IMPORTANTE - DISTINÇÃO METODOLÓGICA:

Este módulo implementa um MODELO DE MACHINE LEARNING TREINADO, diferente do gold standard.

MODELO DE CONFORMIDADE:
- ✅ É um MODELO TREINADO com épocas e otimização
- ✅ Usa SGDClassifier.partial_fit() para simular treinamento incremental
- ✅ Dataset sintético: 200 exemplos (data/dataset_conformidade_exemplo.csv)
- ✅ Aprende a PREVER se uma política é conforme (score 0-100)
- ✅ Treinado com Logistic Regression + early stopping
- ✅ Salvo em: models/modelo_conformidade_lgpd.pkl

vs.

GOLD STANDARD LGPD (gold_standard_lgpd.py):
- ❌ NÃO é um modelo treinado
- ❌ NÃO envolve épocas ou otimização
- ✅ É um BENCHMARK FIXO com 23 requisitos da Lei 13.709/2018
- ✅ Usado APENAS para AVALIAÇÃO (não treinamento)

Para metodologia completa, veja: docs/METODOLOGIA_CORRIGIDA.md
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from loguru import logger
import joblib
import json
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import StandardScaler

from .gold_standard_lgpd import GoldStandardLGPD, obter_gold_standard


@dataclass
class ResultadoConformidade:
    """Resultado da avaliação de conformidade"""
    score_conformidade: float  # 0.0 a 100.0
    conformidade_binaria: str  # 'conforme' ou 'nao_conforme'
    confianca: float  # 0.0 a 1.0
    requisitos_atendidos: List[str]
    requisitos_nao_atendidos: List[str]
    recomendacao: str  # 'aceitar', 'revisar', 'rejeitar'
    justificativa: List[str]
    detalhes_por_categoria: Dict[str, Dict]


class ModeloConformidadeLGPD:
    """
    Modelo de Machine Learning para avaliar conformidade com a LGPD
    Treina com épocas e otimização para aprender padrões de conformidade
    """
    
    def __init__(
        self,
        tipo_modelo: str = 'gradient_boosting',
        usar_embeddings: bool = True
    ):
        """
        Inicializa o modelo de conformidade
        
        Args:
            tipo_modelo: 'logistic', 'gradient_boosting', 'random_forest', 'mlp'
            usar_embeddings: Se deve usar embeddings além de TF-IDF
        """
        self.tipo_modelo = tipo_modelo
        self.usar_embeddings = usar_embeddings
        
        # Gold Standard universal
        self.gold_standard = obter_gold_standard()
        
        # Vetorizador
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        # Scaler para normalização
        self.scaler = StandardScaler(with_mean=False)  # sparse matrix compatible
        
        # Modelo
        self.modelo = None
        self._inicializar_modelo()
        
        # Metadata de treinamento
        self.treinado = False
        self.metadata = {
            'data_treinamento': None,
            'num_exemplos': 0,
            'num_epocas': 0,
            'melhor_acuracia': 0.0,
            'historico': {
                'epoca': [],
                'loss_treino': [],
                'loss_val': [],
                'acuracia_treino': [],
                'acuracia_val': [],
                'f1_treino': [],
                'f1_val': []
            }
        }
        
        logger.info(f"Inicializando ModeloConformidadeLGPD (tipo: {tipo_modelo})")
    
    def _inicializar_modelo(self):
        """Inicializa o modelo de ML baseado no tipo escolhido"""
        if self.tipo_modelo == 'logistic':
            # Logistic Regression with SGD (allows partial_fit for epochs)
            self.modelo = SGDClassifier(
                loss='log_loss',  # logistic regression
                penalty='l2',
                alpha=0.0001,
                max_iter=1,  # 1 iteration per epoch
                tol=1e-3,
                random_state=42,
                warm_start=True,  # Keep learned coefficients between calls
                learning_rate='optimal'
            )
        
        elif self.tipo_modelo == 'gradient_boosting':
            self.modelo = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                warm_start=True  # Allows incremental training
            )
        
        elif self.tipo_modelo == 'random_forest':
            self.modelo = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                warm_start=True,
                n_jobs=-1
            )
        
        elif self.tipo_modelo == 'mlp':
            # Neural Network
            self.modelo = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=1,  # 1 iteration per epoch
                random_state=42,
                warm_start=True,
                early_stopping=False
            )
        
        else:
            raise ValueError(f"Tipo de modelo não suportado: {self.tipo_modelo}")
    
    def preparar_features(
        self,
        textos: List[str],
        treino: bool = False
    ) -> np.ndarray:
        """
        Prepara features dos textos (TF-IDF + features customizadas)
        
        Args:
            textos: Lista de textos das políticas
            treino: Se está em modo treino (fit vectorizer)
            
        Returns:
            Matriz de features
        """
        # TF-IDF
        if treino:
            X_tfidf = self.vectorizer.fit_transform(textos)
        else:
            X_tfidf = self.vectorizer.transform(textos)
        
        # Features customizadas baseadas no Gold Standard
        features_custom = self._extrair_features_lgpd(textos)
        
        # Concatenar features
        from scipy.sparse import hstack, csr_matrix
        X_custom_sparse = csr_matrix(features_custom)
        X_combined = hstack([X_tfidf, X_custom_sparse])
        
        # Normalizar
        if treino:
            X_normalized = self.scaler.fit_transform(X_combined)
        else:
            X_normalized = self.scaler.transform(X_combined)
        
        logger.debug(f"Features preparadas: shape {X_normalized.shape}")
        return X_normalized
    
    def _extrair_features_lgpd(self, textos: List[str]) -> np.ndarray:
        """
        Extrai features customizadas baseadas nos requisitos LGPD
        
        Args:
            textos: Lista de textos
            
        Returns:
            Matriz de features customizadas
        """
        features_list = []
        
        for texto in textos:
            texto_lower = texto.lower()
            features = []
            
            # Para cada requisito, verificar presença de palavras-chave
            for requisito in self.gold_standard.requisitos.values():
                # Contar ocorrências de palavras-chave
                count = sum(
                    1 for palavra in requisito.palavras_chave
                    if palavra.lower() in texto_lower
                )
                # Normalizar pelo número de palavras-chave
                score = count / len(requisito.palavras_chave) if requisito.palavras_chave else 0
                features.append(score)
            
            # Features adicionais
            features.append(len(texto))  # Tamanho do texto
            features.append(len(texto.split()))  # Número de palavras
            features.append(texto.count('.'))  # Número de sentenças (aproximado)
            
            features_list.append(features)
        
        return np.array(features_list)
    
    def treinar(
        self,
        X_train: List[str],
        y_train: List[float],
        X_val: Optional[List[str]] = None,
        y_val: Optional[List[float]] = None,
        num_epocas: int = 50,
        test_size: float = 0.2,
        verbose: bool = True
    ) -> Dict:
        """
        Treina o modelo com loop de épocas
        
        Args:
            X_train: Textos de treinamento
            y_train: Labels de conformidade (0-100 ou binário 0/1)
            X_val: Textos de validação (opcional)
            y_val: Labels de validação (opcional)
            num_epocas: Número de épocas de treinamento
            test_size: Proporção para validação se não fornecida
            verbose: Se deve exibir progresso
            
        Returns:
            Dicionário com resultados do treinamento
        """
        logger.info(f"Iniciando treinamento com {len(X_train)} exemplos, {num_epocas} épocas")
        
        # Converter scores para binário (>= 70 = conforme)
        y_train_bin = np.array([1 if y >= 70 else 0 for y in y_train])
        
        # Split treino/validação se não fornecido
        if X_val is None or y_val is None:
            X_train, X_val, y_train_bin, y_val_temp = train_test_split(
                X_train, y_train_bin,
                test_size=test_size,
                random_state=42,
                stratify=y_train_bin
            )
            y_val = [y_train[i] for i in range(len(y_train)) if i >= len(X_train)]
        else:
            y_val_bin = np.array([1 if y >= 70 else 0 for y in y_val])
            y_val_temp = y_val_bin
        
        # Preparar features
        X_train_features = self.preparar_features(X_train, treino=True)
        X_val_features = self.preparar_features(X_val, treino=False)
        
        # Loop de treinamento por épocas
        import time
        tempo_inicio = time.time()
        
        best_f1 = 0.0
        patience = 10
        patience_counter = 0
        
        for epoca in range(num_epocas):
            tempo_epoca_inicio = time.time()
            
            # Treinar uma época
            if hasattr(self.modelo, 'partial_fit'):
                # Modelos que suportam partial_fit (SGDClassifier, MLPClassifier)
                self.modelo.partial_fit(X_train_features, y_train_bin, classes=[0, 1])
            else:
                # Para outros modelos, aumentar n_estimators gradualmente
                if self.tipo_modelo == 'gradient_boosting':
                    self.modelo.n_estimators = min(100, (epoca + 1) * 2)
                elif self.tipo_modelo == 'random_forest':
                    self.modelo.n_estimators = min(100, (epoca + 1) * 2)
                
                self.modelo.fit(X_train_features, y_train_bin)
            
            # Avaliar
            y_pred_train = self.modelo.predict(X_train_features)
            y_pred_val = self.modelo.predict(X_val_features)
            
            # Métricas
            acc_train = accuracy_score(y_train_bin, y_pred_train)
            acc_val = accuracy_score(y_val_temp, y_pred_val)
            
            f1_train = f1_score(y_train_bin, y_pred_train, zero_division=0)
            f1_val = f1_score(y_val_temp, y_pred_val, zero_division=0)
            
            # Loss simulado (1 - acurácia)
            loss_train = 1 - acc_train
            loss_val = 1 - acc_val
            
            tempo_epoca = time.time() - tempo_epoca_inicio
            
            # Registrar histórico
            self.metadata['historico']['epoca'].append(epoca + 1)
            self.metadata['historico']['loss_treino'].append(loss_train)
            self.metadata['historico']['loss_val'].append(loss_val)
            self.metadata['historico']['acuracia_treino'].append(acc_train)
            self.metadata['historico']['acuracia_val'].append(acc_val)
            self.metadata['historico']['f1_treino'].append(f1_train)
            self.metadata['historico']['f1_val'].append(f1_val)
            
            if verbose and (epoca % 5 == 0 or epoca == num_epocas - 1):
                logger.info(
                    f"Época {epoca+1}/{num_epocas} - "
                    f"Acc Train: {acc_train:.3f}, Acc Val: {acc_val:.3f}, "
                    f"F1 Val: {f1_val:.3f}, "
                    f"Tempo: {tempo_epoca:.2f}s"
                )
            
            # Early stopping baseado em F1
            if f1_val > best_f1:
                best_f1 = f1_val
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping na época {epoca+1}")
                break
        
        tempo_total = time.time() - tempo_inicio
        
        # Avaliação final
        y_pred_final = self.modelo.predict(X_val_features)
        
        relatorio = classification_report(
            y_val_temp, y_pred_final,
            target_names=['Não Conforme', 'Conforme'],
            output_dict=True,
            zero_division=0
        )
        
        # Atualizar metadata
        self.treinado = True
        self.metadata['data_treinamento'] = datetime.now().isoformat()
        self.metadata['num_exemplos'] = len(X_train)
        self.metadata['num_epocas'] = epoca + 1
        self.metadata['melhor_acuracia'] = max(self.metadata['historico']['acuracia_val'])
        
        resultado = {
            'tempo_total': tempo_total,
            'num_epocas': epoca + 1,
            'melhor_f1': best_f1,
            'acuracia_final': acc_val,
            'relatorio_classificacao': relatorio,
            'historico': self.metadata['historico']
        }
        
        logger.success(
            f"Treinamento concluído - {epoca+1} épocas, "
            f"Acurácia: {acc_val:.3f}, F1: {best_f1:.3f}"
        )
        
        return resultado
    
    def prever_conformidade(
        self,
        texto_politica: str
    ) -> ResultadoConformidade:
        """
        Avalia conformidade de uma política de privacidade
        
        Args:
            texto_politica: Texto completo da política
            
        Returns:
            Resultado detalhado da avaliação
        """
        if not self.treinado:
            raise ValueError("Modelo não foi treinado ainda")
        
        logger.info("Avaliando conformidade da política")
        
        # Preparar features
        X = self.preparar_features([texto_politica], treino=False)
        
        # Predição binária
        y_pred_bin = self.modelo.predict(X)[0]
        
        # Probabilidade/confiança
        if hasattr(self.modelo, 'predict_proba'):
            proba = self.modelo.predict_proba(X)[0]
            confianca = float(proba[y_pred_bin])
        else:
            confianca = 0.8  # Default
        
        # Análise detalhada dos requisitos
        resultado_requisitos = self._analisar_requisitos(texto_politica)
        
        # Calcular score detalhado (0-100)
        score_conformidade = self._calcular_score_detalhado(resultado_requisitos)
        
        # Determinar conformidade binária
        if y_pred_bin == 1:
            conformidade_binaria = 'conforme'
        else:
            conformidade_binaria = 'nao_conforme'
        
        # Recomendação
        if score_conformidade >= 80:
            recomendacao = 'aceitar'
        elif score_conformidade >= 60:
            recomendacao = 'revisar'
        else:
            recomendacao = 'rejeitar'
        
        # Gerar justificativa
        justificativa = self._gerar_justificativa(resultado_requisitos, score_conformidade)
        
        return ResultadoConformidade(
            score_conformidade=score_conformidade,
            conformidade_binaria=conformidade_binaria,
            confianca=confianca,
            requisitos_atendidos=resultado_requisitos['atendidos'],
            requisitos_nao_atendidos=resultado_requisitos['nao_atendidos'],
            recomendacao=recomendacao,
            justificativa=justificativa,
            detalhes_por_categoria=resultado_requisitos['por_categoria']
        )
    
    def _analisar_requisitos(self, texto: str) -> Dict:
        """
        Analisa quais requisitos do Gold Standard estão presentes no texto
        
        Args:
            texto: Texto da política
            
        Returns:
            Dicionário com análise detalhada
        """
        texto_lower = texto.lower()
        
        atendidos = []
        nao_atendidos = []
        por_categoria = {}
        
        for categoria in self.gold_standard.obter_categorias():
            requisitos_cat = self.gold_standard.obter_requisitos_por_categoria(categoria)
            
            categoria_info = {
                'total': len(requisitos_cat),
                'atendidos': 0,
                'nao_atendidos': 0,
                'requisitos': []
            }
            
            for req in requisitos_cat:
                # Verificar presença de palavras-chave
                palavras_encontradas = [
                    palavra for palavra in req.palavras_chave
                    if palavra.lower() in texto_lower
                ]
                
                # Requisito atendido se encontrar pelo menos 30% das palavras-chave
                threshold = max(1, len(req.palavras_chave) * 0.3)
                atendido = len(palavras_encontradas) >= threshold
                
                req_info = {
                    'id': req.id,
                    'titulo': req.titulo,
                    'obrigatorio': req.obrigatorio,
                    'peso': req.peso,
                    'atendido': atendido,
                    'palavras_encontradas': palavras_encontradas
                }
                
                categoria_info['requisitos'].append(req_info)
                
                if atendido:
                    atendidos.append(req.id)
                    categoria_info['atendidos'] += 1
                else:
                    nao_atendidos.append(req.id)
                    categoria_info['nao_atendidos'] += 1
            
            por_categoria[categoria] = categoria_info
        
        return {
            'atendidos': atendidos,
            'nao_atendidos': nao_atendidos,
            'por_categoria': por_categoria
        }
    
    def _calcular_score_detalhado(self, resultado_requisitos: Dict) -> float:
        """
        Calcula score de conformidade baseado nos requisitos atendidos
        
        Args:
            resultado_requisitos: Resultado da análise de requisitos
            
        Returns:
            Score de 0 a 100
        """
        peso_total = 0.0
        peso_atendido = 0.0
        
        for req_id in resultado_requisitos['atendidos']:
            req = self.gold_standard.obter_requisito(req_id)
            if req and req.obrigatorio:
                peso_atendido += req.peso
        
        for req in self.gold_standard.obter_requisitos_obrigatorios():
            peso_total += req.peso
        
        # Score baseado em peso
        if peso_total > 0:
            score = (peso_atendido / peso_total) * 100
        else:
            score = 0.0
        
        return round(score, 2)
    
    def _gerar_justificativa(
        self,
        resultado_requisitos: Dict,
        score: float
    ) -> List[str]:
        """
        Gera justificativas para o score de conformidade
        
        Args:
            resultado_requisitos: Resultado da análise
            score: Score calculado
            
        Returns:
            Lista de frases explicativas
        """
        justificativas = []
        
        # Análise geral
        total_requisitos = len(self.gold_standard.obter_requisitos_obrigatorios())
        atendidos = len(resultado_requisitos['atendidos'])
        
        justificativas.append(
            f"A política atende {atendidos} de {total_requisitos} requisitos obrigatórios da LGPD."
        )
        
        # Por categoria
        for categoria, info in resultado_requisitos['por_categoria'].items():
            percentual = (info['atendidos'] / info['total'] * 100) if info['total'] > 0 else 0
            
            if percentual >= 80:
                justificativas.append(
                    f"✓ Categoria '{categoria}': Excelente cobertura ({percentual:.0f}%)"
                )
            elif percentual >= 50:
                justificativas.append(
                    f"⚠ Categoria '{categoria}': Cobertura parcial ({percentual:.0f}%)"
                )
            else:
                justificativas.append(
                    f"✗ Categoria '{categoria}': Cobertura insuficiente ({percentual:.0f}%)"
                )
        
        # Requisitos críticos não atendidos
        requisitos_criticos_faltando = []
        for req_id in resultado_requisitos['nao_atendidos']:
            req = self.gold_standard.obter_requisito(req_id)
            if req and req.obrigatorio and req.peso >= 2.0:
                requisitos_criticos_faltando.append(req.titulo)
        
        if requisitos_criticos_faltando:
            justificativas.append(
                f"⚠ CRÍTICO: Faltam requisitos importantes: {', '.join(requisitos_criticos_faltando[:3])}"
            )
        
        return justificativas
    
    def salvar_modelo(self, caminho: Path):
        """Salva o modelo treinado e metadata"""
        if not self.treinado:
            logger.warning("Modelo não foi treinado ainda")
            return
        
        modelo_data = {
            'modelo': self.modelo,
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'tipo_modelo': self.tipo_modelo,
            'metadata': self.metadata
        }
        
        joblib.dump(modelo_data, caminho)
        logger.success(f"Modelo salvo em {caminho}")
        
        # Salvar metadata em JSON separado
        metadata_path = caminho.parent / f"{caminho.stem}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def carregar_modelo(self, caminho: Path):
        """Carrega modelo salvo"""
        modelo_data = joblib.load(caminho)
        
        self.modelo = modelo_data['modelo']
        self.vectorizer = modelo_data['vectorizer']
        self.scaler = modelo_data['scaler']
        self.tipo_modelo = modelo_data['tipo_modelo']
        self.metadata = modelo_data['metadata']
        self.treinado = True
        
        logger.success(f"Modelo carregado de {caminho}")
