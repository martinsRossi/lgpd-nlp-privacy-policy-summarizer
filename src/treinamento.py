"""
Módulo de Treinamento de Modelos
Treina classificador LGPD e gera gráficos de convergência, matriz de confusão e métricas
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, ConfusionMatrixDisplay
)
from datetime import datetime


def converter_para_json_serializavel(obj: Any) -> Any:
    """
    Converte tipos NumPy/pandas para tipos Python nativos (JSON serializáveis)
    
    Args:
        obj: Objeto a ser convertido
        
    Returns:
        Objeto convertido para tipo Python nativo
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: converter_para_json_serializavel(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [converter_para_json_serializavel(item) for item in obj]
    else:
        return obj


class TreinadorModelo:
    """Classe para treinamento e avaliação de modelos com geração de métricas"""
    
    def __init__(self, diretorio_saida: Path):
        """
        Inicializa o treinador
        
        Args:
            diretorio_saida: Diretório para salvar gráficos e métricas
        """
        self.dir_saida = Path(diretorio_saida)
        self.dir_saida.mkdir(parents=True, exist_ok=True)
        
        self.historico = {
            'epoca': [],
            'loss_treino': [],
            'loss_val': [],
            'acuracia_treino': [],
            'acuracia_val': [],
            'tempo_epoca': []
        }
        
        logger.info(f"Inicializando TreinadorModelo (saída: {self.dir_saida})")
    
    def treinar_com_monitoramento(
        self,
        modelo,
        X_train, y_train,
        X_val, y_val,
        num_epocas: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Treina modelo com monitoramento de métricas por época
        
        Args:
            modelo: Modelo sklearn
            X_train, y_train: Dados de treino
            X_val, y_val: Dados de validação
            num_epocas: Número de épocas (para modelos iterativos)
            verbose: Se deve imprimir progresso
            
        Returns:
            Dicionário com histórico de treinamento
        """
        logger.info(f"Iniciando treinamento com {num_epocas} épocas")
        tempo_inicio = time.time()
        
        # Para modelos sklearn que não têm treinamento iterativo,
        # simularemos épocas com partial_fit ou retreinamento
        for epoca in range(num_epocas):
            tempo_epoca_inicio = time.time()
            
            # Treinar modelo
            if hasattr(modelo, 'partial_fit'):
                modelo.partial_fit(X_train, y_train)
            else:
                modelo.fit(X_train, y_train)
            
            # Calcular métricas de treino
            y_pred_train = modelo.predict(X_train)
            acuracia_train = accuracy_score(y_train, y_pred_train)
            
            # Calcular métricas de validação
            y_pred_val = modelo.predict(X_val)
            acuracia_val = accuracy_score(y_val, y_pred_val)
            
            # Simular loss (1 - acurácia)
            loss_train = 1 - acuracia_train
            loss_val = 1 - acuracia_val
            
            tempo_epoca = time.time() - tempo_epoca_inicio
            
            # Registrar histórico
            self.historico['epoca'].append(epoca + 1)
            self.historico['loss_treino'].append(loss_train)
            self.historico['loss_val'].append(loss_val)
            self.historico['acuracia_treino'].append(acuracia_train)
            self.historico['acuracia_val'].append(acuracia_val)
            self.historico['tempo_epoca'].append(tempo_epoca)
            
            if verbose:
                logger.info(
                    f"Época {epoca+1}/{num_epocas} - "
                    f"Loss Treino: {loss_train:.4f}, Loss Val: {loss_val:.4f}, "
                    f"Acurácia Treino: {acuracia_train:.4f}, Acurácia Val: {acuracia_val:.4f}, "
                    f"Tempo: {tempo_epoca:.2f}s"
                )
        
        tempo_total = time.time() - tempo_inicio
        
        # Identificar melhor época
        melhor_epoca = np.argmax(self.historico['acuracia_val']) + 1
        melhor_acuracia = max(self.historico['acuracia_val'])
        
        resultado = {
            'historico': self.historico,
            'melhor_epoca': melhor_epoca,
            'melhor_acuracia_val': melhor_acuracia,
            'tempo_total': tempo_total,
            'tempo_medio_epoca': tempo_total / num_epocas,
            'convergiu': self._verificar_convergencia()
        }
        
        logger.success(
            f"Treinamento concluído em {tempo_total:.2f}s - "
            f"Melhor época: {melhor_epoca} (Acurácia: {melhor_acuracia:.4f})"
        )
        
        return resultado
    
    def _verificar_convergencia(self, threshold: float = 0.05, janela: int = 3) -> bool:
        """
        Verifica se o treinamento convergiu
        
        Args:
            threshold: Diferença máxima aceitável
            janela: Número de épocas para verificar
            
        Returns:
            True se convergiu
        """
        if len(self.historico['loss_treino']) < janela:
            return False
        
        ultimos_losses = self.historico['loss_treino'][-janela:]
        diferenca_max = max(ultimos_losses) - min(ultimos_losses)
        
        return diferenca_max < threshold
    
    def plotar_curvas_treinamento(
        self,
        salvar: bool = True,
        mostrar: bool = False
    ):
        """
        Gera gráficos de curvas de treinamento (loss e acurácia)
        
        Args:
            salvar: Se deve salvar os gráficos
            mostrar: Se deve exibir os gráficos
        """
        logger.info("Gerando gráficos de treinamento")
        
        # Configurar estilo
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 5)
        
        # Criar figura com 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epocas = self.historico['epoca']
        
        # Gráfico 1: Loss
        ax1.plot(epocas, self.historico['loss_treino'], 
                label='Treino', marker='o', linewidth=2)
        ax1.plot(epocas, self.historico['loss_val'], 
                label='Validação', marker='s', linewidth=2)
        ax1.set_xlabel('Época', fontsize=12)
        ax1.set_ylabel('Loss (Função de Perda)', fontsize=12)
        ax1.set_title('Convergência do Treinamento - Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Acurácia
        ax2.plot(epocas, self.historico['acuracia_treino'], 
                label='Treino', marker='o', linewidth=2)
        ax2.plot(epocas, self.historico['acuracia_val'], 
                label='Validação', marker='s', linewidth=2)
        ax2.set_xlabel('Época', fontsize=12)
        ax2.set_ylabel('Acurácia', fontsize=12)
        ax2.set_title('Convergência do Treinamento - Acurácia', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if salvar:
            # Salvar gráfico de loss
            fig_loss = plt.figure(figsize=(10, 6))
            plt.plot(epocas, self.historico['loss_treino'], 
                    label='Treino', marker='o', linewidth=2, markersize=6)
            plt.plot(epocas, self.historico['loss_val'], 
                    label='Validação', marker='s', linewidth=2, markersize=6)
            plt.xlabel('Época', fontsize=13)
            plt.ylabel('Loss (Função de Perda)', fontsize=13)
            plt.title('Figura 6 - Convergência do Treinamento: Função de Perda', 
                     fontsize=14, fontweight='bold')
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            caminho_loss = self.dir_saida / 'figura_6_treinamento_loss.png'
            plt.savefig(caminho_loss, dpi=300, bbox_inches='tight')
            logger.success(f"Gráfico de loss salvo: {caminho_loss}")
            plt.close()
            
            # Salvar gráfico de acurácia
            fig_acc = plt.figure(figsize=(10, 6))
            plt.plot(epocas, self.historico['acuracia_treino'], 
                    label='Treino', marker='o', linewidth=2, markersize=6)
            plt.plot(epocas, self.historico['acuracia_val'], 
                    label='Validação', marker='s', linewidth=2, markersize=6)
            plt.xlabel('Época', fontsize=13)
            plt.ylabel('Acurácia', fontsize=13)
            plt.title('Figura 7 - Convergência do Treinamento: Acurácia', 
                     fontsize=14, fontweight='bold')
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1])
            caminho_acc = self.dir_saida / 'figura_7_treinamento_acuracia.png'
            plt.savefig(caminho_acc, dpi=300, bbox_inches='tight')
            logger.success(f"Gráfico de acurácia salvo: {caminho_acc}")
            plt.close()
        
        if mostrar:
            plt.show()
    
    def plotar_matriz_confusao(
        self,
        y_true: List,
        y_pred: List,
        labels: List[str],
        salvar: bool = True,
        mostrar: bool = False
    ):
        """
        Gera matriz de confusão
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            labels: Nomes das classes
            salvar: Se deve salvar
            mostrar: Se deve exibir
        """
        logger.info("Gerando matriz de confusão")
        
        # Calcular matriz
        cm = confusion_matrix(y_true, y_pred)
        
        # Plotar
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Usar ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels
        )
        
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title('Figura 8 - Matriz de Confusão do Classificador LGPD', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Classe Predita', fontsize=12)
        ax.set_ylabel('Classe Verdadeira', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if salvar:
            caminho = self.dir_saida / 'figura_8_matriz_confusao.png'
            plt.savefig(caminho, dpi=300, bbox_inches='tight')
            logger.success(f"Matriz de confusão salva: {caminho}")
        
        if mostrar:
            plt.show()
        else:
            plt.close()
    
    def gerar_relatorio_metricas(
        self,
        y_true: List,
        y_pred: List,
        labels: List[str],
        resultado_treinamento: Dict
    ) -> Dict:
        """
        Gera relatório completo com todas as métricas
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            labels: Nomes das classes
            resultado_treinamento: Dicionário do treinamento
            
        Returns:
            Dicionário com relatório completo
        """
        logger.info("Gerando relatório de métricas")
        
        # Métricas globais
        acuracia = accuracy_score(y_true, y_pred)
        
        # Relatório por classe
        relatorio_classe = classification_report(
            y_true, y_pred,
            target_names=labels,
            output_dict=True,
            zero_division=0
        )
        
        # Montar relatório
        relatorio = {
            'timestamp': datetime.now().isoformat(),
            'treinamento': {
                'num_epocas': len(self.historico['epoca']),
                'melhor_epoca': resultado_treinamento['melhor_epoca'],
                'tempo_total_segundos': resultado_treinamento['tempo_total'],
                'tempo_medio_epoca_segundos': resultado_treinamento['tempo_medio_epoca'],
                'convergiu': resultado_treinamento['convergiu']
            },
            'metricas_globais': {
                'acuracia': acuracia,
                'acuracia_treino_final': self.historico['acuracia_treino'][-1],
                'acuracia_val_final': self.historico['acuracia_val'][-1],
                'loss_treino_final': self.historico['loss_treino'][-1],
                'loss_val_final': self.historico['loss_val'][-1]
            },
            'metricas_por_classe': relatorio_classe,
            'historico_completo': self.historico
        }
        
        # Converter para tipos JSON serializáveis
        relatorio = converter_para_json_serializavel(relatorio)
        
        # Salvar em JSON
        caminho_json = self.dir_saida / 'avaliacao_modelo.json'
        with open(caminho_json, 'w', encoding='utf-8') as f:
            json.dump(relatorio, f, indent=2, ensure_ascii=False)
        logger.success(f"Relatório salvo: {caminho_json}")
        
        # Criar versão texto para fácil leitura
        self._salvar_relatorio_texto(relatorio)
        
        return relatorio
    
    def _salvar_relatorio_texto(self, relatorio: Dict):
        """Salva versão texto do relatório"""
        caminho_txt = self.dir_saida / 'avaliacao_modelo.txt'
        
        with open(caminho_txt, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RELATÓRIO DE TREINAMENTO E AVALIAÇÃO DO MODELO\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Data/Hora: {relatorio['timestamp']}\n\n")
            
            f.write("--- INFORMAÇÕES DE TREINAMENTO ---\n")
            f.write(f"Número de Épocas: {relatorio['treinamento']['num_epocas']}\n")
            f.write(f"Melhor Época: {relatorio['treinamento']['melhor_epoca']}\n")
            f.write(f"Tempo Total: {relatorio['treinamento']['tempo_total_segundos']:.2f}s\n")
            f.write(f"Tempo Médio por Época: {relatorio['treinamento']['tempo_medio_epoca_segundos']:.2f}s\n")
            f.write(f"Convergiu: {'Sim' if relatorio['treinamento']['convergiu'] else 'Não'}\n\n")
            
            f.write("--- MÉTRICAS GLOBAIS ---\n")
            f.write(f"Acurácia (Teste): {relatorio['metricas_globais']['acuracia']:.4f}\n")
            f.write(f"Acurácia Treino (Final): {relatorio['metricas_globais']['acuracia_treino_final']:.4f}\n")
            f.write(f"Acurácia Validação (Final): {relatorio['metricas_globais']['acuracia_val_final']:.4f}\n")
            f.write(f"Loss Treino (Final): {relatorio['metricas_globais']['loss_treino_final']:.4f}\n")
            f.write(f"Loss Validação (Final): {relatorio['metricas_globais']['loss_val_final']:.4f}\n\n")
            
            f.write("--- MÉTRICAS POR CLASSE ---\n")
            for classe, metricas in relatorio['metricas_por_classe'].items():
                if isinstance(metricas, dict) and 'precision' in metricas:
                    f.write(f"\nClasse: {classe}\n")
                    f.write(f"  Precisão: {metricas['precision']:.4f}\n")
                    f.write(f"  Recall: {metricas['recall']:.4f}\n")
                    f.write(f"  F1-Score: {metricas['f1-score']:.4f}\n")
                    f.write(f"  Suporte: {metricas['support']}\n")
        
        logger.success(f"Relatório texto salvo: {caminho_txt}")
    
    def plotar_metricas_barras(
        self,
        metricas_dict: Dict,
        salvar: bool = True
    ):
        """
        Plota métricas em gráfico de barras
        
        Args:
            metricas_dict: Dicionário com métricas
            salvar: Se deve salvar
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Usar métricas da validação cruzada se disponível, senão usar métricas globais
        if 'metricas_cv_media' in metricas_dict:
            # Usar média dos folds (MAIS CORRETO)
            metricas_principais = {
                'Acurácia': metricas_dict['metricas_cv_media']['acuracia'],
                'Precisão': metricas_dict['metricas_cv_media']['precisao'],
                'Recall': metricas_dict['metricas_cv_media']['recall'],
                'F1-Score': metricas_dict['metricas_cv_media']['f1_score']
            }
        else:
            # Fallback para métricas antigas
            metricas_principais = {
                'Acurácia': metricas_dict['metricas_globais']['acuracia'],
                'Precisão': metricas_dict['metricas_por_classe']['weighted avg']['precision'],
                'Recall': metricas_dict['metricas_por_classe']['weighted avg']['recall'],
                'F1-Score': metricas_dict['metricas_por_classe']['weighted avg']['f1-score']
            }
        
        cores = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        barras = ax.bar(metricas_principais.keys(), metricas_principais.values(), color=cores)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Métricas de Desempenho do Modelo', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for barra in barras:
            altura = barra.get_height()
            ax.text(barra.get_x() + barra.get_width()/2., altura,
                   f'{altura:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if salvar:
            caminho = self.dir_saida / 'metricas_desempenho.png'
            plt.savefig(caminho, dpi=300, bbox_inches='tight')
            logger.success(f"Gráfico de métricas salvo: {caminho}")
        
        plt.close()
