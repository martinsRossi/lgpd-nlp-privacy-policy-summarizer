"""
Script para gerar gráficos de análise do treinamento do classificador LGPD.

Gera três tipos de visualizações:
1. Gráficos de convergência (acurácia e loss ao longo das épocas)
2. Matriz de confusão do conjunto de testes
3. Métricas de desempenho comparativas (acurácia, precisão, recall, F1-score)

Autor: Sistema TCC LGPD
Data: Dezembro 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from datetime import datetime
import sys

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuração de estilo dos gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Caminhos
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Criar diretório de resultados se não existir
RESULTS_DIR.mkdir(exist_ok=True)

def carregar_dataset():
    """Carrega o dataset de treinamento."""
    dataset_path = DATA_DIR / "lgpd_rotulado_global.csv"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado em {dataset_path}")
    
    df = pd.read_csv(dataset_path, encoding='utf-8', on_bad_lines='skip')
    
    # Filtrar apenas colunas necessárias e remover linhas com valores ausentes
    df = df[['texto', 'categoria']].dropna()
    
    print(f"OK Dataset carregado: {len(df)} sentenças")
    print(f"Categorias: {df['categoria'].nunique()}")
    print(f"Distribuição:")
    for cat, count in df['categoria'].value_counts().items():
        print(f"   {cat}: {count}")
    
    return df

def treinar_com_metricas(df, n_epochs=49, n_folds=5):
    """Treina o modelo por múltiplas épocas coletando métricas."""
    print(f"\nIniciando treinamento: {n_epochs} épocas, {n_folds} folds")
    
    # Reset index para garantir indexação correta
    df = df.reset_index(drop=True)
    
    X = df['texto'].values
    y = df['categoria'].values
    
    # Ajustar número de folds se necessário
    min_class_size = df['categoria'].value_counts().min()
    if min_class_size < n_folds:
        n_folds = max(2, min_class_size)
        print(f"[WARN] Ajustando para {n_folds}")
    
    # Histórico de métricas
    historico = {
        'epoch': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_loss': [],
        'val_loss': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': [],
        'train_f1': [],
        'val_f1': []
    }
    
    y_true_final = []
    y_pred_final = []
    
    # Treinamento por épocas
    for epoch in range(1, n_epochs + 1):
        epoch_train_acc = []
        epoch_val_acc = []
        epoch_train_loss = []
        epoch_val_loss = []
        epoch_train_prec = []
        epoch_val_prec = []
        epoch_train_rec = []
        epoch_val_rec = []
        epoch_train_f1 = []
        epoch_val_f1 = []
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=epoch)
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train = X[train_idx]
            X_val = X[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]
            
            vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.95
            )
            
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_val_tfidf = vectorizer.transform(X_val)
            
            model = MultinomialNB(alpha=0.5)
            model.fit(X_train_tfidf, y_train)
            
            y_train_pred = model.predict(X_train_tfidf)
            y_val_pred = model.predict(X_val)
            
            y_train_proba = model.predict_proba(X_train_tfidf)
            y_val_proba = model.predict_proba(X_val)
            
            train_loss = -np.mean(np.log(y_train_proba[np.arange(len(y_train)), 
                                          model.classes_.searchsorted(y_train)] + 1e-10))
            val_loss = -np.mean(np.log(y_val_proba[np.arange(len(y_val)), 
                                        model.classes_.searchsorted(y_val)] + 1e-10))
            
            epoch_train_acc.append(accuracy_score(y_train, y_train_pred))
            epoch_val_acc.append(accuracy_score(y_val, y_val_pred))
            epoch_train_loss.append(train_loss)
            epoch_val_loss.append(val_loss)
            epoch_train_prec.append(precision_score(y_train, y_train_pred, average='weighted', zero_division=0))
            epoch_val_prec.append(precision_score(y_val, y_val_pred, average='weighted', zero_division=0))
            epoch_train_rec.append(recall_score(y_train, y_train_pred, average='weighted', zero_division=0))
            epoch_val_rec.append(recall_score(y_val, y_val_pred, average='weighted', zero_division=0))
            epoch_train_f1.append(f1_score(y_train, y_train_pred, average='weighted', zero_division=0))
            epoch_val_f1.append(f1_score(y_val, y_val_pred, average='weighted', zero_division=0))
            
            if epoch == n_epochs:
                y_true_final.extend(y_val)
                y_pred_final.extend(y_val_pred)
        
        historico['epoch'].append(epoch)
        historico['train_accuracy'].append(np.mean(epoch_train_acc))
        historico['val_accuracy'].append(np.mean(epoch_val_acc))
        historico['train_loss'].append(np.mean(epoch_train_loss))
        historico['val_loss'].append(np.mean(epoch_val_loss))
        historico['train_precision'].append(np.mean(epoch_train_prec))
        historico['val_precision'].append(np.mean(epoch_val_prec))
        historico['train_recall'].append(np.mean(epoch_train_rec))
        historico['val_recall'].append(np.mean(epoch_val_rec))
        historico['train_f1'].append(np.mean(epoch_train_f1))
        historico['val_f1'].append(np.mean(epoch_val_f1))
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Época {epoch}/{n_epochs} - Val Acc: {historico['val_accuracy'][-1]:.4f}")
    
    historico['y_true_final'] = np.array(y_true_final)
    historico['y_pred_final'] = np.array(y_pred_final)
    historico['classes'] = sorted(df['categoria'].unique())
    
    print(f"\nOK Treinamento concluído!")
    print(f"Acurácia final: {historico['val_accuracy'][-1]:.4f}")
    
    return historico

def gerar_grafico_convergencia(historico):
    """Gera gráficos de convergência (acurácia e loss)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs = historico['epoch']
    
    # Gráfico 1: Acurácia
    axes[0].plot(epochs, historico['train_accuracy'], 'b-', linewidth=2, label='Treino')
    axes[0].plot(epochs, historico['val_accuracy'], 'orange', linewidth=2, label='Validação')
    axes[0].set_xlabel('Época', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Acurácia', fontsize=12, fontweight='bold')
    axes[0].set_title('Evolução da Acurácia ao Longo das Épocas', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right', frameon=True, shadow=True)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([1, max(epochs)])
    
    final_val_acc = historico['val_accuracy'][-1]
    axes[0].annotate(f'Final: {final_val_acc:.4f}',
                     xy=(epochs[-1], final_val_acc),
                     xytext=(epochs[-1] - 10, final_val_acc - 0.05),
                     fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Gráfico 2: Loss
    axes[1].plot(epochs, historico['train_loss'], 'b-', linewidth=2, label='Treino')
    axes[1].plot(epochs, historico['val_loss'], 'orange', linewidth=2, label='Validação')
    axes[1].set_xlabel('Época', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Loss (Cross-Entropy)', fontsize=12, fontweight='bold')
    axes[1].set_title('Evolução da Função de Perda ao Longo das Épocas', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', frameon=True, shadow=True)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([1, max(epochs)])
    
    final_val_loss = historico['val_loss'][-1]
    axes[1].annotate(f'Final: {final_val_loss:.4f}',
                     xy=(epochs[-1], final_val_loss),
                     xytext=(epochs[-1] - 10, final_val_loss + 0.05),
                     fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    output_path = RESULTS_DIR / "convergencia_treinamento.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"OK Gráfico de convergência salvo em: {output_path}")
    plt.close()

def gerar_matriz_confusao(historico):
    """Gera matriz de confusão do conjunto de testes."""
    y_true = historico['y_true_final']
    y_pred = historico['y_pred_final']
    classes = historico['classes']
    
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Porcentagem (%)'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Categoria Predita', fontsize=12, fontweight='bold')
    ax.set_ylabel('Categoria Real', fontsize=12, fontweight='bold')
    ax.set_title('Matriz de Confusão - Classificador LGPD (Conjunto de Validação)',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    output_path = RESULTS_DIR / "matriz_confusao.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"OK Matriz de confusão salva em: {output_path}")
    plt.close()

def gerar_metricas_desempenho(historico):
    """Gera gráfico comparativo de métricas de desempenho."""
    metricas_train = {
        'Acurácia': historico['train_accuracy'][-1],
        'Precisão': historico['train_precision'][-1],
        'Recall': historico['train_recall'][-1],
        'F1-Score': historico['train_f1'][-1]
    }
    
    metricas_val = {
        'Acurácia': historico['val_accuracy'][-1],
        'Precisão': historico['val_precision'][-1],
        'Recall': historico['val_recall'][-1],
        'F1-Score': historico['val_f1'][-1]
    }
    
    categorias = list(metricas_train.keys())
    valores_train = list(metricas_train.values())
    valores_val = list(metricas_val.values())
    
    x = np.arange(len(categorias))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width/2, valores_train, width, label='Treino',
                   color='#3498db', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, valores_val, width, label='Validação',
                   color='#e74c3c', edgecolor='black', linewidth=1.2)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Métrica', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor', fontsize=12, fontweight='bold')
    ax.set_title('Comparação de Métricas de Desempenho: Treino vs Validação',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categorias, fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    output_path = RESULTS_DIR / "metricas_desempenho.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"OK Gráfico de métricas salvo em: {output_path}")
    plt.close()

def gerar_relatorio_metricas(historico):
    """Gera relatório textual com métricas detalhadas."""
    y_true = historico['y_true_final']
    y_pred = historico['y_pred_final']
    classes = historico['classes']
    
    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    
    output_path = RESULTS_DIR / "relatorio_metricas.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RELATÓRIO DE MÉTRICAS - CLASSIFICADOR LGPD\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Data de geração: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Total de épocas: {len(historico['epoch'])}\n")
        f.write(f"Total de exemplos avaliados: {len(y_true)}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("MÉTRICAS FINAIS (ÚLTIMA ÉPOCA)\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"Acurácia (Treino):     {historico['train_accuracy'][-1]:.4f}\n")
        f.write(f"Acurácia (Validação):  {historico['val_accuracy'][-1]:.4f}\n\n")
        
        f.write(f"F1-Score (Treino):     {historico['train_f1'][-1]:.4f}\n")
        f.write(f"F1-Score (Validação):  {historico['val_f1'][-1]:.4f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("RELATÓRIO DETALHADO POR CATEGORIA\n")
        f.write("-" * 80 + "\n\n")
        f.write(report)
    
    print(f"OK Relatório de métricas salvo em: {output_path}")

def main():
    """Função principal."""
    print("=" * 80)
    print("GERAÇÃO DE GRÁFICOS DE TREINAMENTO - CLASSIFICADOR LGPD")
    print("=" * 80)
    print()
    
    try:
        df = carregar_dataset()
        
        historico = treinar_com_metricas(df, n_epochs=49, n_folds=5)
        
        print("\n" + "=" * 80)
        print("GERANDO GRÁFICOS")
        print("=" * 80 + "\n")
        
        gerar_grafico_convergencia(historico)
        gerar_matriz_confusao(historico)
        gerar_metricas_desempenho(historico)
        gerar_relatorio_metricas(historico)
        
        print("\n" + "=" * 80)
        print("OK PROCESSO CONCLUÍDO COM SUCESSO!")
        print("=" * 80)
        print(f"\nTodos os arquivos foram salvos em: {RESULTS_DIR.absolute()}")
        print("\nArquivos gerados:")
        print("  • convergencia_treinamento.png")
        print("  • matriz_confusao.png")
        print("  • metricas_desempenho.png")
        print("  • relatorio_metricas.txt")
        
    except Exception as e:
        print(f"\n[ERROR] Erro durante execução: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
