"""
================================================================================
GRAFICOS DE TREINAMENTO PARA CLASSIFICADOR NAIVE BAYES
================================================================================
Gera graficos tecnicamente corretos para modelos que nao usam epocas:
1. Curva de Aprendizado (learning curve) - mostra como performance melhora com mais dados
2. Matriz de Confusao - erros por categoria
3. Metricas por Categoria - precision/recall/F1 detalhado
================================================================================
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (confusion_matrix, classification_report, 
                            accuracy_score, precision_recall_fscore_support)

# ========== CONFIGURACAO ==========
WORKSPACE = Path(__file__).parent.parent
DATA_PATH = WORKSPACE / "data" / "lgpd_rotulado_global.csv"
RESULTS_DIR = WORKSPACE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Configuracoes do modelo
RANDOM_STATE = 42
N_FOLDS = 5
ALPHA = 0.5  # Suavizacao Laplace

# Estilos visuais
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*80)
print("GRAFICOS DE TREINAMENTO - CLASSIFICADOR NAIVE BAYES")
print("="*80)

# ========== CARREGAR DADOS ==========
print("\n[1/4] Carregando dataset...")
df = pd.read_csv(DATA_PATH)
df = df[['texto', 'categoria']].dropna()

X = df['texto'].values
y = df['categoria'].values
n_samples = len(X)
n_classes = len(np.unique(y))

print(f"Dataset: {n_samples} sentencas, {n_classes} categorias")

# ========== GRAFICO 1: CURVA DE APRENDIZADO ==========
print("\n[2/4] Gerando Curva de Aprendizado...")

# Vetorizador
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    min_df=1,
    sublinear_tf=True
)

# Modelo
modelo = MultinomialNB(alpha=ALPHA)

# Tamanhos de treino a testar (de 20% a 100% dos dados)
train_sizes = np.linspace(0.2, 1.0, 10)

# Calcular learning curve
train_sizes_abs, train_scores, val_scores = learning_curve(
    modelo,
    vectorizer.fit_transform(X),
    y,
    train_sizes=train_sizes,
    cv=N_FOLDS,
    scoring='accuracy',
    n_jobs=-1,
    random_state=RANDOM_STATE
)

# Calcular medias e desvios
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

# Plotar
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_facecolor('#E8E8F0')
fig.patch.set_facecolor('#E8E8F0')

# Banda de desvio - treino
ax.fill_between(train_sizes_abs,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.2,
                color='#4A8CB5',
                zorder=1)

# Banda de desvio - validacao
ax.fill_between(train_sizes_abs,
                val_mean - val_std,
                val_mean + val_std,
                alpha=0.2,
                color='#E85D75',
                zorder=1)

# Linhas principais
ax.plot(train_sizes_abs, train_mean,
        'o-',
        color='#4A8CB5',
        linewidth=2.5,
        markersize=8,
        markerfacecolor='#4A8CB5',
        markeredgewidth=2,
        markeredgecolor='white',
        label='Treino',
        zorder=3)

ax.plot(train_sizes_abs, val_mean,
        'o-',
        color='#E85D75',
        linewidth=2.5,
        markersize=8,
        markerfacecolor='#E85D75',
        markeredgewidth=2,
        markeredgecolor='white',
        label='Validacao',
        zorder=3)

# Configuracoes
ax.set_xlabel('Numero de Exemplos de Treino', fontsize=13, fontweight='bold', 
              color='#34495E', labelpad=10)
ax.set_ylabel('Acuracia', fontsize=13, fontweight='bold', 
              color='#34495E', labelpad=10)
ax.set_title('Curva de Aprendizado: Desempenho vs. Tamanho do Dataset',
             fontsize=15, fontweight='bold', color='#2C3E50', pad=20)

ax.set_ylim(0.5, 1.05)
ax.grid(True, alpha=0.25, linestyle='-', linewidth=1, color='gray', zorder=0)

# Legenda
ax.legend(loc='lower right',
          fontsize=12,
          frameon=True,
          fancybox=False,
          edgecolor='#95a5a6',
          facecolor='white',
          framealpha=0.95)

# Anotacao final
final_text = (f'Com {n_samples} exemplos:\n'
              f'Treino: {train_mean[-1]:.2%}\n'
              f'Validacao: {val_mean[-1]:.2%}')
ax.text(0.02, 0.98, final_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.6',
                 facecolor='white',
                 edgecolor='#95a5a6',
                 linewidth=1.5,
                 alpha=0.95))

plt.tight_layout()
path_learning = RESULTS_DIR / "curva_aprendizado.png"
plt.savefig(path_learning, dpi=300, bbox_inches='tight', facecolor='#E8E8F0')
plt.close()
print(f"  Salvo: {path_learning}")

# ========== GRAFICO 2: MATRIZ DE CONFUSAO ==========
print("\n[3/4] Gerando Matriz de Confusao...")

# Treinar modelo completo para matriz de confusao
X_vec = vectorizer.fit_transform(X)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

y_true_all = []
y_pred_all = []

for train_idx, val_idx in skf.split(X, y):
    X_train = X_vec[train_idx]
    X_val = X_vec[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    
    modelo_fold = MultinomialNB(alpha=ALPHA)
    modelo_fold.fit(X_train, y_train)
    y_pred = modelo_fold.predict(X_val)
    
    y_true_all.extend(y_val)
    y_pred_all.extend(y_pred)

# Calcular matriz
classes = sorted(np.unique(y))
cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Plotar
fig, ax = plt.subplots(figsize=(14, 12))
ax.set_facecolor('#E8E8F0')
fig.patch.set_facecolor('#E8E8F0')

sns.heatmap(cm_norm,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes,
            cbar_kws={'label': 'Percentual (%)'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax)

ax.set_xlabel('Categoria Predita', fontsize=13, fontweight='bold', 
              color='#34495E', labelpad=10)
ax.set_ylabel('Categoria Real', fontsize=13, fontweight='bold', 
              color='#34495E', labelpad=10)
ax.set_title('Matriz de Confusao - Validacao Cruzada (5 Folds)',
             fontsize=15, fontweight='bold', color='#2C3E50', pad=20)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

path_confusion = RESULTS_DIR / "matriz_confusao_correto.png"
plt.savefig(path_confusion, dpi=300, bbox_inches='tight', facecolor='#E8E8F0')
plt.close()
print(f"  Salvo: {path_confusion}")

# ========== GRAFICO 3: METRICAS POR CATEGORIA ==========
print("\n[4/4] Gerando Metricas por Categoria...")

# Calcular metricas detalhadas
precision, recall, f1, support = precision_recall_fscore_support(
    y_true_all, y_pred_all, labels=classes, zero_division=0
)

# Criar DataFrame
metricas_df = pd.DataFrame({
    'Categoria': classes,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Suporte': support
}).sort_values('F1-Score', ascending=True)

# Plotar
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_facecolor('#E8E8F0')
fig.patch.set_facecolor('#E8E8F0')

x_pos = np.arange(len(metricas_df))
width = 0.25

bars1 = ax.barh(x_pos - width, metricas_df['Precision'], width,
                label='Precision', color='#4A8CB5', edgecolor='white', linewidth=1.5)
bars2 = ax.barh(x_pos, metricas_df['Recall'], width,
                label='Recall', color='#E85D75', edgecolor='white', linewidth=1.5)
bars3 = ax.barh(x_pos + width, metricas_df['F1-Score'], width,
                label='F1-Score', color='#52C084', edgecolor='white', linewidth=1.5)

# Adicionar valores nas barras
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        width_val = bar.get_width()
        if width_val > 0:
            ax.text(width_val + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{width_val:.2f}',
                   ha='left', va='center', fontsize=9, fontweight='bold')

ax.set_yticks(x_pos)
ax.set_yticklabels(metricas_df['Categoria'])
ax.set_xlabel('Score', fontsize=13, fontweight='bold', color='#34495E', labelpad=10)
ax.set_ylabel('Categoria LGPD', fontsize=13, fontweight='bold', 
              color='#34495E', labelpad=10)
ax.set_title('Metricas de Classificacao por Categoria',
             fontsize=15, fontweight='bold', color='#2C3E50', pad=20)

ax.set_xlim(0, 1.15)
ax.grid(True, axis='x', alpha=0.25, linestyle='-', linewidth=1, color='gray', zorder=0)

ax.legend(loc='lower right',
          fontsize=11,
          frameon=True,
          fancybox=False,
          edgecolor='#95a5a6',
          facecolor='white',
          framealpha=0.95)

plt.tight_layout()
path_metricas = RESULTS_DIR / "metricas_por_categoria.png"
plt.savefig(path_metricas, dpi=300, bbox_inches='tight', facecolor='#E8E8F0')
plt.close()
print(f"  Salvo: {path_metricas}")

# ========== RELATORIO FINAL ==========
print("\n" + "="*80)
print("PROCESSO CONCLUIDO!")
print("="*80)
print(f"\nArquivos salvos em: {RESULTS_DIR.absolute()}")
print("  1. curva_aprendizado.png - Como performance melhora com mais dados")
print("  2. matriz_confusao_correto.png - Erros de classificacao por categoria")
print("  3. metricas_por_categoria.png - Precision/Recall/F1 detalhado")

# Metricas finais
acc_final = accuracy_score(y_true_all, y_pred_all)
print(f"\nMetricas Finais (Cross-Validation):")
print(f"  Acuracia Global: {acc_final:.2%}")
print(f"  Precision Media: {precision.mean():.2%}")
print(f"  Recall Medio:    {recall.mean():.2%}")
print(f"  F1-Score Medio:  {f1.mean():.2%}")
