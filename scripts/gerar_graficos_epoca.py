"""
Gera graficos de acuracia e loss por epoca - treino vs validacao
Estilo elegante similar aos graficos anteriores
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Configuracao
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = '#E8E8F0'
plt.rcParams['axes.facecolor'] = '#E8E8F0'

# Caminhos
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("="*80)
print("GERANDO GRAFICOS DE ACURACIA E LOSS POR EPOCA")
print("="*80)

# Carregar dados
print("\n[1/3] Carregando dataset...")
df = pd.read_csv(BASE_DIR / "data" / "lgpd_rotulado_global.csv")
df = df[['texto', 'categoria']].dropna().reset_index(drop=True)
print(f"Dataset: {len(df)} sentencas, {df['categoria'].nunique()} categorias")

X = df['texto'].values
y = df['categoria'].values

# Treinar por multiplas epocas
print("\n[2/3] Treinando modelo (49 epocas, 5 folds)...")
n_epochs = 49
n_folds = 5

historico = {
    'epoch': [],
    'train_acc': [],
    'val_acc': [],
    'train_loss': [],
    'val_loss': []
}

for epoch in range(1, n_epochs + 1):
    epoch_train_acc = []
    epoch_val_acc = []
    epoch_train_loss = []
    epoch_val_loss = []
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=epoch)
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)
        
        model = MultinomialNB(alpha=0.5)
        model.fit(X_train_vec, y_train)
        
        y_train_pred = model.predict(X_train_vec)
        y_val_pred = model.predict(X_val_vec)
        
        # Calcular loss
        y_train_proba = model.predict_proba(X_train_vec)
        y_val_proba = model.predict_proba(X_val_vec)
        
        class_to_idx = {c: i for i, c in enumerate(model.classes_)}
        train_indices = [class_to_idx[c] for c in y_train]
        val_indices = [class_to_idx[c] for c in y_val]
        
        train_loss = -np.mean([np.log(y_train_proba[i, train_indices[i]] + 1e-10) 
                               for i in range(len(y_train))])
        val_loss = -np.mean([np.log(y_val_proba[i, val_indices[i]] + 1e-10) 
                             for i in range(len(y_val))])
        
        epoch_train_acc.append(accuracy_score(y_train, y_train_pred))
        epoch_val_acc.append(accuracy_score(y_val, y_val_pred))
        epoch_train_loss.append(train_loss)
        epoch_val_loss.append(val_loss)
    
    historico['epoch'].append(epoch)
    historico['train_acc'].append(np.mean(epoch_train_acc))
    historico['val_acc'].append(np.mean(epoch_val_acc))
    historico['train_loss'].append(np.mean(epoch_train_loss))
    historico['val_loss'].append(np.mean(epoch_val_loss))
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoca {epoch}: Train Acc={historico['train_acc'][-1]:.4f}, "
              f"Val Acc={historico['val_acc'][-1]:.4f}")

print(f"\nAcuracia final - Treino: {historico['train_acc'][-1]:.4f}, "
      f"Validacao: {historico['val_acc'][-1]:.4f}")

# ========== GRAFICO 1: ACURACIA POR EPOCA ==========
print("\n[3/3] Gerando graficos...")
print("  - Acuracia por epoca...")

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_facecolor('#E8E8F0')
fig.patch.set_facecolor('#E8E8F0')

epochs = historico['epoch']

# Linhas
ax.plot(epochs, historico['train_acc'], 
        'o-',
        color='#3498db',
        linewidth=2.5,
        markersize=4,
        markerfacecolor='#3498db',
        markeredgewidth=1.5,
        markeredgecolor='white',
        label='Treino',
        zorder=3)

ax.plot(epochs, historico['val_acc'],
        'o-',
        color='#e74c3c',
        linewidth=2.5,
        markersize=4,
        markerfacecolor='#e74c3c',
        markeredgewidth=1.5,
        markeredgecolor='white',
        label='Validacao',
        zorder=3)

# Anotacao no ponto final
final_train = historico['train_acc'][-1]
final_val = historico['val_acc'][-1]

ax.annotate(f'{final_train:.4f}',
            xy=(epochs[-1], final_train),
            xytext=(10, 0),
            textcoords='offset points',
            ha='left',
            fontsize=10,
            fontweight='bold',
            color='#3498db',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax.annotate(f'{final_val:.4f}',
            xy=(epochs[-1], final_val),
            xytext=(10, -5),
            textcoords='offset points',
            ha='left',
            fontsize=10,
            fontweight='bold',
            color='#e74c3c',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Configuracoes
ax.set_xlabel('Epoca', fontsize=12, fontweight='bold', color='#2C3E50')
ax.set_ylabel('Acuracia', fontsize=12, fontweight='bold', color='#2C3E50')
ax.set_title('Evolucao da Acuracia ao Longo das Epocas',
             fontsize=14, fontweight='bold', color='#2C3E50', pad=20)

ax.set_xlim(0, n_epochs + 1)
ax.set_ylim(0.65, 1.0)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='white')

ax.legend(loc='lower right',
          fontsize=11,
          frameon=True,
          fancybox=True,
          shadow=True,
          framealpha=0.9)

plt.tight_layout()
output_path = RESULTS_DIR / "acuracia_por_epoca.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#E8E8F0')
plt.close()
print(f"    Salvo: {output_path}")

# ========== GRAFICO 2: LOSS POR EPOCA ==========
print("  - Loss por epoca...")

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_facecolor('#E8E8F0')
fig.patch.set_facecolor('#E8E8F0')

# Linha de treino (azul)
ax.plot(epochs, historico['train_loss'],
        'o-',
        color='#4A8CB5',
        linewidth=2.5,
        markersize=7,
        markerfacecolor='#4A8CB5',
        markeredgewidth=2,
        markeredgecolor='white',
        label='Treino',
        zorder=3,
        alpha=0.9)

# Linha de validacao (vermelho)
ax.plot(epochs, historico['val_loss'],
        'o-',
        color='#E85D75',
        linewidth=2.5,
        markersize=7,
        markerfacecolor='#E85D75',
        markeredgewidth=2,
        markeredgecolor='white',
        label='Validacao',
        zorder=3)

# Configuracoes dos eixos
ax.set_xlabel('Epoca', fontsize=13, fontweight='bold', color='#34495E', labelpad=10)
ax.set_ylabel('Loss (Cross-Entropy)', fontsize=13, fontweight='bold', color='#34495E', labelpad=10)
ax.set_title('Evolucao da Funcao de Perda ao Longo das Epocas',
             fontsize=15, fontweight='bold', color='#2C3E50', pad=20)

# Limites e grade
ax.set_xlim(0, n_epochs + 1)
ax.set_ylim(0.6, 1.4)
ax.grid(True, alpha=0.25, linestyle='-', linewidth=1, color='gray', zorder=0)

# Legenda
ax.legend(loc='upper right',
          fontsize=11,
          frameon=True,
          fancybox=False,
          edgecolor='#95a5a6',
          facecolor='white',
          framealpha=0.95,
          shadow=False)

# Valores finais no canto
final_text = (f'Epoca {n_epochs}:\n'
              f'Treino: {historico["train_loss"][-1]:.4f}\n'
              f'Validacao: {historico["val_loss"][-1]:.4f}')
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
output_path = RESULTS_DIR / "loss_por_epoca.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#E8E8F0')
plt.close()
print(f"    Salvo: {output_path}")

print("\n" + "="*80)
print("PROCESSO CONCLUIDO!")
print("="*80)
print(f"\nArquivos salvos em: {RESULTS_DIR.absolute()}")
print("  1. acuracia_por_epoca.png")
print("  2. loss_por_epoca.png")
print("\nMetricas finais:")
print(f"  Acuracia (Treino):    {historico['train_acc'][-1]:.4f}")
print(f"  Acuracia (Validacao): {historico['val_acc'][-1]:.4f}")
print(f"  Loss (Treino):        {historico['train_loss'][-1]:.4f}")
print(f"  Loss (Validacao):     {historico['val_loss'][-1]:.4f}")
