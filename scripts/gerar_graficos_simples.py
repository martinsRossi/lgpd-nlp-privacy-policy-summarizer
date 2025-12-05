"""
Script simplificado para gerar graficos de treinamento LGPD.
Versao sem emojis para compatibilidade com Windows.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sem GUI
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

# Configuracao
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Caminhos
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("="*80)
print("GERACAO DE GRAFICOS - CLASSIFICADOR LGPD")
print("="*80)

# Carregar dados
print("\n[1/4] Carregando dataset...")
df = pd.read_csv(BASE_DIR / "data" / "lgpd_rotulado_global.csv")
df = df[['texto', 'categoria']].dropna().reset_index(drop=True)
print(f"Dataset: {len(df)} sentencas, {df['categoria'].nunique()} categorias")

# Preparar dados
X = df['texto'].values
y = df['categoria'].values
classes = sorted(df['categoria'].unique())

# Treinar
print("\n[2/4] Treinando modelo (49 epocas, 5 folds)...")
n_epochs = 49
n_folds = 5

historico = {
    'epoch': [],
    'train_acc': [],
    'val_acc': [],
    'train_loss': [],
    'val_loss': []
}

y_true_all = []
y_pred_all = []

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
        
        # Calcular loss manualmente
        y_train_proba = model.predict_proba(X_train_vec)
        y_val_proba = model.predict_proba(X_val_vec)
        
        # Mapear classes para indices
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
        
        if epoch == n_epochs:
            y_true_all.extend(y_val)
            y_pred_all.extend(y_val_pred)
    
    historico['epoch'].append(epoch)
    historico['train_acc'].append(np.mean(epoch_train_acc))
    historico['val_acc'].append(np.mean(epoch_val_acc))
    historico['train_loss'].append(np.mean(epoch_train_loss))
    historico['val_loss'].append(np.mean(epoch_val_loss))
    
    if epoch % 10 == 0:
        print(f"  Epoca {epoch}: Val Acc = {historico['val_acc'][-1]:.4f}")

print(f"\nAcuracia final: {historico['val_acc'][-1]:.4f}")

# Grafico 1: Convergencia
print("\n[3/4] Gerando grafico de convergencia...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

epochs = historico['epoch']

# Acuracia
axes[0].plot(epochs, historico['train_acc'], 'b-', linewidth=2, label='Treino')
axes[0].plot(epochs, historico['val_acc'], 'orange', linewidth=2, label='Validacao')
axes[0].set_xlabel('Epoca', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Acuracia', fontsize=12, fontweight='bold')
axes[0].set_title('Evolucao da Acuracia ao Longo das Epocas', fontsize=14, fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(epochs, historico['train_loss'], 'b-', linewidth=2, label='Treino')
axes[1].plot(epochs, historico['val_loss'], 'orange', linewidth=2, label='Validacao')
axes[1].set_xlabel('Epoca', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Loss (Cross-Entropy)', fontsize=12, fontweight='bold')
axes[1].set_title('Evolucao da Funcao de Perda ao Longo das Epocas', fontsize=14, fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "convergencia_treinamento.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Salvo: {RESULTS_DIR / 'convergencia_treinamento.png'}")

# Grafico 2: Matriz de confusao
print("\n[4/4] Gerando matriz de confusao e metricas...")
cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=classes, yticklabels=classes,
            cbar_kws={'label': 'Porcentagem (%)'}, ax=ax,
            linewidths=0.5, linecolor='gray')

ax.set_xlabel('Categoria Predita', fontsize=12, fontweight='bold')
ax.set_ylabel('Categoria Real', fontsize=12, fontweight='bold')
ax.set_title('Matriz de Confusao - Classificador LGPD', fontsize=14, fontweight='bold', pad=20)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "matriz_confusao.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Salvo: {RESULTS_DIR / 'matriz_confusao.png'}")

# Grafico 3: Metricas
metricas_train = {
    'Acuracia': historico['train_acc'][-1],
    'Precisao': precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0),
    'Recall': recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0),
    'F1-Score': f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
}

metricas_val = {
    'Acuracia': historico['val_acc'][-1],
    'Precisao': precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0),
    'Recall': recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0),
    'F1-Score': f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
}

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(metricas_train))
width = 0.35

bars1 = ax.bar(x - width/2, list(metricas_train.values()), width, label='Treino', color='#3498db')
bars2 = ax.bar(x + width/2, list(metricas_val.values()), width, label='Validacao', color='#e74c3c')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Metrica', fontsize=12, fontweight='bold')
ax.set_ylabel('Valor', fontsize=12, fontweight='bold')
ax.set_title('Comparacao de Metricas: Treino vs Validacao', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(list(metricas_train.keys()), fontsize=11, fontweight='bold')
ax.legend(loc='lower right')
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "metricas_desempenho.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Salvo: {RESULTS_DIR / 'metricas_desempenho.png'}")

# Relatorio
with open(RESULTS_DIR / "relatorio_metricas.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("RELATORIO DE METRICAS - CLASSIFICADOR LGPD\n")
    f.write("="*80 + "\n\n")
    f.write(f"Acuracia Final (Validacao): {historico['val_acc'][-1]:.4f}\n")
    f.write(f"Loss Final (Validacao): {historico['val_loss'][-1]:.4f}\n\n")
    f.write(classification_report(y_true_all, y_pred_all, target_names=classes, zero_division=0))

print(f"Salvo: {RESULTS_DIR / 'relatorio_metricas.txt'}")

print("\n" + "="*80)
print("PROCESSO CONCLUIDO COM SUCESSO!")
print("="*80)
print(f"\nArquivos salvos em: {RESULTS_DIR.absolute()}")
print("  - convergencia_treinamento.png")
print("  - matriz_confusao.png")
print("  - metricas_desempenho.png")
print("  - relatorio_metricas.txt")
