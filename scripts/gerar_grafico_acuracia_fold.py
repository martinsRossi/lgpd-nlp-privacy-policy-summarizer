"""
Gera grafico de acuracia por fold - estilo elegante similar ao exemplo fornecido
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

print("Gerando grafico de acuracia por fold...")

# Carregar dados
df = pd.read_csv(BASE_DIR / "data" / "lgpd_rotulado_global.csv")
df = df[['texto', 'categoria']].dropna().reset_index(drop=True)
print(f"Dataset: {len(df)} sentencas")

X = df['texto'].values
y = df['categoria'].values

# Treinar com 5 folds e coletar acuracias
n_folds = 5
acuracias_por_fold = []

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

for fold_num, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    model = MultinomialNB(alpha=0.5)
    model.fit(X_train_vec, y_train)
    
    y_val_pred = model.predict(X_val_vec)
    acc = accuracy_score(y_val, y_val_pred) * 100  # Converter para porcentagem
    
    acuracias_por_fold.append(acc)
    print(f"Fold {fold_num}: {acc:.1f}%")

# Calcular estatisticas
media = np.mean(acuracias_por_fold)
desvio = np.std(acuracias_por_fold)

print(f"\nMedia: {media:.1f}%")
print(f"Desvio Padrao: {desvio:.1f}%")

# Criar grafico
fig, ax = plt.subplots(figsize=(10, 6))

# Configurar fundo
ax.set_facecolor('#E8E8F0')
fig.patch.set_facecolor('#E8E8F0')

# Dados
folds = list(range(1, n_folds + 1))

# Area de desvio padrao (faixa rosa)
ax.fill_between(folds, 
                media - desvio, 
                media + desvio,
                color='#D4A5C7', 
                alpha=0.35,
                label=f'Desvio Padrao: ±{desvio:.1f}%')

# Linha de media (tracejada)
ax.axhline(y=media, 
           color='#8B5A8B', 
           linestyle='--', 
           linewidth=2,
           label=f'Media: {media:.1f}%',
           zorder=2)

# Linha de acuracia por fold
ax.plot(folds, acuracias_por_fold, 
        'o-', 
        color='#4A8CB5',
        linewidth=2.5, 
        markersize=10,
        markerfacecolor='#4A8CB5',
        markeredgewidth=2,
        markeredgecolor='white',
        label='Acuracia por Fold',
        zorder=3)

# Anotacoes nos pontos
for fold, acc in zip(folds, acuracias_por_fold):
    ax.annotate(f'{acc:.1f}%', 
                xy=(fold, acc), 
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=10,
                fontweight='bold',
                color='#2C3E50')

# Configuracoes dos eixos
ax.set_xlabel('Fold (Cross-Validation 5-Fold)', 
              fontsize=12, 
              fontweight='bold',
              color='#2C3E50')
ax.set_ylabel('Acuracia (%)', 
              fontsize=12, 
              fontweight='bold',
              color='#2C3E50')
ax.set_title('Validacao Cruzada Estratificada - Acuracia por Fold', 
             fontsize=14, 
             fontweight='bold',
             color='#2C3E50',
             pad=20)

# Configurar eixos
ax.set_xticks(folds)
ax.set_xlim(0.5, n_folds + 0.5)
ax.set_ylim(60, 90)

# Grid sutil
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='white')

# Legenda
ax.legend(loc='lower right', 
          fontsize=10,
          frameon=True,
          fancybox=True,
          shadow=True,
          framealpha=0.9)

# Ajustar layout
plt.tight_layout()

# Salvar
output_path = RESULTS_DIR / "acuracia_por_fold.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#E8E8F0')
plt.close()

print(f"\nGrafico salvo em: {output_path}")
print("\nConcluido!")
