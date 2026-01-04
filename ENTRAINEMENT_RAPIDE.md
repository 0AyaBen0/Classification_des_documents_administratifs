# Entraînement Rapide sur CPU (< 2 heures)

## 🚀 Commande Optimisée pour CPU

### Option 1 : Modèle Léger (Recommandé - ~1-1.5h)

```bash
python train.py --data-dir data/images --epochs 20 --batch-size 16 --learning-rate 0.001 --device cpu --light
```

**Paramètres :**
- `--light` : Modèle MobileNet (beaucoup plus rapide)
- `--epochs 20` : Suffisant avec learning rate plus élevé
- `--batch-size 16` : Optimal pour CPU (pas trop de mémoire)
- `--learning-rate 0.001` : 10x plus rapide pour converger
- `--device cpu` : Forcer CPU

**Temps estimé : 1-1.5 heures**

### Option 2 : Modèle Standard Optimisé (~1.5-2h)

```bash
python train.py --data-dir data/images --epochs 15 --batch-size 8 --learning-rate 0.0005 --device cpu
```

**Paramètres :**
- `--epochs 15` : Moins d'époques mais learning rate plus élevé
- `--batch-size 8` : Plus petit pour éviter la surcharge mémoire
- `--learning-rate 0.0005` : 5x plus rapide
- `--device cpu` : Forcer CPU

**Temps estimé : 1.5-2 heures**

### Option 3 : Ultra Rapide (~30-45 min) - Qualité réduite

```bash
python train.py --data-dir data/images --epochs 10 --batch-size 32 --learning-rate 0.002 --device cpu --light
```

**Temps estimé : 30-45 minutes** (mais qualité moindre)

---

## 📊 Comparaison des Options

| Option | Modèle | Époques | Batch | LR | Temps | Qualité |
|--------|--------|---------|-------|----|----|---------|
| 1 (Recommandé) | Léger | 20 | 16 | 0.001 | 1-1.5h | ⭐⭐⭐⭐ |
| 2 | Standard | 15 | 8 | 0.0005 | 1.5-2h | ⭐⭐⭐⭐⭐ |
| 3 | Ultra Rapide | 10 | 32 | 0.002 | 30-45min | ⭐⭐⭐ |

---

## ⚙️ Optimisations Supplémentaires

### Réduire le nombre de workers (si erreurs mémoire)

Modifiez temporairement `train_cv.py` ligne ~160 :
```python
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)  # Au lieu de 4
```

### Utiliser moins de données (pour test rapide)

Créez un sous-dossier avec moins d'images :
```bash
# Prendre seulement 10 images par classe pour test
mkdir -p data/images_test
# Copier quelques images manuellement
python train.py --data-dir data/images_test --epochs 5 --batch-size 16 --device cpu --light
```

---

## 🎯 Commande Finale Recommandée

**Pour un bon compromis vitesse/qualité :**

```bash
python train.py --data-dir data/images --epochs 20 --batch-size 16 --learning-rate 0.001 --device cpu --light
```

**Explication :**
- ✅ Modèle léger (MobileNet) = 3-4x plus rapide
- ✅ 20 époques suffisent avec LR élevé
- ✅ Batch size 16 = bon pour CPU
- ✅ Learning rate 0.001 = convergence rapide
- ✅ Early stopping activé (patience=10) = arrêt automatique si pas d'amélioration

---

## 📝 Monitoring

Pendant l'entraînement, vous verrez :
- Temps par époque
- Loss et accuracy
- Early stopping si pas d'amélioration

Le modèle sera sauvegardé dans `models/cv/best_model.pth` automatiquement.

---

## ⚠️ Notes Importantes

1. **Première époque plus lente** : Le chargement initial prend du temps
2. **Early Stopping** : S'arrête automatiquement si pas d'amélioration pendant 10 époques
3. **Mémoire** : Si erreur mémoire, réduisez `--batch-size` à 8 ou 4
4. **Qualité** : Le modèle léger est légèrement moins performant mais beaucoup plus rapide

---

## 🔍 Vérification après Entraînement

```bash
# Vérifier que le modèle est créé
ls -lh models/cv/best_model.pth

# Voir le rapport d'évaluation
cat models/cv/evaluation_report.txt
```

