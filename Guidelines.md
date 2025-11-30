

````markdown
# Guide d'utilisation du projet Vit-Pytorch

## 1. Se placer dans le répertoire du projet
```bash
cd Vit-Pytorch
````

## 2. Gestion des dépendances avec `uv`

* Installer `uv` si ce n'est pas déjà fait :

```bash
pip install uv
```

* Ajouter un package :

```bash
uv add <nom_du_package>
```

* Installer toutes les dépendances listées dans `pyproject.toml` :

```bash
uv pip install -r pyproject.toml
```

> Remarque : Cela est plus propre que d'utiliser un fichier `requirements.txt`.

## 3. Activer l'environnement virtuel

```bash
source .venv/bin/activate
```

## 4. Accéder aux paramètres du modèle

* Pour ImageNet : `configs/train_imagenet1k.py`
* Pour CIFAR : `configs/train_cifar10.py`

**⚠️ Attention :**  
Pour éviter d’écraser les plots déjà générés (qui ont pris 2 à 3 jours à créer), **modifiez impérativement le chemin `plot_dir`**. Il faut faire de même pour `checkpoint_dir`.

- Pour les tests, utilisez un dossier temporaire comme `plots_test`.  
- **Ne jamais utiliser `plots_Vit_Saved`**, au risque d’écraser les résultats existants.


## 5. Lancer le code

```bash
python
uv run main_Imagenet.py
```
ou 

```bash
python
python -m main_Imagenet.py
```

## 6. Exécuter les tests

* Lancer tous les tests :

```bash
python -m pytest tests/
```

* Lancer les tests avec plus de détails :

```bash
python -m pytest -v tests/
```

* Lancer un test précis par son nom :

```bash
python -m pytest -v tests/test_Vit/test_attention_head.py
```


