## Revue de code - Guy DELLOYE

Cette revue analyse l'implémentation du pipeline du projet Dynamic Vision Tranformer.


### Checklist des bonnes pratiques de développement :
* **Travail collaboratif :** le gitignore est bien configuré et évite le versionnement des fichiers lourds.
Les commits sont fréquents et les messages de commit informatifs.

* **Qualité du code :** le code suit la convention snake case et respecte les normes PEP8.
Chaque composant du Transformer est isolé (patch_embed.py, attention_head.py).

* **Structure du projet :** le projet adopte une structure modulaire qui sépare correctement le code source (src/), les configurations de l'environnement (pyproject.toml) et les outputs de recherche.
Il privilégie des scripts Python autosuffisants plutôt que des notebooks, garantissant une linéarité du pipeline indispensable à l'automatisation et à la mise en production.

* **Traitement des données volumineuses :** le projet adopte une architecture moderne en privilégiant l'ingestion de données via des APIs qui évite le stockage local massif et respecte le principe de séparation entre calcul et stockage.
Pour l'analyse des logs, l'usage d'Apache Parquet plutôt que du CSV permettrait même des analyses statistiques (OLAP) plus efficientes sur les métriques d'entraînement.

* **Portabilité :** la portabilité est assurée par une gestion rigoureuse des dépendances via le fichier pyproject.toml.
L'application peut ainsi être réexécutée de manière identique sur n'importe quel environnement sans "adhérences" locales.


### Publication reproductible

Le projet est de très haute qualité et utilise un écosystème Quarto intégrant texte, code et visualisations interactives au sein d'un site web automatisé par GitHub Actions.
Les slides au format quarto-revealjs sont très claires : elles résument bien le projet et ses principaux résultats.


### Améliorations suggérées

* **Optimisation du calcul :** généraliser la détection du `device` pour inclure `mps` et renforcer la portabilité multi-OS.
* **Format de persistance :** migrer les logs de performance du CSV vers le format Parquet pour gagner en efficience de stockage et de lecture.


### Conclusion

Le projet est très bien structuré dans un pipeline automatisé, transparent et parfaitement documenté.

La qualité globale du code et de la restitution Quarto font de ce projet un modèle de recherche reproductible exemplaire.