# TO-DO LIST — TIPE SIMULATION COGÉNÉRATION (CYCLE ERICSSON)

## 1. Cadrage du sujet

- [ ] Titre définitif:
    *Étude numérique et optimisation d’un système de cogénération basé sur un cycle Ericsson*
- [ ] Définir la problématique principale:
    *Comment maximiser le rendement global d’un cycle Ericsson en cogénération ?*
- [ ] Lister les hypothèses du modèle :
  - Cycle idéal dans un premier temps
  - Régime permanent
  - Gaz parfait puis gaz réel
  - Régénérateur idéalisé puis efficacité variable
- [ ] Lister les paramètres d’entrée :
  - Tmin : 273 K
  - Tmax : 900 K
  - Pmin : 10E5 Pa
  - rc = Pmax/Pmin : 8
  - Gaz utilisé : Air (dans un premier temps)


## 2. Mise en place de l’environnement Python

- [ ] Installer les bibliothèques nécessaires
  ```
  pip install numpy scipy matplotlib pandas CoolProp streamlit
  ```
- [ ] Tester CoolProp :
  ```python
  from CoolProp.CoolProp import PropsSI
  PropsSI("H","T",300,"P",1e5,"Air")
  ```
- [ ] Créer la structure du projet :
  ```
  TIPE_Ericsson/
  ├── notebooks/
  ├── src/
  ├── figures/
  ├── data/
  ├── app/
  └── README.md
  ```


## 3. Modélisation du cycle Ericsson (gaz parfait)

- [ ] Créer un Jupyter Notebook : `1_ericsson_ideal.ipynb`
- [ ] Définir les constantes :
  - R, Cp, γ
- [ ] Programmer les 4 transformations :
  - 1→2 : compression isotherme
  - 2→3 : chauffage isobare
  - 3→4 : détente isotherme
  - 4→1 : refroidissement isobare
- [ ] Calculer :
  - Travail de chaque étape
  - Chaleur de chaque étape
  - Travail net du cycle
  - Chaleur reçue
  - Chaleur rejetée
  - Rendement thermique
- [ ] Afficher :
  - Tableau des 4 points (T, P, V, s)
  - Diagramme P–V
  - Diagramme T–S


## 4. Modélisation avec CoolProp (gaz réel)

- [ ] Créer un notebook : `2_ericsson_coolprop.ipynb`
- [ ] Importer CoolProp :
  ```python
  from CoolProp.CoolProp import PropsSI
  ```
- [ ] Remplacer Cp constant par Cp réel
- [ ] Calculer :
  - h (enthalpie)
  - s (entropie)
  - densité
- [ ] Comparer résultats :
  - Gaz parfait vs CoolProp
  - Différences de rendement
- [ ] Tester différents gaz :
  - [ ] Air
  - [ ] Helium
  - [ ] CO₂
  - [ ] H₂ (si intéressant)


## 5. Ajout du modèle de cogénération

- [ ] Créer un notebook : `3_cogeneration.ipynb`
- [ ] Ajouter :
  - rendement électrique η_e
  - rendement thermique η_th
- [ ] Calculer :
  - Puissance électrique Pe
  - Puissance thermique récupérée Pth
  - Rendement global η_global
- [ ] Justifier physiquement chaque terme
- [ ] Comparer :
  - Sans récupération
  - Avec récupération


## 6. Étude paramétrique (PARTIE CLÉ DU TIPE)

- [ ] Créer un notebook : `4_etude_parametrique.ipynb`
- [ ] Faire varier :
  - Taux de compression (5 → 30)
  - Efficacité du régénérateur (0 → 1)
  - T° max
- [ ] Tracer :
  - η en fonction de rc
  - Puissance en fonction de Tmax
  - Rendement global en fonction de l’efficacité du récupérateur
- [ ] Identifier un optimum


## 7. Comparaison avec d’autres cycles (BONUS)

- [ ] Implémenter un modèle simple de :
  - Cycle de Brayton
  - (option) Cycle de Stirling
- [ ] Comparer :
  - Rendement
  - Puissance
  - Intérêt pour la cogénération


## 8. Interface graphique (présentation forte)

### Option conseillée : Streamlit

- [ ] Créer un fichier `app.py`
- [ ] Ajouter des sliders :
  - Tmax
  - Pmin
  - Taux de compression
  - Efficacité du régénérateur
- [ ] Ajouter affichage :
  - Valeurs numériques
  - Graphiques générés automatiquement
- [ ] Tester avec :
  ```
  streamlit run app.py
  ```

Objectif final :
▶ Pouvoir montrer au jury ton TIPE en live


## 9. Exploitation des sources

- [ ] Lire et exploiter :
  - Thèses Ananké
  - Livres de thermodynamique
  - Articles cogénération (CHP)
- [ ] Relever :
  - Valeurs réalistes
  - Schémas
  - Données de comparaison
- [ ] Citer tout proprement


## 10. Préparation de l’oral

- [ ] Faire un plan :
  1. Problématique
  2. Modèle physique
  3. Simulation
  4. Résultats
  5. Limites
  6. Ouverture
- [ ] Préparer :
  - Schémas clairs
  - Graphiques lisibles
  - Définitions des rendements
- [ ] Répéter l’oral en 5 puis 10 minutes
- [ ] Préparer les réponses aux questions probables


## OBJECTIF FINAL

- [ ] Simulation fonctionnelle ✅
- [ ] Étude paramétrique complète ✅
- [ ] Résultats exploitables au jury ✅
- [ ] Interface graphique ✅
- [ ] Argumentaire solide ✅
