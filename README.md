# Data Mining Project

## IV-Modeling :
Dans cette partie, nous allons : 

1. Préparer les données 
3. Préparer un modèle d'apprentissage automatique 
4. Évaluer les prédictions du modèle 
5. Expérimenter différents modèles de classification 
6. Hyperparameter Tuning : KNN avec RandomzedSearchCV 
7. LGBMClassifier 

```python

```
Nous allons travailler avec la librairie Scikit-learn.

<img src="https://github.com/addi-kamal/data-mining-project/blob/main/sklearn.png"  width="700" height="400" />
                      
Scikit-learn est une librairie pour Python spécialisée dans le machine learning (apprentissage automatique). Elle propose dans son framework de nombreuses algorithmes.

## 1.	Préparer les données :

Nous commencerons par l’élimination des colonnes qu’on va pas utiliser dans notre modélisation :

```python
main_cols = df_pre.columns.difference(['children', 'meal', 'reservation_status', 
                                       'reservation_status_date', 'arrival_date']).tolist()
df_pre = df_pre[main_cols]
df_pre.head()
```

Les colonnes de type « String » :

```python
df_pre_object = df_pre.select_dtypes(include=['object']).copy()
df_pre_object.head()
```


On doit transformer ces colonnes en types « intiger », sinon le modèle ça ne va pas marche :

On utilise la technique « one hot encoding » 

```python
# One hot encoding
df_pre = pd.get_dummies(df_pre, columns = ['hotel', 'market_segment', 'distribution_channel', 'assigned_room_type',
                                           'reserved_room_type', 'deposit_type', 'customer_type'])
```
 
Notre objectif ici est de créer un modèle d'apprentissage automatique sur toutes les colonnes restantes dans le dataframe, à l'exception des targets pour prédire les targets « is_canceled ». 

En substance, la colonne « is_canceled » est notre variable cible (également appelée y ou label) et le reste des autres colonnes sont nos variables indépendantes (également appelées caractéristiques ou X). 

```python
X = df_pre.drop('is_canceled', axis = 1)
y = df_pre['is_canceled']
```

Maintenant que nous avons divisé nos données en X et y, nous utiliserons Scikit-Learn pour les diviser en ensembles d'entraînement et de test (80% de training et 20% de test).

```python
# Splitting Data
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
```

## 2.	Préparer un modèle d'apprentissage automatique :

Étant donné que nos données sont maintenant dans des ensembles d'entraînement et de test, nous allons créer un modèle d'apprentissage automatique pour adapter les modèles dans les données d'entraînement, puis effectuer des prédictions sur les données de test. 

Pour ce faire commencent d’abord par l’importation des modèles qu’on va utiliser et les métriques pour évaluer les performances de ces modèles :

```python
# Libs
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

np.random.seed(100)
```
 
Dans cette partie nous allons appliquer une régression logistique en utilisant le classifier **LogisiticRegression** de la librerie scikit-learn :

La régression logistique porte assez mal son nom car il ne s’agit pas à proprement parler d’une régression au sens classique du terme (on essaye pas d’expliquer une variable quantitative mais de classer des individus dans deux catégories). 

Cette méthode présente depuis de nombreuses années est la méthode la plus utilisée aujourd’hui en production pour construire des modèles de classification. 

Nous pouvons lancer la régression maintenant. Après appel du constructeur de la classe LogisticRegression() où nous passons les données, nous faisons appel à la fonction fit() qui génère un objet résultat doté de propriétés et méthodes qui nous seront très utiles par la suite. 

```python
# training
model_logReg = LogisticRegression()
model_logReg.fit(X_train, y_train)
```

## 3.	Évaluer les prédictions du modèle :

Évaluer les prédictions est très important. Vérifions comment notre modèle en appelant la méthode score() et en lui passant les données d'entraînement (X_train, y_train) et de test (X_test, y_test).
 
 ```python
# performance on training data
model_logReg.score(X_train, y_train)

# performance on test data
model_logReg.score(X_test, y_test)
```

Faisons quelques prédictions sur les données de test en utilisant notre dernier modèle et sauvegardons-les dans y_pred :

```python
# make prediction
y_pred = model_logReg.predict(X_test)
```
 
Il est temps d'utiliser les prédictions que notre modèle a trouvé pour l'évaluer :

### a.	Le Rapport de Classification :

Les colonnes de ce rapport de classification sont :

* Précision - Indique la proportion d'identifications positives (classe 1 prédite par le modèle) qui étaient réellement correctes. Un modèle qui ne produit aucun faux positif a une précision de 1,0. 
* Rappel - Indique la proportion de positifs réels qui ont été correctement classés. Un modèle qui ne produit aucun faux négatif a un rappel de 1,0. 
* Score F1 - Une combinaison de précision et de rappel. Un modèle parfait obtient un score F1 de 1,0. 
* Support - Le nombre d'échantillons sur lequel chaque métrique a été calculée. 
* Précision - La précision du modèle sous forme décimale. La précision parfaite est égale à 1,0. 
* Macro moyenne - Abréviation de macro moyenne, la précision moyenne, le rappel et le score F1 entre les classes. La macro moyenne ne classe pas le déséquilibre en effort, donc si vous avez des déséquilibres de classe, faites attention à cette métrique. 
* Moyenne pondérée - Abréviation de moyenne pondérée, précision moyenne pondérée, rappel et score F1 entre les classes. Pondéré signifie que chaque métrique est calculée par rapport au nombre d'échantillons dans chaque classe. Cette métrique favorisera la classe majoritaire (par exemple, donnera une valeur élevée lorsqu'une classe surpassera une autre en raison du plus grand nombre d'échantillons). 

```python
cl_report = classification_report(y_test, y_pred) 
print(cl_report)
```

### b.	La Matrice de confusion :

Une Confusion Matrix (matrice de confusion) ou tableau de contingence est un outil permettant de mesurer les performances d’un modèle de Machine Learning en vérifiant notamment à quelle fréquence ses prédictions sont exactes par rapport à la réalité dans des problèmes de classification.

Pour bien comprendre le fonctionnement d’une matrice de confusion, il convient de bien comprendre les quatre terminologies principales : TP, TN, FP et FN. Voici la définition précise de chacun de ces termes :

* TP (True Positives) : les cas où la prédiction est positive, et où la valeur réelle est effectivement positive. Exemple : le médecin vous annonce que vous êtes enceinte, et vous êtes bel et bien enceinte.
* TN (True Negatives) : les cas où la prédiction est négative, et où la valeur réelle est effectivement négative. Exemple : le médecin vous annonce que vous n’êtes pas enceinte, et vous n’êtes effectivement pas enceinte.
* FP (False Positive) : les cas où la prédiction est positive, mais où la valeur réelle est négative. Exemple : le médecin vous annonce que vous êtes enceinte, mais vous n’êtes pas enceinte.
* FN (False Negative) : les cas où la prédiction est négative, mais où la valeur réelle est positive. Exemple : le médecin vous annonce que vous n’êtes pas enceinte, mais vous êtes enceinte.


                           
Calcule de la matrice de confusion en python : 

```python
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)
```

Pour mieux voir cette matrice de confusion on peut la visualiser a l’aide de la bibliotheque seaborn et matplotlib.

Cette visualisation permet d'analyser rapidement la matrice de confusion, ainsi analyser l'efficacité du modèle.
 
 ```python
f, ax = plt.subplots(figsize=(10,7))
sns.heatmap(cnf_matrix, annot=True, fmt='.0f', ax=ax)
plt.xlabel('y Actual')
plt.ylabel('y Predicted')
plt.show()
```

## 4.	Expérimenter différents modèles de classification :

Nous allons maintenant essayer une série de différents modèles d'apprentissage automatique et voir lequel obtient les meilleurs résultats sur notre ensemble de données

En parcourant la documentation de Scikit-Learn, nous voyons qu'il existe un certain nombre de modèles de classification différents que nous pouvons essayer.

Pour notre cas, les modèles que nous allons essayer de comparer sont : 

* LogisticRegression
* DecisionTreeClassifier
* RandomForestClassifier
* KNeighborsClassifier
* SVC

Nous suivrons le même workflow que nous avons utilisé ci-dessus (sauf cette fois pour plusieurs modèles): 

1. Importer un modèle d'apprentissage automatique 
2. Préparez-le 
3. Ajustez-le aux données et faites des prédictions 
4. Évaluer le modèle ajusté

Grâce à la cohérence de la conception de l'API de Scikit-Learn, nous pouvons utiliser pratiquement le même code pour ajuster, évaluer et faire des prédictions avec chacun de nos modèles.

Pour voir quel modèle fonctionne le mieux, nous allons procéder comme suit : 

1. Instancier chaque modèle dans un dictionnaire 
2. Créer un dictionnaire de résultats vide 
3. Ajustez chaque modèle sur les données d'entraînement 
4. Evaluer chaque modèle sur les données de test 
5. Vérifiez les résultats

Étant donné que chaque modèle que nous utilisons a les mêmes fonctions fit () et score (), nous pouvons parcourir notre dictionnaire de modèles et appeler fit () sur les données d'entraînement, puis appeler score () avec les données de test

```python
%%time 

models = {
    "LogisticRegression" :LogisticRegression() ,
    "DecisionTreeClassifier" : DecisionTreeClassifier(),
    "RandomForestClassifier" : RandomForestClassifier(),
    "KNN" : KNeighborsClassifier(),
    "SVC" : SVC(),
}
results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    results[model_name] = model.score(X_test, y_test)

sort_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
sort_results
```
 
Le modèle qui fonctionne le mieux pour ce probleme est RandomForestClassifier. 

Mais il est toujours possible d’améliorer la performance des autres algorithmes, prenons par exemple le KNN.

## 5.	Hyperparameter Tuning : KNN avec RandomzedSearchCV :


Pour trouver les meilleurs hyper-parametres  il suffit de créer deux listes Python k_range et weight_options, avec les hyper-paramètres. 

Puisque nous avons un ensemble d'hyperparamètres, nous pouvons importer RandomizedSearchCV, lui transmettre le classifier et nos listes d'hyperparamètres et le laisser rechercher la meilleure combinaison.  

```python
# instantiate model
knn = KNeighborsClassifier(n_neighbors=5)

k_range = list(range(1, 10))
weight_options = ['uniform', 'distance']
param_dist = dict(n_neighbors=k_range, 
                  weights=weight_options)

rand = RandomizedSearchCV(knn, 
                          param_dist, 
                          cv=10, 
                          scoring='accuracy', 
                          n_iter=10, 
                          random_state=5)

# fit the grid with data
rand.fit(X_train, y_train)
```

L’objet rand va conserver le bon parametrage et peut directement faire appel à la fonction predict() par exemple. 

Nous pouvons aussi regarder quel paramétrage a été élu via les propriétés best_score_, best_params_ et best_estimator_. 

```python
# examine the best model
print(rand.best_score_)
print(rand.best_params_)
print(rand.best_estimator_)
```

Cela nous donne le meilleur score obtenu avec la meilleure combinaison.

Nous avons essayé de trouver les meilleurs hyperparamètres sur notre modèle en utilisant RandomizedSearchCV.

Maintenant nous allons instancier une nouvelle instance de notre modèle en utilisant les meilleurs hyperparamètres trouvés par RandomizedSearchCV :

**{'weights': 'distance', 'n_neighbors': 6}**

```python
# train your model using all data and the best known parameters

# instantiate model with best parameters
knn = KNeighborsClassifier(n_neighbors=6, weights='distance')
# train the model
knn.fit(X_train, y_train)
```

Après le processus de réglage des hyperparamètres, le score augmente de 2% :

```python
knn.score(X_test, y_test)
```
 
Faisons quelques prédictions sur les données de test en utilisant notre dernier modèle et sauvegardons-les dans y_pred.

```python
# make prediction
y_pred = knn.predict(X_test)
```
  
Il est temps d'utiliser les prédictions que notre modèle a trouvé pour l'évaluer en utilisent la matrice de confusion :

```python
cnf_matrix = confusion_matrix(y_test, y_pred)
f, ax = plt.subplots(figsize=(10,7))
sns.heatmap(cnf_matrix, annot=True, fmt='.0f', ax=ax)
plt.xlabel('y Actual')
plt.ylabel('y Predicted')
plt.show()
```
  
## 6.	LGBMClassifier :

Maintenant on va travailler avec une autre series d’algorithmes, performants et rapides :

Lightgbm est une bibliothèque qui utilise des algorithmes d'apprentissage basés sur les arbres. Il est conçu pour être distribué et efficace par rapport aux autres algorithmes. Un modèle qui peut être utilisé à des fins de comparaison est XGBoost, qui est également une méthode de stimulation et il fonctionne exceptionnellement bien par rapport aux autres algorithmes. Lightgbm peut gérer une grande quantité de données, moins d'utilisation de la mémoire, un apprentissage parallèle et GPU, une bonne précision, une vitesse d'entraînement plus rapide et une efficacité.


Installons d’abord la librairie lightgbm :

```python
pip install lightgbm
```
  
Le principe est le meme : 

1. Importer le modèle d'apprentissage
2. Instanciez-le 	
3. Ajustez-le aux données et faites des prédictions 
4. Évaluer le modèle ajusté 

```python
%time
from lightgbm import LGBMClassifier
# Train a model
model = LGBMClassifier(random_state=31)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Check score
accuracy_score(y_test, y_pred)
```

Ce modèle a deux avantages par rapport aux autres : 
1.	Il donne la meilleure performance
2.	Il est très rapide en terme de temps d’exécution 

