# IV-Modeling :
Dans cette partie, nous allons : 

1. Préparer les données 
3. Préparer un modèle d'apprentissage automatique 
4. Évaluer les prédictions du modèle 
5. Expérimenter différents modèles de classification 
6. Hyperparameter Tuning : KNN avec RandomzedSearchCV 
7. LGBMClassifier 

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

```
