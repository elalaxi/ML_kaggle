# A faire :
* `Extracurricular_Activities` : one hot encode
* Normaliser les valeurs
* Vérifier la distribution de la classe cible
* Fixer la graine pour la séparation en train / test
* fit
* predict
* Si régression linéaire, plot les coeff en diagramme en bâtons

## Modèles à tester :
* naive_bayes.GaussianNB() (de la bibliothèque naive_bayes)
* svm.SVC()
* XGBoost (xgb.XGBClassifier())
* catboost (ça semble remplacer XGBoost surtout quand les variables sont catégorielles)
* tree.DecisionTreeClassifier(max_depth=?)
* RandomForestClassifier(random_state=0)
* LogisticRegression() (de `sklearn.linear_model`)

## Problèmes auxquels faire attention :
* 127 valeurs dupliquées
* 