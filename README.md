# TP 2 — TF-IDF et similarité cosinus

## Objectif

Créer un petit moteur de recherche qui compare une requête à cinq documents et les classe du plus proche au moins proche.

## Méthode

1. **Documents :** lire D1 à D5 dans `dataset_tf_idf.json`.
2. **TF-IDF :** transformer les documents en vecteurs numériques avec `fit_transform()`.
3. **Similarité cosinus :** transformer la requête avec `transform()`, puis comparer son vecteur aux cinq documents.
4. **Classement :** trier les scores du plus grand au plus petit. Si aucun mot de la requête n'est connu, afficher un message d'absence de correspondance.

## Résultats

Requête principale : **« train billet réservation »**

| Rang | Document | Similarité cosinus |
|---:|---|---:|
| 1 | D2 | 0.3797 |
| 2 | D1 | 0.3696 |
| 3 | D5 | 0.3568 |
| 4 | D4 | 0.0892 |
| 5 | D3 | 0.0000 |

**D2** est le plus proche de la requête. **D3** ne partage aucun des mots recherchés.

Autres tests :
- **« bus abonnement ville »** : D3 obtient le meilleur score (**0.5633**).
- **« astronomie galaxie »** : aucune correspondance, car ces mots sont absents du vocabulaire.

## Conclusion

Le programme retrouve les documents les plus proches grâce au TF-IDF et à la similarité cosinus. Un score élevé indique davantage de similarité entre les vecteurs, mais ne garantit pas que le document réponde parfaitement à la demande.

## Exécution

```bash
pip install -r requirements.txt
python tp2_tfidf.py
```

**Remarque :** les scores sont calculés avec les paramètres par défaut de `TfidfVectorizer()` (sans suppression spécifique des stopwords). 
