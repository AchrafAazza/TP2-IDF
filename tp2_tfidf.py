import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Charger les cinq documents et les requêtes depuis le fichier JSON.
chemin = Path(__file__).with_name("dataset_tf_idf.json")
with open(chemin, encoding="utf-8") as fichier:
    donnees = json.load(fichier)

ids = [doc["id"] for doc in donnees["documents"]]
documents = [doc["texte"] for doc in donnees["documents"]]

# 2. Transformer les documents en vecteurs TF-IDF.
vectoriseur = TfidfVectorizer()
matrice = vectoriseur.fit_transform(documents)

# 3 et 4. Comparer une requête, puis classer les documents.
def rechercher(requete):
    vecteur = vectoriseur.transform([requete])
    print(f"\nRequête : {requete}")

    if vecteur.nnz == 0:
        print("Aucune correspondance : les mots sont absents du vocabulaire.")
        return

    scores = cosine_similarity(vecteur, matrice)[0]
    classement = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)

    for identifiant, score in classement:
        print(f"{identifiant} : {score:.4f}")

# Requête principale, puis deux autres tests prévus dans le fichier JSON.
rechercher(donnees["requete_principale"])
for requete in donnees["autres_requetes"]:
    rechercher(requete)
