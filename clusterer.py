import numpy as np
from sklearn.cluster import AgglomerativeClustering

class PhotoClusterer:
    def __init__(self, similarity_threshold=0.85):
        # Μετατρέπουμε την ομοιότητα (0.85) σε απόσταση (0.15)
        # Όσο πιο κοντά στο 1 η ομοιότητα, τόσο πιο αυστηρό το φιλτράρισμα.
        distance_threshold = 1.0 - similarity_threshold
        
        self.clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            linkage='average',
            distance_threshold=distance_threshold
        )

    def cluster_photos(self, embeddings_dict):
        """
        Παίρνει ένα λεξικό {όνομα_αρχείου: διάνυσμα} και επιστρέφει τις ομάδες.
        """
        if not embeddings_dict:
            return {}

        filenames = list(embeddings_dict.keys())
        # Ενώνουμε όλα τα διανύσματα σε έναν μεγάλο πίνακα για να τα διαβάσει το scikit-learn
        X = np.vstack(list(embeddings_dict.values()))
        
        print("-> Υπολογισμός ομαδοποίησης (Clustering)...")
        labels = self.clusterer.fit_predict(X)

        # Οργανώνουμε τα αποτελέσματα σε ένα λεξικό {id_ομάδας: [φωτο1, φωτο2...]}
        clusters = {}
        for filename, label in zip(filenames, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(filename)

        return clusters
    