import os
import shutil
import config
from src.embedder import ImageEmbedder
from src.clusterer import PhotoClusterer
from src.scorer import PhotoScorer

class PhotoPipeline:
    def __init__(self):
        # Εδώ φορτώνουμε μόνο τα εργαλεία που δεν εξαρτώνται από την επιλογή του χρήστη
        self.embedder = ImageEmbedder(config.MODEL_NAME, config.PRETRAINED)
        self.scorer = PhotoScorer()

    def get_user_threshold(self):
        print("\n=== Ρυθμίσεις Διαγραφής Διπλότυπων ===")
        print("1. Very Loose  (0.65) - Βάζει μαζί και αυτές που απλά μοιάζουν ελαφρώς")
        print("2. Loose       (0.75) - Μέτρια αυστηρότητα")
        print("3. Strict      (0.85) - Βάζει μαζί μόνο πολύ παρόμοιες λήψεις")
        print("4. Very Strict (0.92) - Πρέπει να είναι σχεδόν ολόιδιες (π.χ. burst shots)")
        
        while True:
            choice = input("\nΔιάλεξε επίπεδο (1, 2, 3 ή 4): ")
            if choice == '1': return 0.65
            elif choice == '2': return 0.75
            elif choice == '3': return 0.85
            elif choice == '4': return 0.92
            else: print("Λάθος επιλογή. Παρακαλώ πάτα 1, 2, 3 ή 4.")

    def run(self):
        # 0. Επιλογή Αυστηρότητας
        threshold = self.get_user_threshold()
        
        # Τώρα που ξέρουμε το threshold, φτιάχνουμε τον Clusterer
        self.clusterer = PhotoClusterer(similarity_threshold=threshold)
        
        print(f"\n-> Ενεργοποιήθηκε η ομαδοποίηση με όριο ομοιότητας: {threshold}")
        print("-> Διαγραφή παλιών φακέλων και προετοιμασία νέων...")
        
        # Καθαρισμός φακέλων
        if os.path.exists(config.OUTPUT_DIR):
            shutil.rmtree(config.OUTPUT_DIR)
        os.makedirs(config.OUTPUT_DIR)
        
        debug_dir = os.path.join(config.BASE_DIR, "debug_clusters")
        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)
        os.makedirs(debug_dir)

        valid_extensions = ('.png', '.jpg', '.jpeg')
        image_files = [f for f in os.listdir(config.INPUT_DIR) if f.lower().endswith(valid_extensions)]
        
        if not image_files:
            print(f"Δεν βρέθηκαν φωτογραφίες στο {config.INPUT_DIR}")
            return
            
        print(f"-> Βρέθηκαν {len(image_files)} φωτογραφίες.")

        # 1. Εξαγωγή Διανυσμάτων (Embeddings)
        print("-> Εξαγωγή διανυσμάτων...")
        embeddings_dict = {}
        for filename in image_files:
            path = os.path.join(config.INPUT_DIR, filename)
            vec = self.embedder.get_embedding(path)
            if vec is not None:
                embeddings_dict[filename] = vec
                
        # 2. Ομαδοποίηση (Clustering)
        clusters = self.clusterer.cluster_photos(embeddings_dict)
        print(f"-> Δημιουργήθηκαν {len(clusters)} τελικές ομάδες.")

        # Οπτικός έλεγχος των clusters
        for cluster_id, photos in clusters.items():
            cluster_folder = os.path.join(debug_dir, f"Cluster_{cluster_id}")
            os.makedirs(cluster_folder, exist_ok=True)
            for photo in photos:
                src_path = os.path.join(config.INPUT_DIR, photo)
                dst_path = os.path.join(cluster_folder, photo)
                shutil.copy2(src_path, dst_path)

        # 3. Επιλογή Μίας και Μοναδικής Φωτογραφίας ανά Cluster
        final_selection = []
        
        for cluster_id, photos in clusters.items():
            if len(photos) == 1:
                final_selection.append(photos[0])
            else:
                best_photo = None
                best_score = -1
                
                # Αξιολογούμε ποια είναι η πιο καθαρή
                for p in photos:
                    score = self.scorer.get_sharpness_score(os.path.join(config.INPUT_DIR, p))
                    if score > best_score:
                        best_score = score
                        best_photo = p
                        
                final_selection.append(best_photo)

        # 4. Αποθήκευση Τελικών Επιλογών
        print(f"\n=== Τελική Επιλογή: Κρατήσαμε {len(final_selection)} φωτογραφίες (1 από κάθε ομάδα) ===")
        for photo in final_selection:
            src_path = os.path.join(config.INPUT_DIR, photo)
            dst_path = os.path.join(config.OUTPUT_DIR, photo)
            shutil.copy2(src_path, dst_path)
            print(f" - Διατηρήθηκε: {photo}")
            
        print(f"\nΕπιτυχία! Μπορείς να ελέγξεις τα Clusters στον φάκελο 'debug_clusters' και τις τελικές επιλογές στον φάκελο 'output_photos'.")