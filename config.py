import os

# Βρίσκει αυτόματα πού βρίσκεται ο φάκελος του project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Φάκελοι (Υποθέτω ότι έβγαλες το test_50_photos έξω από το src)
INPUT_DIR = os.path.join(BASE_DIR, "test_50_photos") 
OUTPUT_DIR = os.path.join(BASE_DIR, "output_photos")

# Ρυθμίσεις Μοντέλου (Χρησιμοποιούμε το πιο γρήγορο/αξιόπιστο για αρχή)
MODEL_NAME = 'ViT-B-32'
PRETRAINED = 'laion2b_s34b_b79k'

# Στόχος συμπίεσης (25%)
TARGET_PERCENTAGE = 0.25