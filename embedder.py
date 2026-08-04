import torch
from PIL import Image
import open_clip

class ImageEmbedder:
    def __init__(self, model_name, pretrained):
        print(f"-> Φόρτωση του μοντέλου {model_name}... (Αυτό μπορεί να πάρει λίγο την πρώτη φορά)")
        
        # Αν έχεις κάρτα γραφικών Nvidia, θα τη χρησιμοποιήσει, αλλιώς πάει στον επεξεργαστή (CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Φόρτωση του μοντέλου
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.to(self.device)
        self.model.eval() # Το βάζουμε σε mode "αξιολόγησης", όχι εκπαίδευσης

    def get_embedding(self, image_path):
        try:
            # Ανοίγουμε την εικόνα
            image = Image.open(image_path).convert("RGB")
            # Την ετοιμάζουμε για το μοντέλο (resize, crop κλπ που ζητάει το CLIP)
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad(): # Δεν χρειαζόμαστε gradients, γλιτώνουμε RAM
                features = self.model.encode_image(image_input)
                # Κανονικοποίηση (απαραίτητο για το cosine similarity αργότερα)
                features /= features.norm(dim=-1, keepdim=True)
            
            return features.cpu().numpy()
        except Exception as e:
            print(f"Σφάλμα κατά την ανάγνωση της εικόνας {image_path}: {e}")
            return None