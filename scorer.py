import cv2

class PhotoScorer:
    def __init__(self):
        pass

    def get_sharpness_score(self, image_path):
        try:
            # Διαβάζουμε την εικόνα σε ασπρόμαυρο (βοηθάει στην ταχύτητα, δεν χρειαζόμαστε χρώμα για το blur)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return 0.0
            
            # Υπολογίζουμε τη διακύμανση του Laplacian. 
            # Υψηλό σκορ = καθαρή εικόνα (sharp). Χαμηλό σκορ = θολή (blurry).
            score = cv2.Laplacian(image, cv2.CV_64F).var()
            return score
            
        except Exception as e:
            print(f"Σφάλμα κατά τη βαθμολόγηση της {image_path}: {e}")
            return 0.0