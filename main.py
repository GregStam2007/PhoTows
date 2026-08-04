import os
import config
from src.pipeline import PhotoPipeline

def main():
    print("=== Έναρξη PhoTows ===")
    
    # Προετοιμασία του φακέλου εξόδου
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        
    # Κλήση του Μαέστρου
    pipeline = PhotoPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()