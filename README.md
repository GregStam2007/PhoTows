# PhoTows 📸🚜

**Tow away your duplicate photos!** 

PhoTows is an intelligent photo deduplication and culling pipeline. Instead of just looking at file names or sizes, it uses AI embeddings to actually "see" and group visually similar photos, and then automatically selects the sharpest image from each group. 

It is perfect for cleaning up burst shots, similar selfies, or repetitive landscape photos without manually reviewing every single one.

## ✨ Features

*   **AI-Powered Vision:** Uses state-of-the-art **OpenCLIP** (`ViT-B-32`) to extract visual features (embeddings) from your photos.
*   **Smart Clustering:** Groups similar photos together using Scikit-Learn's `AgglomerativeClustering` (Cosine Similarity).
*   **Adjustable Sensitivity:** Choose how aggressive you want the deduplication to be via a simple CLI menu (from "Very Loose" to "Very Strict").
*   **Automatic Sharpness Scoring:** Uses **OpenCV** (Laplacian variance) to evaluate the blur/sharpness of images in a cluster and automatically picks the best one.
*   **Non-Destructive:** Copies the selected photos to an `output_photos` folder without deleting your original files.
*   **Debug Mode:** Generates a `debug_clusters` folder so you can visually inspect how the AI grouped your photos.
