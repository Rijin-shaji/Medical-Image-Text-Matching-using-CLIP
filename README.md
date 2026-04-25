# Medical-Image-Text-Matching-using-CLIP
#  Medical Image–Text Matching using CLIP

##  Overview
This project demonstrates a **Vision-Language Model (VLM)** using OpenAI's CLIP to match medical images with the most relevant textual descriptions. The system encodes both images and text into a shared embedding space and retrieves the best match using similarity scores.

This project highlights how multimodal AI can be applied in healthcare for assisting with image interpretation and decision support.

---

##  Features
- Image-to-text matching using CLIP
- Supports multiple medical descriptions
- Computes similarity using cosine similarity
- Simple CLI-based prediction
- Optional Streamlit UI for interactive usage
- Demonstrates prompt engineering for better results

---

##  How It Works

1. Input image is processed using CLIP's image encoder  
2. Text descriptions are processed using CLIP's text encoder  
3. Both are converted into embeddings in the same vector space  
4. Cosine similarity is calculated between image and text embeddings  
5. The text with the highest similarity score is selected  

---

