## Medical Image-Text Matching using CLIP

## Overview
This project demonstrates a Vision-Language Model (VLM) using CLIP to match medical images with the most relevant textual descriptions. The system encodes both images and text into a shared embedding space and retrieves the best match using similarity scores.

This project highlights how multimodal AI can be applied in healthcare for assisting with image interpretation and decision support.

---

## Features
- Image-to-text matching using CLIP
- Supports multiple medical descriptions
- Computes similarity using cosine similarity
- Command-line based prediction
- Demonstrates prompt engineering for improved results

---

## How It Works

1. The input image is processed using CLIP's image encoder  
2. Text descriptions are processed using CLIP's text encoder  
3. Both image and text are converted into embeddings in a shared vector space  
4. Cosine similarity is computed between the image and text embeddings  
5. The text with the highest similarity score is selected as the final prediction  

---


