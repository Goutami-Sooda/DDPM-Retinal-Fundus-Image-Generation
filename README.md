# Retinal Fundus Image Generator using DDPM (Diffusion Model)

This project demonstrates a Denoising Diffusion Probabilistic Model (DDPM) trained to synthesize retinal fundus images using a U-Net architecture. It uses the Hugging Face Diffusers library for building and training the model. This model can be used for synthetic medical image generation to augment datasets for training diagnostic models or other biomedical tasks.

---

## 🧠 Model Overview

- **Model Type:** Denoising Diffusion Probabilistic Model (DDPM)
- **Architecture:** U-Net
- **Frameworks Used:** PyTorch, Hugging Face Diffusers, Accelerate
- **Trained on:** Diabetic Retinopathy Retinal Fundus Images from a Kaggle dataset

---

## 📂 Repository Contents

| File/Folder | Description |
|-------------|-------------|
| `ddpm_retinal_fundus_image_generation.ipynb` | Main Colab notebook used for training |
| `samples/` | Sample images generated after every 10 epochs of training |
| `model_generated_images/` | Images generated using the final trained model |

---

## 📊 Dataset

- **Name:** Retinal Fundus Images
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/kssanjaynithish03/retinal-fundus-images)
- **Used Subfolder:** `4.Moderate DR (Diabetic Retinopathy)` under `train` folder

---

## 🤗 Hugging Face Model Card

The trained DDPM model along with the scheduler, pipeline, and config files is available on the Hugging Face Hub:

🔗 [View Model on Hugging Face](https://huggingface.co/GS-23/ddpm-unet-retinal-fundus-image-generator)

You can directly use it with the `DDPMPipeline` from Hugging Face Diffusers to generate images as well as further improve the model.

### 🚀 Inference Example

```python
from diffusers import DDPMPipeline
import torch
import matplotlib.pyplot as plt

pipeline = DDPMPipeline.from_pretrained("GS-23/ddpm-unet-retinal-fundus-image-generator")

generated_images = pipeline(batch_size=1, generator=torch.manual_seed(0)).images

for img in generated_images:
    plt.imshow(img)
    plt.axis("off")
    plt.show()
```


## 🙌 Acknowledgments

- Hugging Face `diffusers`, `accelerate`, and `datasets` libraries
- Kaggle dataset contributors
