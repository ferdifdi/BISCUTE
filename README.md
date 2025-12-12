# 🐾 BISCUTE

## 👨‍💻 Team Member [Alphabetical Order]

- Ferdi Fadillah
- Meghana Gudamsetty
- Raquel Brown
- Vincent G. Capone

## Quick Links
- **Presentation**: [Biscute-Presentation](https://github.com/ferdifdi/BISCUTE/blob/main/BISCUTE-presentation.pdf)
- **Our Github for Finetuning the Model**: [cat_breeds_classifier_extended_breeds](https://github.com/ferdifdi/cat_breeds_classifier_extended_breeds)
- **Model**: [ferdifdi/cat-breed-60-classes](https://huggingface.co/ferdifdi/cat-breed-60-classes)

## About this Project

This project is the project for Machine Learning Course (ECE-GY_6143) Fall'25 from New York University.

We want to make and engine that can give recommendation care to our pet (dog and cat), but we need the classifier. Nowadays, a good classifiers is already available, especially the dog one. However, the cat one does not cover some niche cat breeds. 
Hence, we build the website for this case, BISCUTE (Breed Identification & Specific Care Utility Tool Engine). We still use the dog classifier model from internet, but for the cat, we do fine-tuning by ourself.

We divide to 2 tasks:
- Machine learning project (in this github [cat_breeds_classifier_extended_breeds](https://github.com/ferdifdi/cat_breeds_classifier_extended_breeds)) 
    - Only finetuning cat breed classifier
- Web app development (this github)
    - Use the dog breed classifier from internet and use our finetuning cat breed classifier

## About this web app development

**Breed Identification & Specific Care Utility Tool Engine**

A web application that identifies dog and cat breeds from photos and provides detailed, AI-generated care recommendations.

## ✨ Features

- 🔍 **Breed Classification**: Upload a photo to identify dog or cat breeds using state-of-the-art Vision Transformer models
- 📋 **Care Recommendations**: Get detailed care tips including diet, exercise, grooming, health, and temperament

## 🛠️ Technologies

- **Backend**: Flask (Python)
- **ML Models**: Hugging Face Transformers (Vision Transformers)
- **AI Generation**: Groq API (Llama 3.1)
- **Frontend**: HTML, CSS, JavaScript

## 📋 Requirements

- Python 3.8+
- conda (recommended) or pip
- Git

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/ferdifdi/BISCUTE.git
cd BISCUTE
```

### 2. Create and activate conda environment

```bash
conda create -n biscute python=3.11
conda activate biscute
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the ML models

Clone the pre-trained models from Hugging Face:

```bash
# Dog breed classifier
git clone https://huggingface.co/wesleyacheng/dog-breeds-multiclass-image-classification-with-vit
```

# Cat breed classifier
```bash
# cat breed classifier
git clone https://huggingface.co/ferdifdi/cat-breed-60-classes
```
It is our finetuning from https://huggingface.co/dima806/cat_breed_image_detection with extended new 12 cat breeds from this dataset https://www.kaggle.com/datasets/almanaqibmahmooddar/37-cats-breeds-dataset.
The detail of finetuning: https://github.com/ferdifdi/cat_breeds_classifier_extended_breeds/


Or download manually and place them in the project root directory.

### 5. Set up environment variables

Copy the example environment file and add your API key:

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at: https://console.groq.com/

## 🎮 Usage

### Run the application

```bash
python app.py
```

The app will start on `http://localhost:5000`

### Using the app

1. Open your browser and go to `http://localhost:5000`
2. Select whether you want to identify a **Dog** or **Cat**
3. Upload a clear photo of the pet
4. Click **"🔍 Identify Breed & Get Care Tips"**
5. View the predicted breed, confidence score, and detailed care recommendations

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Dog breed model: [wesleyacheng/dog-breeds-multiclass-image-classification-with-vit](https://huggingface.co/wesleyacheng/dog-breeds-multiclass-image-classification-with-vit)
- Cat breed model pretrained that we use for finetuning: [dima806/cat-breed-60-classes-rehearsal_preprocessed_new_breeds](https://huggingface.co/dima806/cat-breed-60-classes-rehearsal_preprocessed_new_breeds)
- Dataset for 12 new breeds for [37 Cats Breeds Dataset](https://www.kaggle.com/datasets/almanaqibmahmooddar/37-cats-breeds-dataset)
