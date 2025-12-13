import os
# Force transformers to use PyTorch only (disable TensorFlow and JAX)
os.environ['USE_TF'] = 'NO'
os.environ['USE_TORCH'] = 'YES'
os.environ['USE_JAX'] = 'NO'

from flask import Flask, request, render_template, jsonify
import requests
import base64
from io import BytesIO
from PIL import Image
from groq import Groq
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

app = Flask(__name__)

# API Keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Local Model paths (update these to your model locations)
DOG_MODEL_PATH = "./dog-breeds-multiclass-image-classification-with-vit"
CAT_MODEL_PATH = "./cat-breed-60-classes"

groq_client = Groq(api_key=GROQ_API_KEY)

# Load models at startup
try:
    print("Loading dog breed model...")
    dog_processor = AutoImageProcessor.from_pretrained(DOG_MODEL_PATH)
    dog_model = AutoModelForImageClassification.from_pretrained(DOG_MODEL_PATH)
    dog_model.eval()
    print("Dog model loaded successfully!")
except Exception as e:
    print(f"Error loading dog model: {e}")
    dog_processor = None
    dog_model = None

try:
    print("Loading cat breed model...")
    cat_processor = AutoImageProcessor.from_pretrained(CAT_MODEL_PATH)
    cat_model = AutoModelForImageClassification.from_pretrained(CAT_MODEL_PATH)
    cat_model.eval()
    print("Cat model loaded successfully!")
except Exception as e:
    print(f"Error loading cat model: {e}")
    cat_processor = None
    cat_model = None

def classify_image(img, pet_type):
    try:
        # Select model based on pet type
        if pet_type == "dog":
            processor = dog_processor
            model = dog_model
        else:
            processor = cat_processor
            model = cat_model
        
        # Check if model is loaded
        if processor is None or model is None:
            return None, None, f"{pet_type.title()} model not loaded. Check server logs."
        
        # Process image
        inputs = processor(images=img, return_tensors="pt")
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_idx = logits.argmax(-1).item()
            confidence = torch.softmax(logits, dim=-1)[0][predicted_idx].item()
        
        # Get breed name
        breed = model.config.id2label[predicted_idx]
        
        return breed, confidence, None
        
    except Exception as e:
        return None, None, str(e)

def generate_care_tips(breed, pet_type):
    prompt = f"""You are a pet care expert. Provide specific care recommendations for a {breed} {pet_type}. 
    
Please include the following sections (keep it concise but informative):
1. **Overview**: Brief description of the breed
2. **Diet & Nutrition**: Feeding recommendations
3. **Exercise Needs**: Activity requirements
4. **Grooming**: Coat care and grooming frequency
5. **Health Considerations**: Common health issues to watch for
6. **Temperament**: Personality traits and behavior tips

Keep each section to 2-3 sentences."""
    
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=800
    )
    
    return chat_completion.choices[0].message.content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        pet_type = request.form.get('pet_type')
        file = request.files.get('image')
        
        if not file:
            return jsonify({'error': 'No image uploaded'}), 400
        
        # Read and process image
        image_bytes = file.read()
        img = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Classify using local model
        breed, confidence, error = classify_image(img, pet_type)
        
        if not breed:
            error_msg = error if error else 'Classification failed.'
            return jsonify({'error': error_msg}), 500
        
        # Generate care tips
        care_tips = generate_care_tips(breed, pet_type)
        
        # Convert image to base64 for display
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'breed': breed.replace('_', ' ').title(),
            'confidence': f"{confidence * 100:.2f}%",
            'care_tips': care_tips,
            'image': f"data:image/jpeg;base64,{img_base64}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=True, reloader_type='stat')
