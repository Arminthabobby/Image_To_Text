import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
from deep_translator import GoogleTranslator

# Load BLIP model and processor
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")

# Translator
translator = GoogleTranslator(source='en', target='ta')

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model.to(device)

def generate_caption(image):
    try:
        # Preprocess the image
        inputs = caption_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device)
        
        # Generate caption
        output_ids = caption_model.generate(
            pixel_values,
            max_length=30,
            num_beams=10,
            num_beam_groups=2,
            diversity_penalty=0.5,
            repetition_penalty=2.0,
            temperature=0.6,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=3
        )
        
        # Decode English caption
        english_caption = caption_processor.decode(output_ids[0], skip_special_tokens=True)
        
        # Translate to Hindi
        hindi_caption = translator.translate(english_caption)
        
        return hindi_caption
    
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
interface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="tamil Caption"),
    title="Image Caption Generator (Tamil)",
    description="Upload an image to get a caption in tamil."
)

interface.launch()
