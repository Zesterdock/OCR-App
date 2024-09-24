import streamlit as st
from PIL import Image
import torch
from transformers import AutoFeatureExtractor, AutoModelForVision2Seq
import re

# Load the GOT model and feature extractor
model_name = "microsoft/got-base"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(model_name)

def perform_ocr(image):
    # Prepare the image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Generate OCR output
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    
    # Decode the output
    ocr_text = model.decode(outputs[0], skip_special_tokens=True)
    return ocr_text

def search_text(text, keyword):
    # Simple case-insensitive search
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    return pattern.findall(text)

def main():
    st.title("OCR and Keyword Search App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Perform OCR'):
            with st.spinner('Processing...'):
                ocr_result = perform_ocr(image)
            st.subheader("Extracted Text:")
            st.text_area("OCR Result", ocr_result, height=200)
            
            # Save OCR result in session state
            st.session_state['ocr_result'] = ocr_result
        
        # Keyword search
        if 'ocr_result' in st.session_state:
            keyword = st.text_input("Enter a keyword to search:")
            if keyword:
                search_results = search_text(st.session_state['ocr_result'], keyword)
                st.subheader("Search Results:")
                if search_results:
                    st.write(f"Found {len(search_results)} occurrences of '{keyword}'")
                    highlighted_text = st.session_state['ocr_result']
                    for result in search_results:
                        highlighted_text = highlighted_text.replace(result, f"**{result}**")
                    st.markdown(highlighted_text)
                else:
                    st.write("No matches found.")

if __name__ == "__main__":
    main()