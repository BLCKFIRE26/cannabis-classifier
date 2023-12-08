import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

#load the model 
model = load_model('sativa_indica_model.h5')

#streamlit
def main():
    st.title("Sativa or Indica??")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption = "Uploaded Image.", use_column_width=True)
        
                # Preprocess the image
        image = image.resize((64, 64))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0  # Normalize pixel values to [0, 1]

        # Make prediction
        result = model.predict(image_array)
        if result[0][0] > 0.5:
            prediction = 'Cannabis Sativa'
        else:
            prediction = 'Cannabis Indica'

        st.subheader("Prediction:")
        st.write(f"The uploaded image is classified as: {prediction}")

if __name__ == "__main__":
    main()
