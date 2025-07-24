import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Vegetable Classifier",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define class names (must match your training labels)
CLASS_NAMES = [
    'Bean',
    'Bitter_Gourd',
    'Bottle_Gourd',
    'Brinjal',
    'Broccoli',
    'Cabbage',
    'Capsicum',
    'Carrot',
    'Cauliflower',
    'Cucumber',
    'Papaya',
    'Potato',
    'Pumpkin',
    'Radish',
    'Tomato'
]


# Function to load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('vegetable_classifier_updated.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Function to preprocess image
def preprocess_image(uploaded_image, target_size=(128, 128)):
    try:
        img = Image.open(uploaded_image)
        img = img.resize(target_size)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

# Function to make prediction
def predict_image(model, img_array):
    try:
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        return CLASS_NAMES[predicted_class_idx], confidence
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

# Main app
def main():
    # Load model
    model = load_model()
    
    # Custom CSS for styling
    st.markdown("""
    <style>
        .header {
            background: linear-gradient(135deg, #43cea2, #185a9d);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .prediction-card {
            background-color: #f8f9fa;
            border-radius: 15px;
            padding: 2rem;
            margin-top: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-left: 5px solid #43cea2;
        }
        .confidence-bar {
            height: 25px;
            background-color: #e9ecef;
            border-radius: 12px;
            margin: 15px 0;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #43cea2, #185a9d);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        .sample-img {
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 5px;
        }
        .sample-img:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .vegetable-info {
            background-color: #e8f4f1;
            border-radius: 15px;
            padding: 1.5rem;
            margin-top: 1rem;
        }
        .footer {
            text-align: center;
            margin-top: 2rem;
            padding: 1rem;
            color: #6c757d;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="header">
        <h1>🥦 Vegetable Image Classifier</h1>
        <p>Upload an image of a vegetable and our AI will identify it</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Upload Your Vegetable Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        # Sample images
        st.markdown("### Or try a sample image:")
        sample_cols = st.columns(4)
        sample_images = [
    "https://images.unsplash.com/photo-1601050690065-0d47c063c948?auto=format&fit=crop&w=300&q=80",  # Replaced broken one
    "https://images.unsplash.com/photo-1593629716493-3f5c69d5e9e4?auto=format&fit=crop&w=300&q=80",
    "https://images.unsplash.com/photo-1598170845058-32b9d6a5da37?auto=format&fit=crop&w=300&q=80",
    "https://images.unsplash.com/photo-1518977676601-b53f82aba655?auto=format&fit=crop&w=300&q=80"
]


        for i, img_url in enumerate(sample_images):
            with sample_cols[i]:
                if st.button(f"Sample {i+1}", key=f"sample_{i}"):
                    uploaded_file = img_url
                st.image(img_url, use_column_width=True, caption=f"Sample {i+1}", output_format="PNG")
        
        # Display uploaded image
        if uploaded_file is not None:
            try:
                if isinstance(uploaded_file, str):  # If it's a URL from sample image
                    img = Image.open(requests.get(uploaded_file, stream=True).raw)
                else:  # If it's an uploaded file
                    img = Image.open(uploaded_file)
                
                # Display image
                st.image(img, caption="Uploaded Image", use_column_width=True)
                
                # Preprocess and predict
                img_array, _ = preprocess_image(uploaded_file)
                
                if img_array is not None:
                    with st.spinner('Analyzing your vegetable...'):
                        predicted_class, confidence = predict_image(model, img_array)
                    
                    if predicted_class and confidence:
                        # Prediction card
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h2>Prediction Result</h2>
                            <h3 style="color: #185a9d;">{predicted_class}</h3>
                            <p>Our AI is <b>{confidence*100:.2f}% confident</b> about this identification</p>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {confidence*100}%">
                                    {confidence*100:.2f}%
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Vegetable info
                        veg_info = get_vegetable_info(predicted_class)
                        if veg_info:
                            st.markdown(f"""
                            <div class="vegetable-info">
                                <h3>About {predicted_class}</h3>
                                <p><strong>Nutrition Facts:</strong> {veg_info['nutrition']}</p>
                                <p><strong>Cooking Tips:</strong> {veg_info['tips']}</p>
                                <p><strong>Storage:</strong> {veg_info['storage']}</p>
                            </div>
                            """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    with col2:
        # Class information
        st.subheader("Vegetable Classes")
        st.markdown("""
        <p>Our model can identify these 15 vegetable types:</p>
        <div style="max-height: 400px; overflow-y: auto; margin-top: 1rem; border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px;">
        """, unsafe_allow_html=True)
        
        # Display all vegetable classes with icons
        veg_icons = {
            "Tomato": "🍅", "Radish": "🌱", "Pumpkin": "🎃", "Potato": "🥔", "Papaya": "🥭",
            "Cucumber": "🥒", "Cauliflower": "🥦", "Carrot": "🥕", "Capsicum": "🫑",
            "Cabbage": "🥬", "Broccoli": "🥦", "Brinjal": "🍆", "Bottle_Gourd": "🥒",
            "Bitter_Gourd": "🥒", "Bean": "🫛"
        }
        
        for veg in CLASS_NAMES:
            icon = veg_icons.get(veg, "🥬")
            st.markdown(f"<div style='padding: 10px; border-bottom: 1px solid #eee;'>{icon} <b>{veg}</b></div>", 
                        unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Model information
        st.subheader("About the Model")
        st.markdown("""
        This vegetable classifier uses a Convolutional Neural Network (CNN) trained on:
        - 15,000 training images
        - 15 vegetable classes
        - 15 epochs of training
        
        The model architecture includes:
        - 4 convolutional layers with max pooling
        - 512-unit dense layer with dropout
        - Softmax output layer
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Vegetable Image Classifier | Powered by TensorFlow & Streamlit | Model: vegetable_identify.h5</p>
    </div>
    """, unsafe_allow_html=True)

# Get vegetable information
def get_vegetable_info(vegetable):
    info = {
        "Tomato": {
            "nutrition": "Rich in lycopene, vitamin C, potassium, folate, and vitamin K.",
            "tips": "Best when vine-ripened. Use in sauces, salads, sandwiches, or roasted.",
            "storage": "Store at room temperature away from direct sunlight."
        },
        "Radish": {
            "nutrition": "Good source of vitamin C, folate, fiber, and potassium.",
            "tips": "Delicious raw in salads, pickled, or roasted with olive oil and herbs.",
            "storage": "Remove greens and store in refrigerator crisper drawer for 1-2 weeks."
        },
        "Pumpkin": {
            "nutrition": "High in beta-carotene, vitamin A, vitamin C, potassium, and fiber.",
            "tips": "Great for soups, pies, roasting, or purees. Roast seeds for a healthy snack.",
            "storage": "Store whole pumpkins in cool, dark place for several months."
        },
        "Potato": {
            "nutrition": "Good source of vitamin C, vitamin B6, potassium, and dietary fiber.",
            "tips": "Bake, boil, roast or mash. Store in cool, dark place away from onions.",
            "storage": "Keep in cool, dark, well-ventilated place for several weeks."
        },
        "Papaya": {
            "nutrition": "Rich in vitamin C, folate, vitamin A, magnesium, and antioxidants.",
            "tips": "Eat fresh, in salads or smoothies. Green papaya can be used in salads.",
            "storage": "Store ripe papayas in refrigerator; unripe at room temperature."
        },
        "Cucumber": {
            "nutrition": "High in water content, vitamin K, and antioxidants.",
            "tips": "Best eaten raw in salads, sandwiches, or as pickles.",
            "storage": "Refrigerate in plastic wrap for up to one week."
        },
        "Cauliflower": {
            "nutrition": "Rich in vitamin C, vitamin K, folate, and fiber.",
            "tips": "Roast whole, make cauliflower rice, or use in place of potatoes.",
            "storage": "Store in refrigerator in plastic bag for up to one week."
        },
        "Carrot": {
            "nutrition": "Excellent source of beta-carotene, fiber, vitamin K1, and potassium.",
            "tips": "Delicious raw, roasted, or steamed. Great in soups and stews.",
            "storage": "Remove greens and store in refrigerator crisper for several weeks."
        },
        "Capsicum": {
            "nutrition": "High in vitamin C, vitamin A, vitamin B6, and folate.",
            "tips": "Great raw in salads, stuffed with grains, or roasted.",
            "storage": "Store unwashed in refrigerator crisper for up to 10 days."
        },
        "Cabbage": {
            "nutrition": "Rich in vitamin C, vitamin K, fiber, and antioxidants.",
            "tips": "Use raw in coleslaw, fermented as sauerkraut, or cooked in soups.",
            "storage": "Keep whole heads in refrigerator for up to two months."
        },
        "Broccoli": {
            "nutrition": "Packed with vitamins C, K, and A. High in fiber and antioxidants.",
            "tips": "Steam lightly to preserve nutrients. Great in stir-fries and salads.",
            "storage": "Store in refrigerator crisper for 3-5 days."
        },
        "Brinjal": {
            "nutrition": "Good source of fiber, potassium, vitamin C, and antioxidants.",
            "tips": "Excellent grilled, roasted, or in dishes like ratatouille or baba ganoush.",
            "storage": "Store in refrigerator for 5-7 days."
        },
        "Bottle_Gourd": {
            "nutrition": "Low in calories, high in water content, good source of vitamin C.",
            "tips": "Common in Indian cuisine. Can be stuffed, added to curries, or made into soup.",
            "storage": "Store in refrigerator for up to one week."
        },
        "Bitter_Gourd": {
            "nutrition": "Rich in vitamins A and C, iron, potassium, and antioxidants.",
            "tips": "Soak in salt water to reduce bitterness before cooking in stir-fries or curries.",
            "storage": "Store in refrigerator for 3-4 days."
        },
        "Bean": {
            "nutrition": "Excellent source of plant protein, fiber, iron, and folate.",
            "tips": "Steam, sauté, or add to soups and stews. Blanch before freezing.",
            "storage": "Refrigerate in plastic bag for up to one week."
        }
    }
    return info.get(vegetable, {
        "nutrition": "Rich in vitamins, minerals, and antioxidants.",
        "tips": "Delicious when fresh and properly prepared.",
        "storage": "Store according to specific vegetable requirements."
    })

if __name__ == "__main__":
    main()