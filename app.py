import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np  # <-- Notice we swapped pad_sequences for numpy!
import pickle

# 1. Setup the UI
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🧠")
st.title("🧠 AI Sentiment Analyzer")
st.write("Type a movie review, tweet, or message below, and the AI will predict the underlying emotion!")


# 2. Load the Brain and the Dictionary
@st.cache_resource
def load_ai():
    model = load_model("sentiment_model.keras")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer


model, tokenizer = load_ai()

# 3. User Input
user_input = st.text_area("Enter your text here:")

if st.button("Analyze Sentiment"):
    if user_input:
        max_length = 120

        # 4. Tokenize the input (Translate English to Math)
        raw_sequence = tokenizer.texts_to_sequences([user_input])[0]

        # Catch words the AI has never seen before
        if len(raw_sequence) == 0:
            st.warning("The AI doesn't recognize any of those words yet! Try a different phrasing.")
        else:
            # 5. THE FIX: The Echo Pad
            # Instead of padding with random zeros, repeat the sentence to fill the 120 columns!
            repeats = (max_length // len(raw_sequence)) + 1
            echoed_sequence = (raw_sequence * repeats)[:max_length]

            # Convert to the 2D matrix Keras expects
            final_matrix = np.array([echoed_sequence])

            # 6. Predict the Emotion
            prediction = model.predict(final_matrix)[0][0]
            positivity_score = prediction * 100

            # 7. Display the Results
            st.markdown("---")
            if prediction >= 0.5:
                st.success(f"**Positive Emotion Detected!** ({positivity_score:.2f}% Positivity)")
            else:
                st.error(f"**Negative Emotion Detected!** ({positivity_score:.2f}% Positivity)")
    else:
        st.warning("Please enter some text first!")