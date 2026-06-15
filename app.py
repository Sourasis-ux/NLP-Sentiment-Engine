import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# 1. Setup the UI
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🧠")
st.title("🧠 AI Sentiment Analyzer")
st.write("Type a movie review, tweet, or message below, and the AI will predict the underlying emotion!")

# 2. Load the Brain and the Dictionary (Cached so it doesn't reload on every keystroke)
@st.cache_resource
def load_ai():
    # Load the math
    model = load_model("sentiment_model.keras")
    # Load the dictionary (un-pickling)
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_ai()

# 3. User Input
user_input = st.text_area("Enter your text here:")

if st.button("Analyze Sentiment"):
    if user_input:
        # 4. Translate English to Math
        # We MUST use the exact same max_length (120) from our training blueprint
        max_length = 120
        
        # Tokenize and pad
        sequence = tokenizer.texts_to_sequences([user_input])
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
        
        # 5. Predict the Emotion
        prediction = model.predict(padded_sequence)[0][0]
        positivity_score = prediction * 100
        
        # 6. Display the Results
        st.markdown("---")
        if prediction >= 0.5:
            st.success(f"**Positive Emotion Detected!** ({positivity_score:.2f}% Positivity)")
        else:
            st.error(f"**Negative Emotion Detected!** ({positivity_score:.2f}% Positivity)")
    else:
        st.warning("Please enter some text first!")