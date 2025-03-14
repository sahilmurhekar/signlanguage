import streamlit as st
import cv2
import pickle
import mediapipe as mp
import numpy as np
import time
import google.generativeai as genai
from gtts import gTTS
import base64
import os

# Configuration
GOOGLE_API_KEY = "AIzaSyAfBnFjJ-80s7iy71wLVGNh2q3NccSjVo0"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)
st.set_page_config(page_title="VisibleVoiceAI 🤲")
# Load the model
@st.cache_resource
def load_model():
    model_dict = pickle.load(open('./model.p', 'rb'))
    return model_dict['model']

# Initialize MediaPipe Hands
@st.cache_resource
def init_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    return mp_hands, mp_drawing, mp_drawing_styles, hands

# Define labels
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
               11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
               21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

def generate_sentence(text):
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"{text} (Frame these words in a proper sentence)"
    response = model.generate_content(prompt)
    return response.text

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_file = "output.mp3"
    tts.save(audio_file)
    return audio_file

def get_audio_player(audio_file):
    with open(audio_file, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        return f'<audio controls><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'

def main():

    # Header Section
    st.title("🤲 VisibleVoiceAI ")
    st.header("Hand Sign Language Translator")
    st.markdown("---")
    st.write("""
    Welcome to **VisibleVoiceAI**! This app transforms your hand gestures into words and sentences, making communication 
    accessible and fun. Whether you're learning sign language or connecting with others, VisibleVoiceAI is here to give 
    your hands a voice. Let’s get started!
    """)
    st.markdown("---")
    # Instructions
    st.subheader("📝 How to Use VisibleVoiceAI")
    st.write("""
    Ready to bring your signs to life? Follow these simple steps to use VisibleVoiceAI:
    1. **Click 'Start Detection'** to turn on the camera and start recognizing your hand signs.
    2. **Show Your Signs**: Hold your hands clearly in front of the camera—think of it as your personal sign language spotlight!
    3. **Click 'Stop Detection'**: Finish signing and let the app process your gestures into text.
    4. **Hear the Magic**: Check out the generated sentence and listen to it with the audio playback!
    """)
    st.info("**Pro Tip:** Good lighting and steady hands make all the difference—give VisibleVoiceAI a clear view!")
    st.markdown("---")
    # Load model and initialize MediaPipe
    model = load_model()
    mp_hands, mp_drawing, mp_drawing_styles, hands = init_mediapipe()
    
    # Session State Initialization
    if 'run' not in st.session_state:
        st.session_state.run = False
    if 'detected_text' not in st.session_state:
        st.session_state.detected_text = ""
    if 'last_prediction_time' not in st.session_state:
        st.session_state.last_prediction_time = 0
    if 'generated_sentence' not in st.session_state:
        st.session_state.generated_sentence = ""
    
    # Control Panel
    st.subheader("🎮 VisibleVoiceAI Control Center")
    st.write("Take charge of your sign language experience with these controls!")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Start Detection",type="primary", key="start"):
            st.session_state.run = True
            st.session_state.generated_sentence = ""
        st.write("Begin capturing your hand signs!")
    with col2:
        if st.button("Stop Detection",type="primary", key="stop"):
            st.session_state.run = False
            if st.session_state.detected_text:
                with st.spinner("Generating sentence..."):
                    st.session_state.generated_sentence = generate_sentence(st.session_state.detected_text)
        st.write("Pause and process your gestures.")
    with col3:
        if st.button("Clear Text",type="primary", key="clear"):
            st.session_state.detected_text = ""
            st.session_state.generated_sentence = ""
        st.write("Reset and start fresh!")
    st.markdown("---")
    # Video and Output Section (Demo)
    st.subheader("📹 Live Video Feed")
    st.write("Here’s where the action happens—watch VisibleVoiceAI decode your signs in real time!")
    frame_window = st.image([])
    st.markdown("---")
    st.subheader("📜 Your Translated Output")
    st.write("See what VisibleVoiceAI hears from your hands—raw signs and polished sentences await!")
    detected_output = st.empty()
    sentence_output = st.empty()
    audio_player = st.empty()
    
    # Camera capture
    cap = cv2.VideoCapture(0)
    delay = 1.5
    
    while st.session_state.run:
        data_aux = []
        x_ = []
        y_ = []
        
        ret, frame = cap.read()
        if not ret:
            continue
            
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)
                    
            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)
            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)
            
            current_time = time.time()
            if current_time - st.session_state.last_prediction_time >= delay and data_aux:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                
                st.session_state.detected_text += predicted_character
                st.session_state.last_prediction_time = current_time
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character if 'predicted_character' in locals() else "", 
                       (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        
        frame_window.image(frame_rgb)
        detected_output.text(f"Detected Signs: {st.session_state.detected_text}")
       
    # Display results after stopping
    if not st.session_state.run:
        detected_output.text(f"Detected Signs: {st.session_state.detected_text}")
        if st.session_state.generated_sentence:
            sentence_output.markdown(f"**Generated Sentence:** {st.session_state.generated_sentence}")
            audio_file = text_to_speech(st.session_state.generated_sentence)
            audio_player.markdown(get_audio_player(audio_file), unsafe_allow_html=True)
            os.remove(audio_file)  # Clean up audio file
    
    cap.release()
    st.markdown("---")
    # Fun Facts Section (Accordion, below detection)
    st.subheader("🎉 Fun Facts About Sign Language")
    with st.expander("Expand to Discover Cool Tidbits!"):
        st.write("""
        Did you know? Sign language is more than just hand gestures—it’s a vibrant way to communicate! Here are some fun facts:
        - There are over 300 sign languages worldwide, each with its own grammar and style.
        - American Sign Language (ASL) is closer to French Sign Language than British Sign Language—history is wild!
        - Skilled signers can “talk” at 120-200 words per minute, matching spoken language speeds.
        - VisibleVoiceAI uses cutting-edge AI to turn your signs into speech—pretty cool, right?
        """)
    st.markdown("---")
    # FAQ Section
    st.subheader("❓ Frequently Asked Questions")
    with st.expander("Got Questions? We’ve Got Answers!"):
        st.write("""
        **Q: Why isn’t VisibleVoiceAI detecting my signs?**  
        A: Ensure your hands are well-lit and in the camera frame. Slow and steady wins the race here!
        
        **Q: Does this work with all sign languages?**  
        A: VisibleVoiceAI is trained on ASL letters A-Z. Other languages or gestures are a future adventure!
        
        **Q: How are sentences created?**  
        A: Google’s generative AI weaves your detected letters into sentences—tech magic at its finest!
        
        **Q: Is my video saved?**  
        A: Not at all! Everything’s real-time, and your privacy stays intact.
        """)
    st.markdown("---")
    # About Section (Last)
    st.subheader("ℹ️ About VisibleVoiceAI")
    with st.expander("Learn More About This Project"):
        st.write("""
        **Creator:** Sahil Murhekar  
        **Version:** 1.0  
        **Last Updated:** March 13, 2025  
        **Description:** VisibleVoiceAI is a real-time hand sign recognition system that translates gestures into text 
        and sentences using AI. Powered by MediaPipe for hand tracking, a custom model for letter detection, and Google’s 
        generative AI for sentence crafting, this app is all about making communication accessible.  
        **Mission:** To give everyone a voice—visible and audible—one sign at a time!
        """)

if __name__ == "__main__":
    main()