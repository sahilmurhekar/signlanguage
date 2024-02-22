from gtts import gTTS
import os

# Text to be converted to speech
with open("input.txt", "r") as file:
    text = file.read()

# Language in which you want to convert
language = "en-au"

# Create a gTTS object
tts = gTTS(text=text, lang=language, slow=False)

# Save the converted audio to a file
tts.save("output.mp3")
