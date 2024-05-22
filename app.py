import base64
from io import BytesIO
import os
import pickle
from uuid import uuid4

import speech_recognition as sr
from flask import Flask, render_template, request, jsonify, session
from PIL import Image
from sentence_transformers import SentenceTransformer, util

# load model
model = SentenceTransformer("all-MiniLM-L6-v2")

app = Flask(__name__)
app.secret_key = str(uuid4())

# Function to convert image to base64 for HTML display
def image_to_base64(image):
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    return img_str

# Speech-to-text function
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak something...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        text = ""
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            print("You said:", text)
        except sr.UnknownValueError:
            print("Sorry, I could not understand what you said.")
        except sr.RequestError as e:
            print("Sorry, an error occurred:", e)
        except sr.WaitTimeoutError:
            print("Sorry, I could not hear what you said.")
        session['speech_text'] = text
        return text

# Text-to-image function
def get_image_from_text(text):
    flowers = pickle.load(open("model.pkl", "rb"))
    
    # Convert text to embedding
    embedding = model.encode(text, convert_to_tensor=True)

    # Sort flowers by similarity
    flowers.sort(key=lambda x: util.cos_sim(embedding, x[1]).sum(), reverse=True)
    
    # Convert image to base64. We only take the first 4 most similar images
    img_base64 = [image_to_base64(Image.open(image[0])) for image in flowers[:4] if os.path.exists(image[0])]
    
    return img_base64

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get text from speech
        if session.get('speech_text'):
            # get images from text
            images_base64 = get_image_from_text(session['speech_text'])
            return render_template("index.html", images=images_base64)
    
    return render_template("index.html", images=None)

@app.route("/get_speech_text/")
def get_speech_text():
    speech_text = speech_to_text()
    return jsonify({"text": speech_text})

if __name__ == "__main__":
    app.run(debug=True)
