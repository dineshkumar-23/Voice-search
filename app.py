from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
import librosa
import torch
import sounddevice as sd
from flask import Flask, render_template, request, jsonify, redirect

app = Flask(__name__)

tokenizer = Wav2Vec2Tokenizer.from_pretrained('facebook/wav2vec2-base-960h')
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')

def transcribe(file):
    audio, rate = librosa.load(file, sr=16000)
    input_values = tokenizer(audio, padding='longest', return_tensors='pt').input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)
    return transcription[0]


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    if request.method == "POST":
        print("Data received")
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            transcript = transcribe(file)
    return render_template('index.html', transcript=transcript)


if __name__ == "__main__":
    app.run(debug=True)