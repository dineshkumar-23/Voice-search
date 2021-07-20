from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
import torch
import torchaudio
from flask import Flask, render_template, request, jsonify, redirect

app = Flask(__name__)

tokenizer = Wav2Vec2Tokenizer.from_pretrained('facebook/wav2vec2-base-960h')
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')

def transcribe(file):
    waveform, sample_rate = torchaudio.load(file)
    transformed_sample = torchaudio.transforms.Resample(sample_rate, 16000)(waveform[0,:].view(1, -1))
    transformed_sample = transformed_sample.squeeze()
    raw_speech_input = tokenizer(transformed_sample, padding="longest", return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(raw_speech_input).logits
    predicted_ids = torch.argmax(logits, axis=-1)
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
    app.run(port=8000,debug=True)