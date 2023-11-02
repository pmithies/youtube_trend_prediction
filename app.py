from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
model = load_model('youtube_model.h5')

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts([])  # Fit the tokenizer with an empty list, as it's already trained.

@app.route('/')
def index():
    return render_template('index.html', prediction='')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    new_text_sequence = tokenizer.texts_to_sequences([text])
    new_text_padded = pad_sequences(new_text_sequence, maxlen=100)
    prediction = model.predict(new_text_padded)

    sentiment = "Trending" if prediction[0][0] >= 0.5 else "Not Trending"
    sentiment_class = "positive" if prediction[0][0] >= 0.5 else "negative"

    return render_template('index.html', prediction=sentiment, sentiment_class=sentiment_class)


if __name__ == '__main__':
    app.run(debug=True)
