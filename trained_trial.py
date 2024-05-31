import joblib
import pandas as pd
import matplotlib.pyplot as plt

pipeline = joblib.load('text_emotion.pkl')
emotions = ['anger','boredom','empty','enthusiasm','fear','fun','hate','joy','love','neutral','relief','sadness','surprise','worry']

def predict_emotions(sentence):
    probabilities = pipeline.predict_proba([sentence])[0]
    emotion_probabilities = {emotion: probability for emotion, probability in zip(emotions, probabilities)}
    plt.figure(figsize=(10, 6))
    plt.bar(emotion_probabilities.keys(), emotion_probabilities.values(), color='skyblue')
    plt.xlabel('Emotion')
    plt.ylabel('Probability')
    plt.title('Emotion Analysis')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

sentence = input("Enter a sentence: ")
predict_emotions(sentence)
