# Emotion-Detection-From-Text
This repository contains Python scripts for performing emotion detection using text data. Emotion detection is a natural language processing (NLP) task that involves identifying and categorizing emotions expressed in textual data.

## Dataset
The dataset used for emotion detection consists of three CSV files:
1. **tweet_emotions.csv**: Contains tweet IDs, sentiments, and content.
2. **emotion_data.csv**: Contains sentiments and text.
3. **emotion_to_text.csv**: Contains text and sentiments.

These three datasets are combined into a single file named **final.csv** using the `three_to_one.py` program. The final dataset contains 72,000 records.

## Scripts
1. **three_to_one.py**: Combines the three CSV files into one final dataset.
2. **save_model.py**: Builds and trains a machine learning model for emotion detection using logistic regression. The trained model is saved as `text_emotion.pkl`.
3. **trained_trial.py**: This script asks the user to input a sentence and then displays a bar graph showing the distribution of 14 emotions based on the trained model (`text_emotion.pkl`).

## Usage
1. Clone the repository to your local machine.
    ```bash
    pip clone https://github.com/DiaUdeshi/Emotion-Detection-From-Text
2. Run `three_to_one.py` to combine the three datasets into one CSV file.
3. Execute `save_model.py` to train the emotion detection model and save it as `text_emotion.pkl`.
4. Finally, run `training_trial.py` and input a sentence to see the distribution of emotions displayed in a bar graph.

## Dependencies
- Python 3.x
- pandas
- scikit-learn
- matplotlib
- joblib
