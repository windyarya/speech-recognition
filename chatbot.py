from cgi import print_arguments
from distutils.command.clean import clean
from email import message
import random
import json
import pickle
import numpy as np
import nltk
import speech_recognition as sr
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import pyttsx3

engine = pyttsx3.init()

r = sr.Recognizer()

lemmatizer = WordNetLemmatizer()
intents = json.load(open('intents.json'))

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_wors = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results= [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("BOT is Running")

while True:
    with sr.Microphone() as source2:
        r.adjust_for_ambient_noise(source2)
        print("Listening...")
        # audio2 = r.listen(source2, timeout=10, phrase_time_limit=10) # ini ada time limit 10 detik
        audio2 = r.listen(source2, phrase_time_limit=10) # ini ada time limit 10 detik
        # audio2 = r.listen(source2) #ini kadang kalo terlalu banyak noise bakal stuck
    try:
        print("Recognizing...")
        message = r.recognize_google(audio2)
        message = message.lower()
        # print("Did you say " + MyText)
    except:
        print("I can't hear you")
    
    ints = predict_class(message)
    res = get_response(ints, intents)
    # output string
    print(res)

    newVoiceRate = 145
    engine.setProperty('rate',newVoiceRate)
    engine.say(res)
    engine.runAndWait()

    if res in ["Goodbye", "See you too", "Talk to you later"]:
        quit()

