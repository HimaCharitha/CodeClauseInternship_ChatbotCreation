# Simple Chatbot with NLP (NLTK & spaCy)

This repository contains a simple yet effective chatbot built using natural language processing (NLP) libraries, specifically NLTK and spaCy. The bot is trained on predefined responses and can interact with users based on keyword matching and intent recognition.

## Features

- **Predefined Responses**: The chatbot is equipped with a set of predefined responses for common interactions like greetings, farewells, and requests for help.
- **Keyword Matching**: Utilizes keyword matching to determine user intent and respond appropriately.
- **NLP Processing**: Leverages spaCy for tokenization and lemmatization of user input to enhance understanding.
- **Interactive Chat**: Runs an interactive chat session in the terminal, allowing users to engage in conversation with the bot.
- **Extendable**: Easily extend the bot’s functionality by adding more keywords and responses or integrating more advanced NLP techniques.

## Getting Started

### Prerequisites

Make sure you have Python installed along with the following libraries:

- `spaCy`
- `NLTK`
- `Python`

## Running the Chatbot
To start the chatbot, run the chatbot.ipynb Jupyter notebook. You can launch Jupyter Notebook by executing the following command in your terminal:
jupyter notebook chatbot.ipynb

## Usage
The chatbot starts with a greeting and awaits user input. Type your messages to interact with the bot. The conversation continues until you type 'bye'.

## Code Overview
Here's a brief overview of the code in chatbot.ipynb:
- import io
import random
import string  # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

- import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages
nltk.download('punkt')
nltk.download('wordnet')

- f = open('info.txt', 'r', errors='ignore')
raw = f.read()
raw = raw.lower()  # converts to lowercase

- sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words

- lemmer = nltk.stem.WordNetLemmatizer()
- def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

- def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

- GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

- def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

-- flag = True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while flag == True:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("ROBO: You are welcome..")
        else:
            if greeting(user_response) is not None:
                print("ROBO: " + greeting(user_response))
            else:
                print("ROBO: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("ROBO: Bye, feel free to contact me again!")

## Future Enhancements
- **Advanced NLP Techniques: Integrate machine learning models for more sophisticated intent recognition.
- **GUI: Develop a graphical user interface for a more user-friendly experience.
- **Custom Training Data: Allow the bot to be trained on custom datasets to enhance its response accuracy.
##Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the chatbot.
##License
This project is licensed under the MIT License. See the LICENSE file for details.
This updated `README.md` file reflects the changes in the code and specifies that the bot is now implemented in a Jupyter notebook (`chatbot.ipynb`).


