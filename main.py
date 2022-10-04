import nltk
from nltk.stem.lancaster import LancasterStemmer
import pygame
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
from tkinter import *

# Loading our JSON Data
with open("objects.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    # Extracting Data
    words = []
    labels = []
    docs_x = []
    docs_y = []
    '''loop through our JSON data and extract the data we want.
    For each pattern we will turn it into a list of words using nltk.word_tokenizer
    rather than having them as strings.
    We will then add each pattern into our docs_x list and its associated tag into the docs_y list'''
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # Tokenize aka get all words in our pattern with nltk
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    # Stemming
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    # Remove duplicates
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat(input):
    results = model.predict([bag_of_words(input, words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    if results[results_index] > 0.75:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        msg = random.choice(responses)
    else:
        msg = random.choice(data['intents'][0]['responses'])

    return msg


# Color,fonts etc
BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"
FONT = "Garamond 14"
FONT_BOLD = "Garamond 13 bold"


# GUI
class ChatApplication:
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()
        self.init_message("Νarrator", "You are an ordinary peasant who live in the city of Novigrand"
                                " and decided to spend your afternoon in your local inn. "
                                "There you notice a white haired man with two swords on his back that "
                                "seems highly unusual. "
                                "You asked the innkeeper about it and he replied that he is a witcher. "
                                " Not knowing what this mean , you decide to start a conversation with stranger. ")
        self.init_message("Νarrator", "Say hi to begin.")

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Conversation")
        self.window.resizable(width=True, height=True)
        self.window.configure(width=550, height=550, bg=BG_COLOR)
        # head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Speak...", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # message entry box
        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        # say button
        say_button = Button(bottom_label, text="Say", font=FONT_BOLD, width=20, bg=BG_GRAY,
                            command=lambda: self._on_enter_pressed(None))
        say_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")

    def _insert_message(self, msg, sender):
        if not msg:
            return

        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        msg2 = f"{'Witcher'}: {chat(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)

    def init_message(self, talker, msg):
        msg2 = f"{talker}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)


# Music set up
pygame.mixer.init()
crash_sound = pygame.mixer.Sound("ambience.mp3")
crash_sound.set_volume(0.2)
crash_sound.play(loops=-1)

# Start Application
app = ChatApplication()
app.run()

