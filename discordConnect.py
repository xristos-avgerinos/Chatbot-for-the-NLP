import discord
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle


# Loading our JSON Data
with open(
        "../../../Users/xrist/Desktop/ΕΠΕΞΕΡΓΑΣΙΑ ΦΥΣΙΚΗΣ ΓΛΩΣΣΑΣ/ΕΡΓΑΣΙΑ 2022/ChatBotProject/pythonProject/objects.json") as file:
    data = json.load(file)

try:
    with open(
            "../../../Users/xrist/Desktop/ΕΠΕΞΕΡΓΑΣΙΑ ΦΥΣΙΚΗΣ ΓΛΩΣΣΑΣ/ΕΡΓΑΣΙΑ 2022/ChatBotProject/pythonProject/data.pickle", "rb") as f:
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

    with open(
            "../../../Users/xrist/Desktop/ΕΠΕΞΕΡΓΑΣΙΑ ΦΥΣΙΚΗΣ ΓΛΩΣΣΑΣ/ΕΡΓΑΣΙΑ 2022/ChatBotProject/pythonProject/data.pickle", "wb") as f:
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


class MyClient(discord.Client):
    async def on_ready(self):
        print('Logged in as')
        print(self.user.name)
        print(self.user.id)
        print('------')

    async def on_message(self, message):
        # we do not want the bot to reply to itself
        if message.author.id == self.user.id:
            return

        else:
            inp = message.content
            result = model.predict([bag_of_words(inp, words)])[0]
            result_index = numpy.argmax(result)
            tag = labels[result_index]

            if result[result_index] > 0.7:
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']

                bot_response = random.choice(responses)
                await message.channel.send(bot_response.format(message))
            else:
                await message.channel.send("I didnt get that. Can you explain or try again.".format(message))

client = MyClient()
client.run('Token')
