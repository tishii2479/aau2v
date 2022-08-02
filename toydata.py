from random import randint, choice, shuffle
import torch


def create_toydata(num_topic: int, data_size: int):
    documents = []
    words = []
    key_words = [[] for _ in range(num_topic)]

    for _ in range(1, 201):
        s = ''
        for _ in range(10):
            s += chr(ord('a') + randint(0, 26))
        words.append(s)

    for i in range(num_topic):
        for j in range(1, 11):
            s = chr(ord('a') + i) * j
            key_words[i].append(s)

    for i in range(num_topic):
        for _ in range(data_size):
            doc = []
            for _ in range(randint(30, 50)):
                doc.append(choice(key_words[i]))
            for _ in range(randint(150, 200)):
                doc.append(choice(words))
            shuffle(doc)
            documents.append(doc)

    for i in range(num_topic):
        words += key_words[i]

    word_embedding = torch.eye(len(words))

    return (documents, None), (words, word_embedding)


def create_labeled_toydata(num_topic: int):
    (documents, _), (words, word_embedding) = create_toydata(num_topic)
    labels = []
    for i in range(num_topic):
        for _ in range(5):
            labels.append(i)
    return (documents, labels), (words, word_embedding)
