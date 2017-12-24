from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
import time
import pandas as pd
import collections

#General utility functions
def print_elapsed_time(start):
    end = time.time()
    elapsed = end - start
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    return ("%d:%02d:%02d" % (h, m, s))


# Remove any <tags> within text
def extract_text_only(text):
    soup = BeautifulSoup(text, "lxml")
    #    soup = BeautifulSoup(text,"html5lib")
    return soup.get_text()


def tokenize_removepuncuation(text):
    # words only
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(text)


def most_common_words_letter(most_common_words, most_common_letters):
    # Create word_features, a list of most common words on all languauges
    # this will be used on feature set fed to classifier
    word_features = set()
    letter_features = set()

    for k, v in most_common_words.items():
        for word in v:
            word_features.add(word)

    # Create letter_features, a list of most common letters on all languages
    for k, v in most_common_letters.items():
        for letter in v:
            letter_features.add(letter)

    return word_features, letter_features

def most_common_wordsonly(most_common_words):
    # Create word_features, a list of most common words on all languauges
    # this will be used on feature set fed to classifier
    word_features = set()

    for k, v in most_common_words.items():
        for word in v:
            word_features.add(word)

    return word_features


# Takes two Counter objects, removes common elements
def remove_common_elements(counter1, counter2):
    elements_intersect = (counter1 & counter2).most_common()
    for n in elements_intersect:
        key = n[0]
        del counter1[key]
        del counter2[key]
    return counter1, counter2
