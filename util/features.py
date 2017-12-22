# Feature extraction and creation functions
from collections import Counter
import os
import string
import pickle
import nltk
import numpy
import pickle
from multiprocessing import Pool
from util.general import *

# Loop through directory and extract text
# directory is the path to directory to process
def get_text_from_directory(directory):
    language_label = directory.split("/")[-1]
    documents = []
    counter = 0
    # keep a count on unique words seen on documents
    word_counter = Counter()
    alphabet_counter = Counter()
    alphabet = set()
    # Collect a minimun of 5,000 words per directory(language)
    for filename in os.listdir(directory):
        # try:
            text_file = open(directory + "/" + filename, "r").read()
            text = extract_text_only(text_file)
            # Tokenize words and remove punctuation
            tokenized_text = tokenize_removepuncuation(text)
            # add to dict counter
            word_counter.update(tokenized_text)
            # get letters and add to alphabet
            [alphabet_counter.update(list(n)) for n in tokenized_text]
            documents.append((tokenized_text, language_label))
            counter = counter + 1
    #     except:
    #         print(directory + " - Issue with filename:" + filename + " Ignoring.")
    # # documents contains list of all documents in directory with the following format
    #     [(['worda1','worda2,'worda3'],'LANG-A'),['wordb1','wordb2,'wordb3'],'LANG-A'),...]
    # word_counter - set of all words found in document
    # alphabet_counter - set of all letter found in document
    return documents, word_counter, alphabet_counter


def extract_data_from_corpora(corpora_directory, number_of_words, number_of_letter, save_pickles):
    all_documents = []
    most_common_words = {}
    most_common_letters = {}

    # Loop through all directories contain corpora with all languages
    # directory will be the folder containing documents on that language
    for directory in os.listdir(corpora_directory):
        # full_path contains
        full_path = corpora_directory + directory
        if (os.path.isdir(full_path)):
            print("About to process directory " + directory)
            # process directory, text contains documents list with rows (['worda1','worda2,'worda3'],'LANG-A')
            # word_counter contains count of all words seen
            text, word_counter, alphabet = get_text_from_directory(full_path)
            print("Number of words for this language:" + str(len(word_counter)))
            # Keep only letters that are not common ascii letters
            for letter in list(alphabet):
                if (letter in list(string.ascii_letters) or letter in list(string.digits)):
                    del alphabet[letter]
            # Keep track of most common words per language to use on feature set
            most_common_words[directory] = word_counter.most_common(number_of_words)
            most_common_letters[directory] = alphabet.most_common(number_of_letter)
            if (save_pickles):
                # Save to pickle so it can be read without having to process again
                pickle_out = open("pickles/word_counter_" + directory + ".pickle", "wb")
                pickle.dump(word_counter, pickle_out)
                pickle_out.close()
                pickle_out = open("pickles/alphabet_" + directory + ".pickle", "wb")
                pickle.dump(alphabet, pickle_out)
                pickle_out.close()
                pickle_out = open("pickles/documents_" + directory + ".pickle", "wb")
                pickle.dump(text, pickle_out)
                pickle_out.close()

            all_documents = all_documents + text

    if (save_pickles):
        # Save all_documents to pickle for later use
        pickle_out = open("pickles/all_documents.pickle", "wb")
        pickle.dump(all_documents, pickle_out)
        pickle_out.close()
    return all_documents, most_common_words, most_common_letters


# this function receives a document and creates a feature based list
# Input:
#   [(['worda1','worda2,'worda3'],'LANG-A'),
#     ['wordb1','wordb2,'wordb3'],'LANG-B'),...]
# Ouput:
# [('αποφασιστικής': False,
#   'Περιφερειακής': True,
#   'pontot': False,...),'ru'),...]

def document_features(document, word_features, letter_features):
    #normalize words
    document = [w.lower() for w in document]
    document_words = set(document)
    features = {}
    # Add word that are part of common words
    for word in word_features:
        features[word] = (word in document_words)
    # Add letters that are part of common letters
    # get letters and add to alphabet
    document_alphabet = set()
    [document_alphabet.update(list(n)) for n in document_words]
    # document_alphabet now contains all letters on this document
    # Keep only letters that are not common ascii letters
    for letter in list(document_alphabet):
        if (letter in list(string.ascii_letters) or letter in list(string.digits)):
            document_alphabet.remove(letter)
    # Add to feature if exist
    for l in letter_features:
        features[l] = (l in document_alphabet)
    return features


def extract_data_from_corpora_pickles(pickle_directory, number_of_documents, number_of_words, number_of_letters):
    all_documents = []
    most_common_words = {}
    most_common_letters = {}

    # Read data from pickle files
    for filename in os.listdir(pickle_directory):
        language = (filename.split('_')[-1]).split('.')[0]
        if ('word_counter' in filename):
            all_words_for_language = pickle.load(open(pickle_directory + "/" + filename, "rb"))
            ##eliminate any words that intersect with another language
            for lang in most_common_words:
                all_words_for_language, most_common_words[lang] = remove_common_elements(all_words_for_language,
                                                                                         most_common_words[lang])

            most_common_words[language] = all_words_for_language

        elif ('alphabet' in filename):
            all_letters_for_language = pickle.load(open(pickle_directory + "/" + filename, "rb"))
            ##eliminate any words that intersect with another language
            for lang in most_common_letters:
                all_letters_for_language, most_common_letters[lang] = remove_common_elements(all_letters_for_language,
                                                                                             most_common_letters[lang])

            most_common_letters[language] = all_letters_for_language
        elif ('documents' in filename):
            documents = pickle.load(open(pickle_directory + "/" + filename, "rb"))
            doc_limit = number_of_documents if (number_of_documents >= 0) else len(all_documents)
            all_documents += documents[:doc_limit]

    # return only up to the amount required
    for lang in most_common_words:
        word_limit = number_of_words if (number_of_words >= 0) else len(most_common_words[lang])
        most_common_words[lang] = most_common_words[lang].most_common(word_limit)

    for lang in most_common_letters:
        letter_limit = number_of_letters if (number_of_letters >= 0) else len(most_common_letters[lang])
        most_common_letters[lang] = most_common_letters[lang].most_common(letter_limit)

    return all_documents, most_common_words, most_common_letters


def extract_documents_from_corpora_pickles(pickle_directory, number_of_documents):
    all_documents = []

    # Read data from pickle files
    for filename in os.listdir(pickle_directory):
        language = (filename.split('_')[-1]).split('.')[0]
        if ('documents' in filename):
            documents = pickle.load(open(pickle_directory + "/" + filename, "rb"))
            doc_limit = number_of_documents if (number_of_documents >= 0) else len(all_documents)
            all_documents += documents[:doc_limit]

    return all_documents

def load_document_pickle(arguments):
    pickle_file = arguments[0]
    number_of_documents = arguments[1]
    documents = pickle.load(open(pickle_file, "rb"))
    doc_limit = number_of_documents if (number_of_documents >= 0) else len(documents)
    return documents[:doc_limit]

def extract_documents_from_corpora_pickles_parallel(pickle_directory, number_of_documents):
    all_documents = []
    dataset = []
    # Read data from pickle files
    for filename in os.listdir(pickle_directory):
        if ('documents' in filename):
            dataset.append((pickle_directory+"/"+filename,number_of_documents))

    # Run this with a pool of 5 agents having a chunksize of 3 until finished
    with Pool() as pool:
        all_documents = pool.map(load_document_pickle, dataset)

    return all_documents

def extract_wordsletters_from_corpora_pickles(pickle_directory, number_of_words, number_of_letters):
    most_common_words = {}
    most_common_letters = {}

    # Read data from pickle files
    for filename in os.listdir(pickle_directory):
        language = (filename.split('_')[-1]).split('.')[0]
        if ('word_counter' in filename):
            all_words_for_language = pickle.load(open(pickle_directory + "/" + filename, "rb"))
            ##eliminate any words that intersect with another language
            for lang in most_common_words:
                all_words_for_language, most_common_words[lang] = remove_common_elements(all_words_for_language,
                                                                                         most_common_words[lang])

            most_common_words[language] = all_words_for_language

        elif ('alphabet' in filename):
            all_letters_for_language = pickle.load(open(pickle_directory + "/" + filename, "rb"))
            ##eliminate any words that intersect with another language
            for lang in most_common_letters:
                all_letters_for_language, most_common_letters[lang] = remove_common_elements(all_letters_for_language,
                                                                                             most_common_letters[lang])

            most_common_letters[language] = all_letters_for_language

    # return only up to the amount required
    for lang in most_common_words:
        word_limit = number_of_words if (number_of_words >= 0) else len(most_common_words[lang])
        most_common_words[lang] = most_common_words[lang].most_common(word_limit)

    for lang in most_common_letters:
        letter_limit = number_of_letters if (number_of_letters >= 0) else len(most_common_letters[lang])
        most_common_letters[lang] = most_common_letters[lang].most_common(letter_limit)

    return most_common_words, most_common_letters