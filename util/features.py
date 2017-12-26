# Feature extraction and creation functions
# -*- coding: utf-8 -*-
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

def get_most_frequent_letters(tokenized_text, number_of_letters):
    # get letters and add to alphabet
    alphabet_counter = Counter()
    [alphabet_counter.update(list(n)) for n in tokenized_text]
    most_common = alphabet_counter.most_common(number_of_letters)
    most_common_str = ""
    for l in most_common:
        most_common_str += l[0]
    most_common_str = most_common_str[:number_of_letters]
    return most_common_str

def get_most_frequent_ending_of_words(tokenized_text,number_of_characters):
    ending_counter = Counter()

    for n in tokenized_text:
        ending_counter[n[-number_of_characters:]] += 1

    most_common = ending_counter.most_common(10)
    most_common_list = []
    for l in most_common:
        most_common_list.append(l)

    return most_common_list

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

def document_features_fromwords(document, word_features, number_of_common_letters):
    #normalize words
    document = [w.lower() for w in document]
    document_words = set(document)
    features = {}
    #Set most common letters as a feature
    most_common_letters = get_most_frequent_letters(document_words, number_of_common_letters)
    for n in range(2,number_of_common_letters+1):
        features['common_letters_'+str(n)] = most_common_letters[:n]
    #Set last three letters of words  also as a feature
    most_common_ending = get_most_frequent_ending_of_words(document_words, 3)
    counter = 0
    for n in most_common_ending:
        counter += 1
        features['common_ending_top_' + str(counter)] = n[0]
    # Add word that are part of common words
    for word in word_features:
        features[word] = (word in document_words)
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
                all_words_for_language, most_common_words[lang] = remove_common_elements(all_words_for_language,most_common_words[lang])

            most_common_words[language] = all_words_for_language

        elif ('alphabet' in filename):
            all_letters_for_language = pickle.load(open(pickle_directory + "/" + filename, "rb"))

            ##eliminate any words that intersect with another language
            for lang in most_common_letters:
                all_letters_for_language, most_common_letters[lang] = remove_common_elements(all_letters_for_language,most_common_letters[lang])

            most_common_letters[language] = all_letters_for_language


    # return only up to the amount required
    for lang in most_common_words:
        word_limit = number_of_words if (number_of_words >= 0) else len(most_common_words[lang])
        most_common_words[lang] = most_common_words[lang].most_common(word_limit)

    for lang in most_common_letters:
        letter_limit = number_of_letters if (number_of_letters >= 0) else len(most_common_letters[lang])
        most_common_letters[lang] = most_common_letters[lang].most_common(letter_limit)

    return most_common_words, most_common_letters

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

#Get a counter object and lower all words
def convert_to_lower(counter_list):
    # Save to file for review - words
    counter = Counter()
    for k, v in counter_list.most_common():
        counter.update({k.lower():v})
    return counter

#Get Counter object and return only those that add up to
#percentage of total.  For instance
#select_elements_up_to_percentage(Counter({'a':10,'b':10,'c':5,'d':5}), 30)
#returns Counter{'a':10})
def select_elements_up_to_percentage(counter_obj, upto_percentage):
    #get sum to calculate percentages
    counter = Counter()
    # results = []
    totalsum = 0
    for k, v in counter_obj.items():
        totalsum += v

    percentage_sum = 0
    #Calculate percentage of each instance
    for k, v in counter_obj.most_common():
        percentage = (v/totalsum)*100
        percentage_sum += percentage
        # results.append((k,v,percentage, percentage_sum))
        if(percentage_sum>upto_percentage):
            break
        counter[k]=v

    return counter

def extract_most_common_letters(pickle_directory,number_of_letters):
    most_common_letters = {}

    for filename in os.listdir(pickle_directory):
        language = (filename.split('_')[-1]).split('.')[0]

        if ('alphabet' in filename):
            all_letters_for_language = pickle.load(open(pickle_directory + "/" + filename, "rb"))
            all_letters_for_language = convert_to_lower(all_letters_for_language)
            most_common_letters[language] = all_letters_for_language

    # Save to file for review - words
    for lang, letter_counter in most_common_letters.items():
        file_out = open('stats/' + lang + "_freq_letters.csv", "w")
        file_out.write('word,count\n')
        for k, v in letter_counter.most_common():
            file_out.write(k + "," + str(v) + "\n")
        file_out.close()

    return most_common_letters


def extract_wordsletters_from_corpora_pickles_save_stats_files(pickle_directory, number_of_words, number_of_letters):
    most_common_words = {}
    most_common_letters = {}
    # Read data from pickle files
    for filename in os.listdir(pickle_directory):
        language = (filename.split('_')[-1]).split('.')[0]
        if ('word_counter' in filename):
            all_words_for_language = pickle.load(open(pickle_directory + "/" + filename, "rb"))
            all_words_for_language = convert_to_lower(all_words_for_language)
            most_common_words[language] = all_words_for_language

        elif ('alphabet' in filename):
            all_letters_for_language = pickle.load(open(pickle_directory + "/" + filename, "rb"))
            all_letters_for_language = convert_to_lower(all_letters_for_language)
            most_common_letters[language] = all_letters_for_language

    #Remove words/letters that are found on multiple languages
    ##eliminate any words that intersect with another language
    most_common_keys = most_common_words.keys()
    for lang in most_common_keys:
        rest_of_languages = list(most_common_keys)
        rest_of_languages.remove(lang)
        len_before = len(most_common_words[lang])
        for lang2 in rest_of_languages:
            most_common_words[lang], most_common_words[lang2] = remove_common_elements(most_common_words[lang],most_common_words[lang2])
        len_after = len(most_common_words[lang])
        diff = len_before - len_after
        print(lang +" before:"+str(len_before)+" after:" + str(len_after)+" lost:"+str(diff))

    #Save to file for review - words
    for lang, word_counter in most_common_words.items():
        file_out = open('stats/' + lang + "_words.csv", "w")
        file_out.write('word,count\n')
        for k,v in word_counter.most_common():
            file_out.write(k+","+str(v)+"\n")
        file_out.close()

    ##eliminate any words that intersect with another language
    most_common_letter_keys = most_common_letters.keys()
    for lang in most_common_letter_keys:
        rest_of_languages = list(most_common_letter_keys)
        rest_of_languages.remove(lang)
        len_before = len(most_common_letters[lang])
        for lang2 in rest_of_languages:
            most_common_letters[lang], most_common_letters[lang2] = remove_common_elements(most_common_letters[lang],
                                                                                           most_common_letters[lang2])
        len_after = len(most_common_letters[lang])
        diff = len_before - len_after
        print(lang + " letters before:" + str(len_before) + " after:" + str(len_after) + " lost:" + str(diff))

    #Save to file for review - words
    for lang, letter_counter in most_common_letters.items():
        file_out = open('stats/' + lang + "_letters.csv", "w")
        file_out.write('word,count\n')
        for k,v in letter_counter.most_common():
            file_out.write(k+","+str(v)+"\n")
        file_out.close()


    # # return only up to the amount required
    # for lang in most_common_words:
    #     word_limit = number_of_words if (number_of_words >= 0) else len(most_common_words[lang])
    #     most_common_words[lang] = most_common_words[lang].most_common(word_limit)
    #
    # for lang in most_common_letters:
    #     letter_limit = number_of_letters if (number_of_letters >= 0) else len(most_common_letters[lang])
    #     most_common_letters[lang] = most_common_letters[lang].most_common(letter_limit)

    return most_common_words, most_common_letters

def extract_words_from_corpora_pickles_upto_per(pickle_directory, upto_percentage):
    most_common_words = {}
    # Read data from pickle files
    for filename in os.listdir(pickle_directory):
        language = (filename.split('_')[-1]).split('.')[0]
        if ('word_counter' in filename):
            all_words_for_language = pickle.load(open(pickle_directory + "/" + filename, "rb"))
            all_words_for_language = convert_to_lower(all_words_for_language)
            most_common_words[language] = all_words_for_language

    #Remove words that are found on multiple languages
    ##eliminate any words that intersect with another language
    most_common_keys = most_common_words.keys()
    for lang in most_common_keys:
        rest_of_languages = list(most_common_keys)
        rest_of_languages.remove(lang)
        len_before = len(most_common_words[lang])
        for lang2 in rest_of_languages:
            most_common_words[lang], most_common_words[lang2] = remove_common_elements(most_common_words[lang],most_common_words[lang2])
        len_after = len(most_common_words[lang])
        diff = len_before - len_after
        # print(lang +" before:"+str(len_before)+" after:" + str(len_after)+" lost:"+str(diff))

    #Save to file for review - words
    # for lang, word_counter in most_common_words.items():
    #     file_out = open('stats/' + lang + "_words.csv", "w")
    #     file_out.write('word,count\n')
    #     for k,v in word_counter.most_common():
    #         file_out.write(k+","+str(v)+"\n")
    #     file_out.close()

    # return only up to the percentage required
    for lang in most_common_words:
        most_common_words[lang] = select_elements_up_to_percentage(most_common_words[lang], upto_percentage)

    return most_common_words