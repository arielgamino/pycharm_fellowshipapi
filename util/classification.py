# Classification functions

from util.features import *
import time

def classify_document(document, classifier, word_features, letter_features):
    return classifier.classify(document_features(tokenize_removepuncuation(document), word_features, letter_features))

def classify_document_words(document, classifier, word_features):
    return classifier.classify(document_features_fromwords(tokenize_removepuncuation(document), word_features))


def test_europarltest_file(eurofile, resultsfile, everyother, classifier, word_features, letter_features):
    # Read test file and classify each sentence in file
    positive_ctr = 0
    negative_ctr = 0
    total_ctr = 0
    # save results to file for processing
    fileout = open(resultsfile, 'w')
    # columns
    fileout.write('predicted, language given, correctly classified?\n')
    fileout.write('predicted, language given, correctly classified?\n')

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_correctly = open('correctly_classified_'+timestamp, 'w')
    file_incorrectly = open('incorrectly_classified_' + timestamp, 'w')

    language_counter = Counter()

    processed_counter = 0
    with open(eurofile, 'r') as f:
        for line in f:
            processed_counter += 1
            if (processed_counter % everyother == 0):
                total_ctr += 1
                # language is first two letters in line
                language = line[:2]
                # sentence is rest, clean up spaces
                sentence = line[2:].strip()
                # Detect language based on model
                language_detected = classify_document(sentence, classifier, word_features, letter_features)
                correctly_classified = language_detected == language
                # tally correct and incorrect
                if (correctly_classified):
                    # correctly classified
                    positive_ctr += 1
                    file_correctly.write(language_detected + ',' + language + ',' + str(correctly_classified)+sentence+'\n')
                    language_counter[language+'_correct'] += 1
                else:
                    # incorrectly classified
                    negative_ctr += 1
                    file_incorrectly.write(language_detected + ',' + language + ',' + str(correctly_classified) + sentence + '\n')
                    language_counter[language + '_incorrect'] += 1

                fileout.write(language_detected + ',' + language + ',' + str(correctly_classified) + '\n')

    fileout.close()
    file_correctly.close()
    file_incorrectly.close()

def test_europarltest_file_words(eurofile, resultsfile, everyother, classifier, word_features):
    # Read test file and classify each sentence in file
    positive_ctr = 0
    negative_ctr = 0
    total_ctr = 0
    # save results to file for processing
    fileout = open(resultsfile, 'w')
    # columns
    fileout.write('predicted, language given, correctly classified?\n')

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_correctly = open('correctly_classified_' + timestamp, 'w')
    file_incorrectly = open('incorrectly_classified_' + timestamp, 'w')

    language_counter = Counter()

    processed_counter = 0
    with open(eurofile, 'r') as f:
        for line in f:
            processed_counter += 1
            if (processed_counter % everyother == 0):
                total_ctr += 1
                # language is first two letters in line
                language = line[:2]
                # sentence is rest, clean up spaces
                sentence = line[2:].strip()
                # Detect language based on model
                language_detected = classify_document_words(sentence, classifier, word_features)
                correctly_classified = language_detected == language
                # tally correct and incorrect
                if (correctly_classified):
                    # correctly classified
                    positive_ctr += 1
                    file_correctly.write(language_detected + ',' + language + ',' + str(correctly_classified) + sentence + '\n')
                    language_counter[language + '_correct'] += 1
                else:
                    # incorrectly classified
                    negative_ctr += 1
                    file_incorrectly.write(language_detected + ',' + language + ',' + str(correctly_classified) + sentence + '\n')
                    language_counter[language + '_incorrect'] += 1

                fileout.write(language_detected + ',' + language + ',' + str(correctly_classified) + '\n')

    fileout.close()
    file_correctly.close()
    file_incorrectly.close()

    return total_ctr, positive_ctr, negative_ctr, language_counter