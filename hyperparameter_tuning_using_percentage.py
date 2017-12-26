from util.general import *
from util.features import *
from util.classification import *

pickles_directory = "pickles"

start1 = time.time()

stats_out = []

europarl_testfile = "europarl.test"

#for number_of_documents in [500, 1000, 3000, 5000]:
for number_of_documents in [9000]:
    # -------------Step 1-------------
    # ----READ FROM PICKLE FILES (Pre-read)----
    # Get data to create features from corpora
    pickles_directory = "pickles"
    print("-------------------------------------------------")
    print("Number of documents to extract: " + str(number_of_documents))

    # Part 1 - Extract documents from corpora
    start = time.time()
    all_documents = extract_documents_from_corpora_pickles(pickles_directory, number_of_documents)
    print("Elapsed time reading all documents:" + print_elapsed_time(start))
    print("Total Documents:" + str(len(all_documents)))

    stats = collections.OrderedDict()
#    for words_letters in [(2000,0),(2000,100),(3000,40)]:
    for upto_percentage in [20]:
        print("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")
        print("Percentage of common words to use:" + str(upto_percentage))
        # Part 2 - get common words, letters
        start = time.time()
        most_common_words = extract_words_from_corpora_pickles_upto_per(pickles_directory, upto_percentage)
        elapsed_reading_wl = print_elapsed_time(start)
        print("Elapsed time reading all words, letters:" + elapsed_reading_wl)
        print("all_documents:" + str(len(all_documents)))
        print("most_common_words:" + str(len(most_common_words)))
        for k, v in most_common_words.items():
            print(k+" words:"+str(len(v)))

        # -------------Step 2-------------
        # Create featureset to be used for training
        # this is a list of documents with features and label
        start = time.time()

        # create word_features
        word_features = most_common_wordsonly(most_common_words)
        print("words_features:" + str(len(word_features)))

        # create featureset
        featuresets = [(document_features_fromwords(d, word_features), c) for (d, c) in all_documents]
        elapsed_feature_creation = print_elapsed_time(start)
        print("Elapsed time featureset creation:" + elapsed_feature_creation)
        print("featuresets:" + str(len(featuresets)))
        # -------------Step 3-------------
        # Split train, test for model classification and scoring
        numpy.random.shuffle(featuresets)
        # calculate how many items to slice by (95% train, 5% test)
        slice_by = int((80 * len(featuresets)) / 100)
        train_set, test_set = featuresets[:slice_by], featuresets[slice_by:]
        print("Train set:" + str(len(train_set)))
        print("Test set:" + str(len(test_set)))

        # -------------Step 4-------------
        # Build the Model
        start = time.time()
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        elapsed_training = print_elapsed_time(start)
        print("Elapsed time for training:" + elapsed_training)
        start = time.time()
        accuracy = nltk.classify.accuracy(classifier, test_set)
        print("Classifier Accuracy:" + str(accuracy))
        elapsed_accuracy = print_elapsed_time(start)
        print("Elapsed time for accuracy testing:" + elapsed_accuracy)

        # -------------Step 5-------------
        # Test against europarl_test file

        results_outfile = "europarl_test_classified_attempt_" + str(number_of_documents) + "_" + str(upto_percentage) + ".csv"
        everyother = 20
        start = time.time()
        total_ctr, positive_ctr, negative_ctr, language_counter = test_europarltest_file_words(europarl_testfile, results_outfile, everyother,classifier, word_features)
        # results
        for k,v in language_counter.items():
            print("       "+k+":"+str(v))
        print("       Total attempted: " + str(total_ctr))
        print("  Classified correctly: " + str(positive_ctr))
        print("Classified incorrectly: " + str(negative_ctr))
        euro_accuracy = (positive_ctr / total_ctr) * 100
        print("  Europartest Accuracy: "+str(euro_accuracy))
        elapsed_accuracy = print_elapsed_time(start)
        print("Elapsed time for accuracy testing:" + elapsed_accuracy)  # Save classifier for deployment

        # Save to pickle so it can be tested later with europarl.test file
        pickle_out = open("models/classifier_" + str(number_of_documents) + "_" + str(upto_percentage) + ".pickle", "wb")
        pickle.dump(classifier, pickle_out)
        pickle_out.close()
        pickle_out = open("models/word_features" + str(number_of_documents) + "_" + str(upto_percentage) + ".pickle", "wb")
        pickle.dump(word_features, pickle_out)
        pickle_out.close()
        pickle_out = open("models/letter_features" + str(number_of_documents) + "_" + str(upto_percentage) + ".pickle", "wb")
        pickle.dump(word_features, pickle_out)
        pickle_out.close()

        stats = {'a_number_of_documents': number_of_documents,
                 'b_upto_percentage': upto_percentage,
                 'd_words_features:': len(word_features),
                 'f_feautureset': len(featuresets),
                 'g_train set': len(train_set),
                 'h_test set': len(test_set),
                 'j_elapsed_feature_creation': elapsed_feature_creation,
                 'k_elapsed_reading_wl': elapsed_reading_wl,
                 'l_elapsed_training': elapsed_training,
                 'm_elapsed_accuracy': elapsed_accuracy,
                 'n_accuracy': accuracy,
                 'elapsed_accuracy':elapsed_accuracy,
                 'euro_test_accuracy':euro_accuracy
         }

stats_out.append(stats)

#Save results to Excel
df = pd.DataFrame(stats_out)
writer = pd.ExcelWriter('hypertuning_results_wordsonly.xlsx')
df.to_excel(writer,'Results')
writer.save()

print("Completed:" + print_elapsed_time(start1))