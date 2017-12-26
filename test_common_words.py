from util.general import *
from util.features import *
from util.classification import *

pickles_directory="pickles"
# most_common_words, most_common_letters = extract_wordsletters_from_corpora_pickles_save_stats_files(pickles_directory, 100,100)

# most_common_words = extract_words_from_corpora_pickles_upto_per(pickles_directory,50)
# for lang in most_common_words:
#     print(lang+":"+str(len(most_common_words[lang])))

#extract_most_common_letters(pickles_directory,100)
tokenized_text = ['xwwwyyyzz','Ariel','Gamiño','This', 'his','bliss','and', 'that','What', 'is', 'the','deal','Merry','arry','hat','l']
a = get_most_frequent_letters(tokenized_text, 10)
print(a)

most_common_ending = get_most_frequent_ending_of_words(tokenized_text,3)
print(most_common_ending)

features = {}
counter = 0
for n in most_common_ending:
    counter +=1
    features['common_ending_top_' + str(counter)] = n[0]

print(features)

# all_words_for_language = pickle.load(open("pickles/word_counter_et.pickle", "rb"))
# all_words_for_language = convert_to_lower(all_words_for_language)
# r = select_elements_up_to_percentage(all_words_for_language,50)
#
# index = 0
# print(len(r))
# for n in r:
#     index += 1
#     print(n)
#
# print(index)



