from util.general import *
from util.features import *
from util.classification import *

pickles_directory="pickles"
most_common_words, most_common_letters = extract_wordsletters_from_corpora_pickles(pickles_directory, 100,100)
