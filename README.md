# Language Detection Fellowship.ai challenge

This repository contains the code in response to the [Fellowship.ai challenge] (https://fellowship.ai/challenge/) for language detection. It uses the nltk libray and is written in Python.

I experimented with different ways to generate features to use for training. I started by looking at the most common words and letters for each language and trying multiple hyperparameter combinations. I then introduced the top 10, three letter endings for each language and the most used common letters.   At the end the latter gave the best results (see table at bottom of page).

The [Detect Language Final] (http://Detect%20Language%20Final.ipynb) jupyter notebook is a good summary that shows the steps and the result.

Also of note, the [final_hyperparameter_tuning_using_percentage_of_words.py] (final_hyperparameter_tuning_using_percentage_of_words.py) contains the hyperparameter tuning code.


## Getting Started

The corpora given on the challenge is the [European Parliament Proceedings Parallel Corpus] (http://www.statmt.org/europarl/). This includes 21 different languages. 

File Structure:

* txt: contains the unzipped corpora. There are 21 folders each contain the documents for each language (bg, es, et, fi, fr, etc.). **Note:** Do to its size, this repository does not contain the actual files. They must be downloaded manually and install in this folder if they wished to be processed.

* pickles: contains all documents from corpora, all words found and all letters used on each language. These files are serialized(pickle) from their original format and used to train the classifiers. Instead of reading the raw files from txt directory every time, I used this for faster training.

* models: contains deployed models once trained. This is used on the last step to evaluate accuracy of different models against the entire test file.

* util: is a Python package with functions used in this repository. c

#This project was divided into six steps:

**Step 1**

Process all files on the corpora, and extract the text from each one of the files by removing any HTML tag. Save as pickles files for easier processing when hypertuning. The Create Pickles from Corporate reads the /txt directory and creates the processed serialized version in /pickles. 

**Step 2**
Extract most common words up to a defined percentage per each language. This is used as part of the featureset used for training

**Step 3**
Divide date set into training and test sets

**Step 4**
Build and score the model. Build with training set, score with test set.

**Step 5**
Deploy model against the europarl.test test file and show score of model.

## Results

After tuning three hyperparameters (documents to extract, percentage of top words, and number of common letters) the best result against the europarl.test was of **94.75**.  The following table shows the different scores obtained based on the different parameters.

FINAL ACCURACY|Documents to extract|Top percentage of words to use|Number of common Letters to use|Classifier accuracy on test set|Europarl test accuracy|Model accuracy against europarl.test|All documents processed|Number of features used|Training set|Test set|Classifier accuracy on test set|Europarl test accuracy|Model accuracy against europarl.test
--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
94.75|5000|0|10|96.59|92.66|94.75|105000|0|84000|21000|96.59|92.66|94.75
94.73|5000|0|7|96.87|93.42|94.73|105000|0|84000|21000|96.87|93.42|94.73
94.66|5000|0|5|96.54|93.33|94.66|105000|0|84000|21000|96.54|93.33|94.66
94.6|5000|0|15|96.9|92.09|94.6|105000|0|84000|21000|96.9|92.09|94.6
94.52|10000|5|5|97.03|50|94.52|187071|129|149656|37415|97.03|50|94.52
93.56|10000|10|10|96.44|55|93.56|187071|415|149656|37415|96.44|55|93.56
93.51|5000|5|5|96.34|94.76|93.51|110000|121|88000|22000|96.34|94.76|93.51
92.63|5000|5|5|96.8|66|92.63|105000|129|84000|21000|96.8|66|92.63
92.6|5000|10|5|95.99|94|92.6|110000|399|88000|22000|95.99|94|92.6
91.91|5000|10|10|96.41|73|91.91|105000|415|84000|21000|96.41|73|91.91
91.44|10000|15|5|94.93|87|91.44|187071|856|149656|37415|94.93|87|91.44
91.39|3000|5|5|96.06|93.09|91.39|66000|119|52800|13200|96.06|93.09|91.39
90.15|3000|10|5|94.93|92.14|90.15|66000|389|52800|13200|94.93|92.14|90.15
89.77|5000|15|5|94.25|90.33|89.77|110000|827|88000|22000|94.25|90.33|89.77
89.05|5000|15|5|94.48|111|89.05|105000|856|84000|21000|94.48|111|89.05
87.5|10000|20|6|92.35|119|87.5|187071|1491|149656|37415|92.35|119|87.5
85.88|3000|15|5|93.28|88.33|85.88|66000|805|52800|13200|93.28|88.33|85.88
85.87|1000|5|5|94.84|89.19|85.87|22000|119|17600|4400|94.84|89.19|85.87
84.65|3000|20|5|90.65|86.95|84.65|66000|1423|52800|13200|90.65|86.95|84.65
84.41|1000|10|5|93.38|86.8|84.41|22000|389|17600|4400|93.38|86.8|84.41
83.98|5000|20|6|91.18|164|83.98|105000|1491|84000|21000|91.18|164|83.98
80.49|1000|0|7|96.19|75.23|80.49|21000|0|16800|4200|96.19|75.23|80.49
80.22|1000|0|15|96.35|73.9|80.22|21000|0|16800|4200|96.35|73.9|80.22
80.2|1000|0|5|96.42|74.57|80.2|21000|0|16800|4200|96.42|74.57|80.2
80.11|1000|0|10|95.78|73.33|80.11|21000|0|16800|4200|95.78|73.33|80.11
78.33|1000|15|5|90.7|82.47|78.33|22000|805|17600|4400|90.7|82.47|78.33

### Prerequisites

Download the European Parliament Proceedings Parallel Corpus and unzip into the /txt folder.
Generate pickle files by rooning the Create Pickles from Corpora.

## Deployment

Once the models have been generated, you can run the 'Deploy - Loop through classifiers for testing of Europarl.test file' notebook to test against the europarl.test file.
This is only if you are to generate models by tweaking the different hyperparameters.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


