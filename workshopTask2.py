# Useful links:


# Detection:
#   Precision = number of sentences correctly detected as ambiguous divided by the total number detected by the system as ambiguous
#   Recall = number of sentences correctly detected as ambiguous divided by the total number annotated by humans as ambiguous
# Resolution:
#   Precision = number of correctly resolved anaphors divided by the total number of anaphors attempted to be resolved
#   Recall = number of correctly resolved anaphors divided by the total number of unambiguous anaphors

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from csv import reader
import re
from sklearn import linear_model
from sklearn import *
from sklearn import metrics


def print_model_performace_for_test_doc(y_actual, y_pred):
    accuracy = metrics.accuracy_score(y_actual, y_pred)
    f1score = metrics.f1_score(y_actual, y_pred, average='macro')
    precision = metrics.precision_score(y_actual, y_pred, average='macro')
    recall = metrics.recall_score(y_actual, y_pred, average='macro')
    true_prediction = 0
    for i in range(len(y_actual)):
        if y_actual[i] == y_pred[i]:
            true_prediction += 1
    print("------------------------------------------------------------------")
    print("Confusion_matrix:\n", metrics.confusion_matrix(y_actual, y_pred))
    print('Accuracy: ', accuracy, "\t F1-Score: ", f1score, "\t Precision: ", precision, "\t Recall: ", recall)
    print("TOTAL: ", len(y_actual), " - TRUE PREDICTED: ", true_prediction)


def preprocessing(sentence):
    # 1. Word tokenization
    tokenized_words = word_tokenize(sentence)

    # 3. Lemmatization
    lemmatized_words = []
    lemmatizer = WordNetLemmatizer()
    for word in tokenized_words:
        lemmatized_words.append(lemmatizer.lemmatize(word))

    # 4. POS tagging
    tagged_words = nltk.pos_tag(lemmatized_words)

    return tagged_words


def featureExtraction(tags):
    print(tags)
    feature_vector = [0]

    # for candidate i and a pronoun j
    # word_distance: i-j aradasında kaç kelime var (integer)
    # gender_agreement: i-j aynı cinsiyette mi (her - she)
    # number_agreement: i-j'nin ikisi de çoğul ya da tekil mi
    # parallelism: i-j ikisi de subject ya da object mi
    
    antecedent_indexes = [] # The man is hungry, (NN*)
    anaphora_indexes = []   # so, he eats dinnner. (PRP*)
    index = 0
    for tag in tags:
        if tag[1].startswith("NN"):
            antecedent_indexes.append(index)

        if tag[1] == "PRP":
            anaphora_indexes.append(index)

        if antecedent_indexes != [] and anaphora_indexes != []: # there should be at least one from both of them.
            

    return feature_vector


def main():
    training_sentences_y = []
    training_sentences_y_id = []
    # open file in read mode
    with open('disambiguation_answers_file.csv', 'r', encoding='utf8') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            training_sentences_y.append(row[1])
            training_sentences_y_id.append(row[0])
    del training_sentences_y[0]  # delete header
    del training_sentences_y_id[0]  # delete header

    training_sentences_x = []
    training_sentences_x_id = []
    # open file in read mode
    with open('training_set.csv', 'r', encoding='utf8') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            new_row = row[1].replace("<referential>", "")
            new_row = new_row.replace("</referential>", "")
            if row[0] in training_sentences_y_id: # if it is ambiguous, add
                training_sentences_x.append(new_row)

    feature_vectors = []
    for sentence in training_sentences_x:
        tags = preprocessing(sentence)
        feature_vector = featureExtraction(tags)
        feature_vectors.append(feature_vector)
    '''
    lreg = linear_model.LogisticRegression()
    lreg.fit(feature_vectors, training_sentences_y)

    predicted_sentences_y = lreg.predict(feature_vectors)
    print_model_performace_for_test_doc(training_sentences_y, predicted_sentences_y)
    '''

if __name__ == '__main__':
    main()
