# Useful links:


# Detection:
#   Precision = number of sentences correctly detected as ambiguous divided by the total number detected by the system as ambiguous
#   Recall = number of sentences correctly detected as ambiguous divided by the total number annotated by humans as ambiguous
# Resolution:
#   Precision = number of correctly resolved anaphors divided by the total number of anaphors attempted to be resolved
#   Recall = number of correctly resolved anaphors divided by the total number of unambiguous anaphors
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from csv import reader
import re
from sklearn import linear_model
from sklearn import *
from sklearn import metrics
import csv


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


def featureExtraction(tags, ref):
    print("--------------------------------------")
    print(tags)


    # for candidate i and a pronoun j
    # word_distance: i-j aradasında kaç kelime var (integer)
    # gender_agreement: i-j aynı cinsiyette mi (her - she)
    # number_agreement: i-j'nin ikisi de çoğul ya da tekil mi
    # parallelism: i-j ikisi de subject ya da object mi
    # sonrasında and + NN var mı?
    # cümlenin uzunluğu
    print(type(tags))
    antecedents = [] # The man is hungry, (NN*)
    antecedent_number = 0
    #referentials = []   # so, he eats dinnner. (referential)
    anaphoras = []
    index = 0


    #k = 0 # reflere NNP tag vermiş, o karışmasın diye tag değiştiriyorum burda
    #for tag in tags:
    #    if tag[0] == "BEGINreferential" or tag[0] == "ENDreferential":
    #        tags[k] = (tag[0], 'REFERANTIAL')
    #    k = k + 1

    for tag in tags:
        if tag[1].startswith("NN"):
            antecedent_number = antecedent_number + 1
            antecedents.append([index,tag[0],tag[1]])

        if tag[1].startswith("PRP"):
            anaphoras.append([index,tag[0],tag[1]])
        index = index + 1

        #if tag[1] == "BEGINreferential":
        #    referentials.append([index,tag[0],tag[1]])


    #if antecedents != [] and referentials != []:  # there should be at least one from both of them.
        # lenght of sentence
     #   feature_vector[0] = len(tags)
     #   for antecedent in antecedents:
     #       for anaphora in anaphoras:
     #           feature_vector[0] = len(tags)

    print("antecedents")
    print(antecedents)
    print("anaphoras")
    print(anaphoras)
    print("ref")
    print(ref)
    #print("tags: ", tags)

    vector_size = 0
    #for antecedent in antecedents:
    #    vector_size = vector_size + len(antecedent)
    #print(vector_size)
    feature_vectors = []
    feature_vector = 2*[0]
    #feature_vector = np.zeros((len(antecedents), 2))
    #feature_vector = [[0] * len(antecedents)] #* len(antecedents)
    i=0
    for anaphora in anaphoras:
        for antecedent in antecedents:
            feature_vector = 2*[0]
            if anaphora[1] == ref:
                #print(anaphora[1])
                if anaphora[0] < antecedent[0]: #1 if anaphora before antecedent
                    feature_vector[0] = 1
            if len(antecedents) == len(anaphoras): #1 if anaphora number is equal to antecedent number
                feature_vector[1] = 1

        feature_vectors.append(feature_vector)
    return feature_vectors


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

    referential_list = []
    training_sentences_x = []
    # open file in read mode
    with open('training_set.csv', 'r', encoding='utf8') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            sentence = ', '.join(row[1:])
            # row variable is a list that represents a row in csv
            if row[0] in training_sentences_y_id: # if it is unambiguous, add

                sentence = ', '.join(row[1:]) # csv_reader , lerden ayırdığı için cümle için virgülleri de ayırıyor. onları birleştiriyoruz.

                ref_list, index_ref_start_, index_ref_starta, index_ref_startb, l = [], [], [], [], 0
                index_ref_start_ = [m.start() for m in re.finditer('<referential>', sentence)]
                index_ref_starta = [m.start() for m in re.finditer('<referential id="a">', sentence)]
                index_ref_startb = [m.start() for m in re.finditer('<referential id="b">', sentence)]
                index_ref_end = [m.start() for m in re.finditer('</referential>', sentence)]

                if len(index_ref_start_) > 0: index_ref_start, l = index_ref_start_, len('<referential>')
                if len(index_ref_starta) > 0: index_ref_start, l = index_ref_starta, len('<referential id="a">')
                if len(index_ref_startb) > 0: index_ref_start, l = index_ref_startb, len('<referential id="b">')

                for index in range(len(index_ref_start)):
                    ref = sentence[index_ref_start[index]+l:index_ref_end[index]]
                    ref_list.append(ref)

                try: sentence = sentence.replace('<referential>', "")
                except: pass
                try: sentence = sentence.replace('<referential id="a">', "")
                except: pass
                try: sentence = sentence.replace('<referential id="b">', "")
                except: pass
                sentence = sentence.replace("</referential>", "")
                
                referential_list.append(ref_list)
                training_sentences_x.append(sentence)

            print("referential_list: ", referential_list)
            print("-----------------------------------")

    print(training_sentences_x)

    print((referential_list))
    feature_vectors = []
    i=0
    for sentence in training_sentences_x:
        tags = preprocessing(sentence)
        feature_vector = featureExtraction(tags, referential_list[i])
        print(feature_vector)
        feature_vectors.append(feature_vector)
        i=i+1

    '''
    lreg = linear_model.LogisticRegression()
    lreg.fit(feature_vectors, training_sentences_y)

    predicted_sentences_y = lreg.predict(feature_vectors)
    print_model_performace_for_test_doc(training_sentences_y, predicted_sentences_y)
    '''

if __name__ == '__main__':
    main()
