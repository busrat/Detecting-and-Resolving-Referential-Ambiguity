# Useful links:
# https://www.researchgate.net/publication/313867290_Using_NLP_to_Detect_Requirements_Defects_An_Industrial_Experience_in_the_Railway_Domain
# https://github.com/BenedettaRosadini/QuARS-/tree/master/jape
# https://sci-hub.se/10.1007/978-3-319-54045-0_24
# https://core.ac.uk/download/pdf/301374847.pdf
# D. L. Thanh, \Two machine learning approaches to coreference resolution," 2009.
# https://journals.ekb.eg/article_15909.html

# https://pythonprogramming.net/part-of-speech-tagging-nltk-tutorial/


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
from sklearn import linear_model
import re

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

def preprocessing(sentence):
    # 1. Word tokenization
    tokenized_words = word_tokenize(sentence)

    # 2. Stopwords elimination is removed (to get and or etc.)

    # 3. Lemmatization
    lemmatized_words = []
    lemmatizer = WordNetLemmatizer()
    for word in tokenized_words:
        lemmatized_words.append(lemmatizer.lemmatize(word))

    # 4. POS tagging
    tagged_words = nltk.pos_tag(lemmatized_words)

    print(tagged_words)
    return tagged_words


def featureExtraction(tags):
    '''
        4: çoğul çoğul mu?
        5: cinsiyetler uyumlu mu?
        6: word distance - arada kaç kelime var
        7: NN + NN -> he she it anaphoraNominative
    '''

    rule1_prp_flag = False
    feature_vector = 14 * [0]
    prp_counter = 0
    nn_counter = 0
    punctuation_counter = 0
    verb_counter = 0
    noun_counter = 0
    conj_counter = 0
    uppercaseLetters = 0
    in_counter = 0
    i = 0
    wrb_flag = False
    for tag in tags:
        # RULE 0: he she it, NNP'den önce geliyor mu: geliyorsa 1, gelmiyorsa 0
        if feature_vector[0] == 0:
            if tag[1] == "PRP":  # header_property he, she, it, they
                rule1_prp_flag = True
            if rule1_prp_flag == True and tag[1].startswith("NN"):
                feature_vector[0] = 1

        # RULE 1: bağlaç var mı: varsa 1 yoksa 0
        if tag[1] == "CC":
            conj_counter += 1
            if feature_vector[1] == 0:
                feature_vector[1] = 1

        # RULE 2: birden fazla pronoun var mı: varsa 1 yoksa 0
        if tag[1] == "PRP":
            prp_counter += 1
            if prp_counter > 1 and feature_vector[2] == 0:
                feature_vector[2] = 1

        # RULE 3: NN + NN -> he she it var mı: varsa 1 yoksa 0
        if feature_vector[3] == 0:
            if tag[1].startswith("NN"):
                nn_counter += 1
            if nn_counter >= 2:
                if tag[1] == "PRP":
                    feature_vector[3] = 1

        # RULE 4: verb sayısı
        if tag[1].startswith("VB"):
            verb_counter += 1
            if verb_counter > 4:
                if feature_vector[4] == 0:
                    feature_vector[4] = 1

        # RULE 5: cümle değiştiren noktalama sayısı
        if tag[1] in [',', '.', ';', '!', '?', ':', '``', "''"]:
            punctuation_counter += 1
            if punctuation_counter > 2 and feature_vector[9] == 0:
                feature_vector[5] = 1

        # RULE 6: referent olabileceklerin sayısı
        # if tag[1].startswith("NN"):
        if tag[1] == "NN" or tag[1] == "NNS" or tag[1] == "NNP":
            noun_counter += 1

        if tag[1] == "WRB":
            wrb_flag = True

         # RULE: birden fazla büyük harf var mı     
        if feature_vector[12] == 0:     
            uppercaseLetters += len(re.findall(r'[A-Z]',tag[0]))
            if uppercaseLetters > 1:
                feature_vector[12] = 1
                
        if tag[1] == "IN":
            in_counter += 1
            if in_counter > 2 and feature_vector[13] == 0:
                feature_vector[13] = 1
            
    feature_vector[7] = prp_counter
    feature_vector[6] = noun_counter
    feature_vector[8] = verb_counter
    feature_vector[9] = punctuation_counter
    feature_vector[10] = conj_counter
    
    if wrb_flag == True and prp_counter>0:
        feature_vector[11] = 1

    return feature_vector


def main():
    training_sentences_x = []
    # open file in read mode
    with open('training_set.csv', 'r', encoding='utf8') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            new_row = row[1].replace("<referential>", "")
            new_row = new_row.replace("</referential>", "")
            training_sentences_x.append(new_row)

    del training_sentences_x[0]  # delete header

    training_sentences_y = []
    # open file in read mode
    i = 0
    with open('detection_answers_file.csv', 'r', encoding='utf8') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            if not i == 0:
                if row[1] == "AMBIGUOUS":
                    training_sentences_y.append(1)
                else:
                    training_sentences_y.append(0)
            i = 1

    feature_vectors = []
    i = 0
    for sentence in training_sentences_x:
        tags = preprocessing(sentence)
        print(sentence)
        feature_vector = featureExtraction(tags)
        feature_vectors.append(feature_vector)
        print(feature_vector)

    lreg = linear_model.LogisticRegression()
    lreg.fit(feature_vectors, training_sentences_y)

    predicted_sentences_y = lreg.predict(feature_vectors)

    # print("training_sentences_y:  ", training_sentences_y)
    # print("predicted_sentences_y: ", predicted_sentences_y)
    true_prediction = 0
    for i in range(len(training_sentences_y)):
        if training_sentences_y[i] == predicted_sentences_y[i]:
            true_prediction += 1

    print("TOTAL: ", len(training_sentences_y), " - TRUE PREDICTED: ", true_prediction)


if __name__ == '__main__':
    main()
