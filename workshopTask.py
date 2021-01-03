# Useful links:
# https://www.researchgate.net/publication/313867290_Using_NLP_to_Detect_Requirements_Defects_An_Industrial_Experience_in_the_Railway_Domain
# https://github.com/BenedettaRosadini/QuARS-/tree/master/jape
# https://sci-hub.se/10.1007/978-3-319-54045-0_24
# https://core.ac.uk/download/pdf/301374847.pdf
# D. L. Thanh, \Two machine learning approaches to coreference resolution," 2009.
# https://journals.ekb.eg/article_15909.html



#Detection:
#   Precision = number of sentences correctly detected as ambiguous divided by the total number detected by the system as ambiguous
#   Recall = number of sentences correctly detected as ambiguous divided by the total number annotated by humans as ambiguous
#Resolution:
#   Precision = number of correctly resolved anaphors divided by the total number of anaphors attempted to be resolved
#   Recall = number of correctly resolved anaphors divided by the total number of unambiguous anaphors

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from csv import reader
from sklearn import linear_model

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
    feature_vector = 6*[0]
    prp_counter = 0
    nn_counter = 0

    pronoun_counter = 0
    noun_counter = 0
    for tag in tags:
        if not tag[0] == tag[1]: # it is not a tag of poncutation
            # RULE 1: he she it, NNP'den önce geliyor mu: geliyorsa 1, gelmiyorsa 0
            if feature_vector[0] == 0:
                if tag[1] == "PRP": # header_property he, she, it, they
                    rule1_prp_flag = True
                if rule1_prp_flag == True and (tag[1] == "NN" or tag[1] == "NNP" or tag[1] == "NNS"):
                    feature_vector[0] = 1

            # RULE 2: bağlaç var mı: varsa 1 yoksa 0
            if feature_vector[1] == 0:
                if tag[1] == "CC":
                    feature_vector[1] = 1

            # RULE 3: birden fazla pronoun var mı: varsa 1 yoksa 0
            if feature_vector[2] == 0:
                if tag[1] == "PRP":
                    prp_counter += 1
                    if prp_counter > 1:
                        feature_vector[2] = 1

            # RULE 4: NN + NN -> he she it var mı: varsa 1 yoksa 0
            if feature_vector[3] == 0:
                if tag[1] == "NN" or tag[1] == "NNS" or tag[1] == "NNP":
                    nn_counter += 1
                if nn_counter >= 2:
                    if tag[1] == "PRP":
                        feature_vector[3] = 1

            # RULE 5: anaphors sayısı
            if tag[1] == "PRP":
                    pronoun_counter += 1

            # RULE 6: referent olabileceklerin sayısı
            if tag[1] == "NN" or tag[1] == "NNS" or tag[1] == "NNP":
                    noun_counter += 1

    feature_vector[4] = pronoun_counter
    feature_vector[5] = noun_counter
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

    del training_sentences_x[0] # delete header

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

    #print("training_sentences_y:  ", training_sentences_y)
    #print("predicted_sentences_y: ", predicted_sentences_y)
    true_prediction = 0
    for i in range(len(training_sentences_y)):
        if training_sentences_y[i] == predicted_sentences_y[i]:
            true_prediction += 1

    print("TOTAL: ", len(training_sentences_y), " - TRUE PREDICTED: ", true_prediction)

if __name__ == '__main__':
    main()
