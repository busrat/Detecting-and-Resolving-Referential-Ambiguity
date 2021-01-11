# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 900000

# Not.
# Sadece baz olması için, taslak. 1ref_disambiguation_answers_file.csv ile çalışıyor çünkü iki referanslıları göz önünde bulundurmuyor şu an.
# sanırım bir takım string okuma, veri türü vs işlemlerle ilgili problemler var. debug edilmesi ve detaylı bakılması gerekiyor.
# bir cümle çok uzun orada bayagı bekletiyor
from pycorenlp import StanfordCoreNLP
from csv import reader
import re

nlp = StanfordCoreNLP('http://localhost:9000')


def accuracyCalculation(training_sentences_y, predicted_y):
    total = len(training_sentences_y)
    true_prediction = 0
    for i in range(len(predicted_y)):
        found = predicted_y[i].lower()
        label = training_sentences_y[i].strip('"').strip(' "').lower()

        if label == found or label in found or found in label:
            print("TRUE ->", label, "<--->", found, "<-")
            true_prediction += 1
        else:
            print("FALSE ->", label, "<--->", found, "<-")
    return true_prediction, total


def resolve(annotatiton, reference):
    annotationValues = [k for k in annotatiton['corefs'].values()]
    annotationValuesList = [item for sublist in annotationValues for item in sublist]

    referenceIndex = [i for i, d in enumerate(annotationValuesList) if d['text'].lower() == reference[0].lower()]
    if (len(referenceIndex) == 0):
        return "none"
    else:
        referenceIndex = referenceIndex[0]
        referenceFeatures = \
        [d for i, d in enumerate(annotationValuesList) if d['text'].lower() == reference[0].lower()][0]
        del annotationValuesList[referenceIndex]
        candidateScores = {}
        distances = {}
        for candidate in annotationValuesList:
            candidateScores[candidate['text']] = 0
            distance = candidate['id'] - referenceFeatures['id']
            distances[candidate['text']] = distance

            if candidate['number'] == referenceFeatures['number']:
                candidateScores[candidate['text']] += 1
            if candidate['gender'] == referenceFeatures['gender']:
                candidateScores[candidate['text']] += 1

        closest_candidate = min(distances, key=distances.get)
        if distances[closest_candidate]>0:
            candidateScores[closest_candidate] += 1
        return max(candidateScores, key=candidateScores.get)

def main():
    training_sentences_y = []
    training_sentences_y_id = []
    # open file in read mode
    with open('1ref_disambiguation_answers_file.csv', 'r', encoding='utf8') as read_obj:
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
    with open('Data/training_set.csv', 'r', encoding='utf8') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            sentence = ', '.join(row[1:])
            sentence.replace('"', '')
            # row variable is a list that represents a row in csv
            if row[0] in training_sentences_y_id or \
                    row[0] + "-a" in training_sentences_y_id or \
                    row[
                        0] + "-b" in training_sentences_y_id:  # we compare first columns and look if it is unambiguous, add
                ref_list, index_ref_start, index_ref_starta, index_ref_startb, l = [], [], [], [], 0
                index_ref_start_ = [m.start() for m in re.finditer('<referential>', sentence)]
                index_ref_starta = [m.start() for m in re.finditer('<referential id="a">', sentence)]
                index_ref_startb = [m.start() for m in re.finditer('<referential id="b">', sentence)]
                index_ref_end = [m.start() for m in re.finditer('</referential>', sentence)]

                if len(index_ref_start_) > 0: index_ref_start, l = index_ref_start_, len('<referential>')
                if len(index_ref_starta) > 0: index_ref_start, l = index_ref_starta, len('<referential id="a">')
                if len(index_ref_startb) > 0: index_ref_start, l = index_ref_startb, len('<referential id="b">')

                for index in range(len(index_ref_start)):
                    ref = sentence[index_ref_start[index] + l:index_ref_end[index]]
                    ref_list.append(ref)

                try:
                    sentence = sentence.replace('<referential>', "")
                except:
                    pass
                try:
                    sentence = sentence.replace('<referential id="a">', "")
                except:
                    pass
                try:
                    sentence = sentence.replace('<referential id="b">', "")
                except:
                    pass
                sentence = sentence.replace("</referential>", "")

                referential_list.append(ref_list)
                training_sentences_x.append(sentence)

    predicted_y = []
    for i, sentence in enumerate(training_sentences_x):
        # annotate sentence
        annotatiton = nlp.annotate(sentence,
                                   properties={'timeout': '900000', 'annotators': 'dcoref', 'outputFormat': 'json',
                                               'ner.useSUTime': 'false'})
        try:
            predicted_y.append(resolve(annotatiton, referential_list[i]))
        except NameError:
            predicted_y.append("none")

    true_prediction, total = accuracyCalculation(training_sentences_y, predicted_y)
    print("ACCURACY: ", true_prediction * 100 / total, " - TOTAL: ", total, " - TRUE PREDICTED: ", true_prediction)


if __name__ == '__main__':
    main()
