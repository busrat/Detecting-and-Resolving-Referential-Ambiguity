#Useful links:
# https://www.researchgate.net/publication/313867290_Using_NLP_to_Detect_Requirements_Defects_An_Industrial_Experience_in_the_Railway_Domain
# https://github.com/BenedettaRosadini/QuARS-/tree/master/jape
# https://sci-hub.se/10.1007/978-3-319-54045-0_24



#Detection:
#   Precision = number of sentences correctly detected as ambiguous divided by the total number detected by the system as ambiguous
#   Recall = number of sentences correctly detected as ambiguous divided by the total number annotated by humans as ambiguous
#Resolution:
#   Precision = number of correctly resolved anaphors divided by the total number of anaphors attempted to be resolved
#   Recall = number of correctly resolved anaphors divided by the total number of unambiguous anaphors

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

def preprocessing(sentence):
    # 1. Word tokenization
    tokenized_words = word_tokenize(sentence)

    # 2. Stopwords elimination
    stop_words = set(stopwords.words('english'))
    filtered_words = []
    for word in tokenized_words:
        if not word in stop_words:
            filtered_words.append(word)

    # 3. Lemmatization
    lemmatized_words = []
    lemmatizer = WordNetLemmatizer()
    for word in filtered_words:
        lemmatized_words.append(lemmatizer.lemmatize(word))

    # 4. POS tagging
    tagged_words = nltk.pos_tag(lemmatized_words)


    # 5. Parsing
    reg_exp = "NP: { < DT >? < JJ > * < NN >}"
    rp = nltk.RegexpParser(reg_exp)
    result = rp.parse(tagged_words)

    print(result)
    return result

def detectionReferentialAmbiguity(preprocessed_sentence):
    # rule: when sentence containing "anaphora or pronoun such as they or them" replaces with "the farthest noun"
    # rule: when sentence containing "plural nouns" add "each" before it

    result = "True"
    return result

def main():
    sentence1 = "All material that is stored in the repository will enter <referential>it</referential> via the Ingest function."
    sentence2 = "The library may want to accept important digital materials in non-standard formats in case we are able to migrate <referential>them</referential> to a more usable format in the future."

    ambiguity_indicators = "I, he, she,it, me, her, him, them, hers, his, its, your, their," \
                           " our, herself, himself, itself, ours, ourselves, yourself, themselves," \
                           "yourselves, that, theirs, these, they, this, which, who, you, yours," \
                           " someone, anyone, everyone, somebody, anybody, everybody, something," \
                           " anything, everything"
    # ambiguity_indicators_list = np.split(", ")


    preprocessing(sentence1)
if __name__ == '__main__':
    main()

