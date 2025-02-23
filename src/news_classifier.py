import joblib
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline

lemmatizer = nltk.WordNetLemmatizer()


def extract_words(text):
    words = nltk.word_tokenize(text.lower())
    words = nltk.pos_tag(words, tagset="universal")
    words = [lemmatizer.lemmatize(word, tag[0].lower()) for word, tag in words if tag in ("NOUN", "VERB")]
    return " ".join(sorted(set(words)))


if __name__ == "__main__":
    train_data = pd.read_csv("./data/train.csv")
    y_train = train_data["Class Index"].to_list()
    x_train = (train_data["Title"] + " " + train_data["Description"]).to_list()
    x_train = list(map(extract_words, x_train))

    cvect = CountVectorizer(binary=True)
    nbayes = BernoulliNB(binarize=None)
    model = make_pipeline(cvect, nbayes)
    model.fit(x_train, y_train)

    test_data = pd.read_csv("./data/test.csv")
    y_test = test_data["Class Index"].to_list()
    x_test = (test_data["Title"] + " " + test_data["Description"]).to_list()
    x_test = list(map(extract_words, x_test))

    accuracy = model.score(x_test, y_test)
    print(f"{100*accuracy=:.2f}%")

    if input("save model? (y/N) ") == "y":
        joblib.dump(model, "./data/model.joblib")
