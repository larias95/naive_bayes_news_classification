import joblib

from news_classifier import extract_words

if __name__ == "__main__":
    model = joblib.load("./data/model.joblib")

    while (text := input(">>> ")) != "exit":
        tag = model.predict([extract_words(text)])[0]
        print(tag)
