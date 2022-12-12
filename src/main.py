import glob
import re
import pandas as pd
from parsivar import Normalizer, Tokenizer, FindStems
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def main():
    # Loading Dataset from file
    df_train, df_test = load_dataset()

    # clean and pre-process texts
    df_train = get_processed_data(df_train)
    df_test = get_processed_data(df_test)

    X_train = list(df_train['text'])
    X_test = list(df_test['text'])

    y_train = df_train['category']
    y_test = df_test['category']

    # loading stop words
    with open('../raw/persian_stopwords.txt', encoding='utf-8') as f:
        content = f.readlines()
    stop_words = [x.strip() for x in content]

    print("~~~~~~~~~~~~ Naive Bayes With CountVectorizer ~~~~~~~~~~~~")
    count_vectorizer = CountVectorizer(max_features=500, stop_words=stop_words)
    vec_train = count_vectorizer.fit_transform(X_train).toarray()
    vec_test = count_vectorizer.transform(X_test).toarray()
    clf = GaussianNB()
    clf.fit(vec_train, y_train)
    result = clf.predict(vec_test)
    print_results(y_test, result)

    print("~~~~~~~~~~~~ KNN With TF ~~~~~~~~~~~~")
    transformer_tf = TfidfVectorizer(max_features=500, stop_words=stop_words, use_idf=False)
    tf_train = transformer_tf.fit_transform(X_train).toarray()
    tf_test = transformer_tf.transform(X_test).toarray()

    iterations = [1, 5, 15]
    for iteration in iterations:
        print(f"for K ={iteration}")
        clf = KNeighborsClassifier(n_neighbors=iteration, metric='euclidean')
        clf.fit(tf_train, y_train)
        result = clf.predict(tf_test)
        print_results(y_test, result)

    print("~~~~~~~~~~~~ KNN With TF-IDF ~~~~~~~~~~~~")
    transformer_tfidf = TfidfVectorizer(max_features=500, stop_words=stop_words, use_idf=True)
    tfidf_train = transformer_tfidf.fit_transform(X_train).toarray()
    tfidf_test = transformer_tfidf.transform(X_test).toarray()

    iterations = [1, 5, 15]
    for iteration in iterations:
        print("for K =", str(iteration))
        knn = KNeighborsClassifier(n_neighbors=iteration, metric='euclidean').fit(tfidf_train, y_train)
        result = knn.predict(tfidf_test)
        print_results(y_test, result)


def load_dataset():
    TRAIN_PATH = "../raw/Dataset/Train/"
    TEST_PATH = "../raw/Dataset/Test/"

    categories = [
        'Economics',
        'Sociology',
        'Sports',
        'Religions',
        'Tech',
        'Strategic',
        'Politics'
    ]

    df_train = pd.DataFrame(columns=["text", "category"])
    df_test = pd.DataFrame(columns=["text", "category"])

    for category in categories:
        train_files = glob.glob(TRAIN_PATH + category + "/*.txt")
        for file_path in train_files:
            with open(file_path, encoding='utf-8', newline='\n') as file:
                text = file.read()
            df_train.loc[len(df_train)] = [text, category]

        test_files = glob.glob(TEST_PATH + category + "/*.txt")
        for file_path in test_files:
            with open(file_path, encoding='utf-8', newline='\n') as file:
                text = file.read()
            df_test.loc[len(df_test)] = [text, category]

    return df_train, df_test


def get_processed_data(data_frame):
    normalizer = Normalizer()
    tokenizer = Tokenizer()
    stemmer = FindStems()

    for i in range(len(data_frame)):
        text = data_frame.loc[i]["text"]
        text = normalizer.normalize(text, new_line_elimination=True)
        text = re.sub('[\u200c]', ' ', text)
        text = re.sub('[0-9]', ' ', text)
        words = tokenizer.tokenize_words(text)
        words = [stemmer.convert_to_stem(word) for word in words]
        text = ' '.join(words)
        data_frame.loc[i]["text"] = text

    return data_frame


def print_results(y_test, result):
    print(f"accuracy: {accuracy_score(y_test, result)}")
    print(f"precision: {precision_score(y_test, result, average='macro', zero_division=0)}")
    print(f"recall: {recall_score(y_test, result, average='macro', zero_division=0)}")
    print(f"f-measure: {f1_score(y_test, result, average='macro', zero_division=0)}")
    print("confusion matrix:\n", confusion_matrix(y_test, result))


if __name__ == '__main__':
    main()
