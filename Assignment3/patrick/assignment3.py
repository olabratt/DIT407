import tarfile
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import numpy as np

encodings = ["utf-8", "ascii", "ISO-8859-1"]

SEED = 1234

def decodeEmail(email):
    email = email.read()
    for encoding in encodings:
        try:
            return email.decode(encoding)
        except UnicodeDecodeError:
            pass

def getEmails(tar):
    emails = []

    files = tarfile.open(tar)
    for member in files.getmembers():
        file = files.extractfile(member)
        if file is not None:
            emails.append(decodeEmail(file))
    files.close()
    return emails

def evaluate(x_train, x_test, y_train, y_test, nb):
    nb.fit(x_train, y_train)
    y_pred = nb.predict(x_test)

    accuracy = (y_test == y_pred).sum() / len(y_test)
    true_spam = ((y_test == 1) & (y_pred == 1)).sum()
    false_spam = ((y_test == 0) & (y_pred == 1)).sum()
    true_ham = ((y_test == 0) & (y_pred == 0)).sum()
    false_ham = ((y_test == 1) & (y_pred == 0)).sum()

    accuracy = (true_spam + true_ham) / (true_ham + true_spam + false_ham + false_spam)
    precision = true_spam / (true_spam + false_spam)
    recall = true_spam / (true_spam + false_ham)

    return (accuracy, precision, recall, [[true_spam, false_ham], [false_spam, true_ham]]);

def train_test(emails, categories):
    x_train, x_test, y_train, y_test = train_test_split(emails, categories, test_size=0.2, random_state=SEED)

    cv = CountVectorizer()
    x_train = cv.fit_transform(x_train)
    x_test = cv.transform(x_test)

    bnb = BernoulliNB()
    accuracy, precision, recall, conf_matrix = evaluate(x_train, x_test, y_train, y_test, bnb)
    print("Bernoulli train test accuracy :", accuracy)
    print("Bernoulli train test precision :", precision)
    print("Bernoulli train test recall :", recall)
    print("Bernoulli train test confusion matrix :", conf_matrix)

    mnb = MultinomialNB()
    accuracy, precision, recall, conf_matrix = evaluate(x_train, x_test, y_train, y_test, mnb)
    print("Multinomial train test accuracy :", accuracy)
    print("Multinomial train test precision :", precision)
    print("Multinomial train test recall :", recall)
    print("Multinomial train test confusion matrix :", conf_matrix)


easy_ham = getEmails("Assignment3/20021010_easy_ham.tar.bz2")
hard_ham = getEmails("Assignment3/20021010_hard_ham.tar.bz2")
spam = getEmails("Assignment3/20021010_spam.tar.bz2")

easy_and_spam = np.array(easy_ham + spam)
easy_and_spam_cat = np.array([0] * len(easy_ham) + [1] * len(spam))

hard_and_spam = np.array(hard_ham + spam)
hard_and_spam_cat = np.array([0] * len(hard_ham) + [1] * len(spam))

print("====== Easy ham train test ======")
train_test(easy_and_spam, easy_and_spam_cat)

print()

print("====== Hard ham train test ======")
train_test(hard_and_spam, hard_and_spam_cat)