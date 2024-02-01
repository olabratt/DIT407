import tarfile
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

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

def evaluate(emails, categories):
    x_train, x_test, y_train, y_test = train_test_split(emails, categories, test_size=0.25, random_state=SEED)

    cv = CountVectorizer()
    x_train = cv.fit_transform(x_train)
    x_test = cv.transform(x_test)

    mnb = MultinomialNB()
    mnb.fit(x_train, y_train)

    bnb = BernoulliNB()
    bnb.fit(x_train, y_train)

    category_predicted = bnb.predict(x_test)

    print("Accuracy : ", (y_test == category_predicted).sum() / len(y_test))



easy_ham = getEmails("Assignment3/20021010_easy_ham.tar.bz2")
hard_ham = getEmails("Assignment3/20021010_hard_ham.tar.bz2")
spam = getEmails("Assignment3/20021010_spam.tar.bz2")

easy_and_spam = easy_ham + spam
easy_and_spam_cat = [0] * len(easy_ham) + [1] * len(spam)

hard_and_spam = hard_ham + spam
hard_and_spam_cat = [0] * len(hard_ham) + [1] * len(spam)

evaluate(easy_and_spam, easy_and_spam_cat)
evaluate(hard_and_spam, hard_and_spam_cat)