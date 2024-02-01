import tarfile
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

tars = ["Assignment3/20021010_easy_ham.tar.bz2", "Assignment3/20021010_spam.tar.bz2"]

encodings = ["utf-8", "ascii", "ISO-8859-1"]

SEED = 1234

def decodeEmail(email):
    email = email.read()
    for encoding in encodings:
        try:
            return email.decode(encoding)
        except UnicodeDecodeError:
            pass

emails = []
emails_category = []

for tar in tars:
    files = tarfile.open(tar)
    for member in files.getmembers():
        file = files.extractfile(member)
        if file is not None:
            emails.append(decodeEmail(file))
            if "spam" in tar:
                emails_category.append(1)
            else:
                emails_category.append(0)
    files.close()

x_train, x_test, y_train, y_test = train_test_split(emails, emails_category, test_size=0.25, random_state=SEED)

cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

bnb = MultinomialNB()
bnb.fit(x_train, y_train)

category_predicted = bnb.predict(x_test)

print("Accuracy : ", (y_test == category_predicted).sum() / len(y_test))
