from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from sklearn import svm
import numpy as np
from datetime import datetime
import logging
from datetime import datetime
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(filename)s %(lineno)d: %(levelname)s  %(message)s')

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    t1 = datetime.now()
    logging.info("plot learning curve start at {}".format(t1))
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.xlabel("Train example")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_score_mean = np.mean(train_scores, axis=1)
    train_score_std = np.std(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)
    test_score_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_score_mean-train_score_std, train_score_mean+train_score_std, alpha=0.1, color='r')

    plt.fill_between(train_sizes, test_score_mean-test_score_std, test_score_mean+test_score_std, alpha=0.1, color='g')

    plt.plot(train_sizes, train_score_mean, 'o-', color='r', label="training score")
    plt.plot(train_sizes, test_score_mean, 'o-', color='g', label="cross-validate score")
    plt.legend(loc="best")
    logging.info("plot finished at {}".format(datetime.now()))
    return plt

def show_some_digit(images, labels):

    images_and_labels = list(zip(images, labels))
    for index, (image, label) in enumerate(images_and_labels[:8]):
        plt.subplot(2, 4, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Digit: {}'.format(label), fontsize=20)

def test_svm(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    logging.info("training svm model......")
    clf = svm.SVC(gamma=0.001, C=1.0)
    t1 = datetime.now()
    clf.fit(X_train, Y_train)
    t2 = datetime.now()
    logging.info('train completed, cost: {} micro seconds'.format((t2 - t1).microseconds))
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, Y_test)
    logging.info("total score: {}".format(score))
    logging.info(""
                 "classification report for svm algorithm:")
    print(classification_report(Y_test, y_pred))
    logging.info("confusion matrix:")
    print(confusion_matrix(Y_test, y_pred))
    return clf

def main():
    # (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    # logging.info(train_x.shape)
    # logging.info(train_y.shape)
    # train_x_1d = train_x.reshape(train_x.shape[0], -1)
    # test_x_1d = test_x.reshape(test_x.shape[0], -1)
    # train_x = train_x / 255.0
    # test_x = test_x / 255.0
    digits = datasets.load_digits()
    logging.info("shape of raw image data: {}".format(digits.images.shape))
    logging.info("shape of data: {}".format(digits.data.shape))
    plt.figure(figsize=(8, 6), dpi=200)
    show_some_digit(digits.images, digits.target)
    plt.figure(figsize=(8, 6), dpi=200)
    plot_learning_curve(svm.SVC(gamma=0.001, C=1.0), "learn curve of svm", digits.data, digits.target)

    test_svm(digits.data, digits.target)
    plt.show()

if __name__ == '__main__':
    main()