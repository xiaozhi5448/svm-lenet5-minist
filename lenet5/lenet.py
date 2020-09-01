from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.metrics import classification_report
import os

def generate_model():
    model = keras.Sequential()

    model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

    model.add(layers.AveragePooling2D())

    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))

    model.add(layers.AveragePooling2D())

    model.add(layers.Flatten())

    model.add(layers.Dense(units=120, activation='relu'))

    model.add(layers.Dense(units=84, activation='relu'))

    model.add(layers.Dense(units=10, activation='softmax'))
    return model

def main():
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    train_x = train_x / 255.0
    test_x = test_x / 255.0
    train_x = tf.expand_dims(train_x, 3)
    test_x = tf.expand_dims(test_x, 3)
    val_x = train_x[:5000]
    val_y = train_y[:5000]
    if os.path.exists('model.m'):
        model = keras.models.load_model('model.m')
    else:

        model = generate_model()
        model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        model.fit(train_x[5000:], train_y[5000:], epochs=3, validation_data=(val_x, val_y))
        model.save('model.m')
    pred_y = model.predict_classes(test_x)
    res = model.evaluate(test_x, test_y)

    print('accuracy:{}'.format(res[1]))
    print('classification report:')
    print(classification_report(test_y, pred_y))
    conf_matrix = tf.math.confusion_matrix(test_y, pred_y, 10)
    print('confusion matrix:')
    print(conf_matrix.numpy())


if __name__ == '__main__':
    main()