import sys
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from keras.applications import VGG19
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

LABELS = ['No_pain', 'Pain']
WIDTH = int(sys.argv[1])
HEIGHT = int(sys.argv[2])


# Image Data Augmentation.
def data_gen(dictionary):
    generator = ImageDataGenerator(
        samplewise_std_normalization=True,
        rotation_range=20,
        brightness_range=(-1, 1),
        shear_range=0.5,
        rescale=2,
        width_shift_range=0.2,
        height_shift_range=0.2
    )

    train_data = generator.flow_from_directory(
        directory=dictionary['Training'],
        target_size=(WIDTH, HEIGHT),
        classes=LABELS,
        class_mode='binary',
        shuffle=True
    )

    test_data = generator.flow_from_directory(
        directory=dictionary['Testing'],
        target_size=(WIDTH, HEIGHT),
        classes=LABELS,
        class_mode='binary',
        shuffle=True
    )

    val_data = generator.flow_from_directory(
        directory=dictionary['Validaiton'],
        target_size=(WIDTH, HEIGHT),
        classes=LABELS,
        class_mode='binary',
        shuffle=True
    )

    print('Classification for images of size ' + str(WIDTH) + ' X ' + str(HEIGHT))

    # Creates a Model.
    model1 = classifier()
    model1.fit(train_data, epochs=10, steps_per_epoch=WIDTH, verbose=1, validation_data=val_data, validation_steps=64)

    loss, accuracy = model1.evaluate(test_data)

    y_pred = model1.predict(test_data)
    y_pred = y_pred < 0.35

    labels = list(test_data.class_indices.keys())
    cm = confusion_matrix(test_data.classes, y_pred)

    # The results are printed.
    print('The accuracy of the model: ', accuracy)

    precision, recall, fscore, score = precision_recall_fscore_support(test_data.classes, y_pred, average='binary')

    print('The Precision of the model: ', precision)
    print('The Recall of the model: ', recall)
    print('The F1 Score of the model: ', fscore)

    print('The Confusion Matrix: ', cm)
    display_ = ConfusionMatrixDisplay(cm, display_labels=labels)
    display_.plot()


# Model is created using VGG19.
def classifier():
    model = tf.keras.Sequential()

    model.add(VGG19(input_shape=(WIDTH, HEIGHT, 3), weights='imagenet', include_top=False))

    model.add(Flatten())

    model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=512, activation='relu'))

    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=RMSprop(learning_rate=0.00001, momentum=0.9), loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
