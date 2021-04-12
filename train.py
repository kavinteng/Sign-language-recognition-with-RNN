import os
import sys
import argparse
from train_utils import make_label, load_data, build_model
import keras
from matplotlib import pyplot as plt

def main():
    dirname = '/Project/mediapipe/outputdata/Absolute'
    x_train, y_train, x_test, y_test = load_data(dirname)
    lenxtr = (x_train[0])
    lenxte = (x_train[0][0])
    lenytr = (x_train[0][0][0])
    lenyte = len(y_test)
    print("1aa",lenxtr,"2aa",lenytr,"3aa",lenxte)
    num_val_samples = (x_train.shape[0]) // 5
    print(x_train.shape[0])
    model = build_model(y_train.shape[1])
    print('Training stage')
    print('==============')
    history = model.fit(x_train, y_train, epochs=120, batch_size=16, validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, batch_size=16, verbose=0)
    # print(history.history.keys())
    model.summary()

    plt.figure(1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('eieieieiei')
    # plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('acc.png')

    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    # plt.ylabel('loss')
    plt.ylabel('55555555555555555')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')

    print('Test performance: accuracy={0}, loss={1}'.format(acc, score))
    model.save('cuda.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument("--input_train_path", help=" ")
    args = parser.parse_args()
    input_train_path = args.input_train_path
    main()
