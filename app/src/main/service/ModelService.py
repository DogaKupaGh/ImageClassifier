import os
import pickle
import random
import shutil
import string
import time

import cv2
import numpy as np
from keras.api.keras import Sequential
from keras.api.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow._api.v2 import image
from keras.models import load_model

from app.src.resources.Environments import pathResultsMap, pathModels, inputSizeW, inputSizeH


def trainModel(pathSet: str, pathSetSplit: str, pathResultsMap: str, pathOutputs: str, pathModels: str, inputSizeW: int, inputSizeH: int, epochsCount: int,trainPercentage: float):  # trainPercentage 70, 80
    trainPercentage = float(trainPercentage / 100)
    trainFolder = pathSetSplit + "/train"
    validationFolder = pathSetSplit + "/validation"

    # create the new folders
    if not os.path.exists(pathSetSplit):
        os.makedirs(pathSetSplit)

    if not os.path.exists(trainFolder):
        os.makedirs(trainFolder)
    else:
        shutil.rmtree(trainFolder)
        os.makedirs(trainFolder)
    if not os.path.exists(validationFolder):
        os.makedirs(validationFolder)
    else:
        shutil.rmtree(validationFolder)
        os.makedirs(validationFolder)

    for folder in os.listdir(pathSet):
        folderPath = os.path.join(pathSet, folder)
        if os.path.isdir(folderPath):
            trainFolderPath = os.path.join(trainFolder, folder)
            validationFolderPath = os.path.join(validationFolder, folder)

            # Train ve Validation klasörlerini oluşturma
            if not os.path.exists(trainFolderPath):
                os.makedirs(trainFolderPath)
            if not os.path.exists(validationFolderPath):
                os.makedirs(validationFolderPath)

            # Dosyaları kopyalama
            files = os.listdir(folderPath)
            random.shuffle(files)
            trainFiles = files[:int(trainPercentage * len(files))]
            validationFiles = files[int(trainPercentage * len(files)):]

            for file in trainFiles:
                shutil.copy2(os.path.join(folderPath, file), os.path.join(trainFolderPath, file))
            for file in validationFiles:
                shutil.copy2(os.path.join(folderPath, file), os.path.join(validationFolderPath, file))

    dataCount = len([f for f in os.listdir(pathSet) if os.path.isdir(os.path.join(pathSet, f))])

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(inputSizeW, inputSizeH, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4)) # 0.3  0.4   0.5
    model.add(Dense(dataCount,
                    activation='softmax'))

    # MODEL ÖZETİ
    model.summary()

    # MODEL DERLEME
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    trainGenerator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        trainFolder,
        target_size=(inputSizeW, inputSizeH),
        batch_size=32, #16 32 64
        color_mode='rgb',
        class_mode='categorical')

    validationGenerator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        validationFolder,
        target_size=(inputSizeW, inputSizeH),
        batch_size=32, #16 32 64
        color_mode='rgb',
        class_mode='categorical')

    trainClasses = trainGenerator.class_indices

    ResultMap = {}
    for v, n in zip(trainClasses.values(), trainClasses.keys()):
        ResultMap[v] = n

    modelName:str = ''.join(random.choice(string.ascii_lowercase) for i in range(12))

    with open(pathResultsMap + modelName + ".pkl", 'wb') as f:
        pickle.dump(ResultMap, f)

    startTime = time.time()

    model.fit(
        trainGenerator,
        steps_per_epoch=len(trainGenerator),
        epochs=epochsCount,
        validation_data=validationGenerator,
        validation_steps=len(validationGenerator),
        verbose=1)

    endTime = time.time()

    xTrain, yTrain = trainGenerator.next()
    xVal, yVal = validationGenerator.next()

    with open(pathOutputs + modelName + '.txt', 'w') as f:
        f.write('Epoch\tLoss\tAccuracy\tVal_Loss\tVal_Accuracy\n')
        for epoch in range(epochsCount):
            loss, accuracy = model.train_on_batch(xTrain, yTrain)
            valLoss, valAccuracy = model.test_on_batch(xVal, yVal)
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(epoch + 1, loss, accuracy, valLoss, valAccuracy))
        duration = round(endTime - startTime)
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)

        if int(hours) != 0:
            f.write('\n{} hours {} minutes {} seconds'.format(hours, minutes, seconds))
        elif int(minutes) != 0:
            f.write('\n{} minutes {} seconds'.format(minutes, seconds))
        else:
            f.write('\n{} seconds'.format(seconds))

    # save the model
    model.save(pathModels + modelName + '.h5')

    return modelName + ".h5"


def testModel(modelName: str, pathTestImage: str):
    model = load_model(pathModels + modelName)
    img = cv2.imread(pathTestImage)
    resized_img = cv2.resize(img, (256, 256)) / 255.0
    prediction = model.predict(np.expand_dims(resized_img, axis=0))
    predictedClass = np.argmax(prediction)

    with open(pathResultsMap + modelName.replace(".h5", ".pkl"), 'rb') as fileReadStream:
        ResultMap = pickle.load(fileReadStream)

    predictedName = ResultMap[predictedClass]
    accuracy = round(np.max(prediction) * 100, 2)

    cv2.putText(img, f"Tahmin: {predictedName}, Dogruluk: {accuracy}%", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

    img = cv2.resize(img, (768, 768))
    cv2.imshow("Tahmin", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# def testModel(modelName: str, pathTestImage: str):
#     model = load_model(pathModels + modelName)
#     img = cv2.imread(pathTestImage)
#     # resize = image.resize(img, (256,256))
#     # yhat = model.predict(np.expand_dims(resize / 255, 0))
#
#     with open(pathResultsMap + modelName.replace(".h5", ".pkl"), 'rb') as fileReadStream:
#         ResultMap = pickle.load(fileReadStream)
#
#
#     prediction = model.predict(img, verbose=0)
#     predictedClass = np.argmax(prediction)
#     predictedName = ResultMap[predictedClass]
#     accuracy = round(np.max(prediction) * 100, 2)
#
#     cv2.putText(img, predictedName + " " + str(
#         accuracy) + "%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     cv2.imshow('Result', img)
#     cv2.waitKey(0)