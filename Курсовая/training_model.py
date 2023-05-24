import keras
import numpy as np
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
import keras.layers.convolutional
import keras.layers.core
from keras.utils import np_utils

# constant
# Размер подвыборки (общее число тренировочных объектов)
BATCH_SIZE = 64
# Кол-во раз обучения(повышается точность распознования нейроной сетью)
NB_EPOCH = 10
# 10 цифр кол-во выходных нейронов
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
# Проверка на значение ошибки(если достигает порога, то останавливается)
early_stopping = EarlyStopping(monitor='value_loss')

# скачиваем данные и разделяем на набор для обучения и тестовый
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Создаем массивы данных (кол-во измерений второй параметр)
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# Нормализуем
X_train /= 255.0
X_test /= 255.0
# Преобразование данных в массивы (NB_CLASSES- кол-во цифр делает массив по 10 элементов)
Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)
# Создаем модель
model = keras.Sequential()
# Добавляем слои свертки (Ядро с матрицей весов)
# MaxPooling2D - эта функция сжимает картинку или слой свертки
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=2))

model.add(Conv2D(64, (5, 5), padding='valid', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=2))

# 2D-данные в 1D-данные.
model.add(Flatten())
# Добавляем полносвязные слои(ко всем нейронам)
model.add(Dense(256, activation='relu'))

model.add(Dense(84, activation='relu'))
# 10 выходных сигналов (вероятностей),
model.add(Dense(10, activation='softmax'))

# Сборка модели
model.summary()

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Функция обучения модели
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                    epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT,
                    verbose=1, callbacks=[early_stopping])

print('Testing...')

score = model.evaluate(X_test, Y_test, verbose=0)
# Тестовая оценка
#print("\nTest score:", score[0])
# Точность теста
#print('Test accuracy:', score[1])

# save model
model.save('My_Neiroset.h5')
#print("Модель сохранена")
