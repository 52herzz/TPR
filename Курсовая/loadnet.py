from keras.models import load_model
#Заранее обучили и вытягиваем каждый раз уже обученную
model = load_model('My_Neiroset.h5')
print("Загрузка прошла успешно")