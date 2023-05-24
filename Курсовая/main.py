import loadnet
from tkinter import messagebox
import tkinter as tk
from PIL import Image, ImageTk, ImageGrab
# import ImageGrab
import numpy as np
from tkinter.ttk import Label
import matplotlib.pyplot as plt
import cv2
import pyautogui
import imutils
import os


class new_paint_panel(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Распознавалка рукописных цифр")
        self.previous_x = self.previous_y = 0
        self.x = self.y = 0
        self.points_recorded = []
        self.brush_size = 10

        # Метод где делается снимок экрана и форматирование картинки
        def save():
            canvas = self._canvas()  # Координаты окна
            image = pyautogui.screenshot(region=(canvas))

            image = np.array(image)
            image = image[2:300, 2:300]

            image = Image.fromarray(image, 'RGB')

            bg = Image.new('RGB', (420, 420), (0, 0, 0))
            bg.paste(image, (60, 60))

            image = np.array(bg)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            plt.imshow(gray, cmap=plt.cm.binary)
            plt.show()

            edged = cv2.Canny(gray, 10, 250)

            plt.imshow(edged, cmap=plt.cm.binary)
            plt.show()

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

            contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 1)
            #cv2.imshow('contours', image)

            cntre = contours[0]
            x, y, w, h = cv2.boundingRect(cntre)

            x -= 50
            y -= 50
            w += 100
            h += 100

            image = image[y:y + h, x:x + w]

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (28, 28))

            plt.imshow(image, cmap=plt.cm.binary)
            plt.show()

            img_array = image.astype('float32')
            img_array /= 255.0

            img_array = img_array.reshape(1, 28, 28, 1)
            # Загрузка фотографии в модель
            arr = loadnet.model.predict(img_array)
            # Сортированный список одной размерности
            arr1 = np.argmax(arr, axis=1)
            # Меняем размерность(список с 10ю элементами)
            arr = arr.reshape(10, 1)
            print(arr)
            if max(arr) > 0.5:
                messagebox.showinfo("Результат", "Вероятно вы нарисовали цифру: %.0f" % arr1[0])
            else:
                messagebox.showinfo("Результат", "Не удалось распознать, нарисуйте цифру заново")

        # Отрисовка элементов формы
        self.canv = tk.Canvas(self, width=300, height=300, bg="black", cursor="cross")
        self.canv.pack(side="top", fill="both", expand=True)

        self.button_print = tk.Button(self, text="Распознать", command=save)
        self.button_print.pack(side="top", fill="both", expand=True)

        self.button_clear = tk.Button(self, text="Стереть", command=self.clear_all)
        self.button_clear.pack(side="top", fill="both", expand=True)

        self.canv.bind("<Motion>", self.tell_me_where_you_are)
        self.canv.bind("<B1-Motion>", self.draw_from_where_you_are)

    # Возвращает окно, на котором мы рисуем
    def _canvas(self):
        print('self.canv.winfo_rootx() = ', self.canv.winfo_rootx())
        print('self.canv.winfo_rooty() = ', self.canv.winfo_rooty())
        print('self.canv.winfo_x() =', self.canv.winfo_x())
        print('self.canv.winfo_y() =', self.canv.winfo_y())
        print('self.canv.winfo_width() =', self.canv.winfo_width())
        print('self.canv.winfo_height() =', self.canv.winfo_height())
        x = self.canv.winfo_rootx() + self.canv.winfo_x()
        y = self.canv.winfo_rooty() + self.canv.winfo_y()
        x1 = x + self.canv.winfo_width()
        y1 = y + self.canv.winfo_height()
        box = (x, y, x1, y1)
        print('box = ', box)
        return box

    # Очищает форму
    def clear_all(self):
        self.canv.delete("all")

    # Запоминает предыдущие координаты
    def tell_me_where_you_are(self, event):
        self.previous_x = event.x
        self.previous_y = event.y

    #
    def draw_from_where_you_are(self, event):
        # Кисть(создает овалы)
        self.canv.create_oval(event.x - self.brush_size,
                              event.y - self.brush_size,
                              event.x + self.brush_size,
                              event.y + self.brush_size,
                              fill="#ffffff", outline="#ffffff")


# Точка входа в программу
if __name__ == "__main__":
    root = new_paint_panel()
    root.mainloop()

