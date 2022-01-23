# Импорт всех классов
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout

from kivy.core.window import Window

# Глобальные настройки
Window.size = (300, 500)
Window.clearcolor = (0, 0, 0, 1)
Window.title = "Конвертер"


class MyApp(App):

    # Создание всех виджетов (объектов)
    def __init__(self):
        super().__init__()
        self.label = Label(text='Конвертер')
        self.miles = Label(text='Мили')
        self.metres = Label(text='Метры')
        self.santimetres = Label(text='Сантиметры')
        self.button = Button(text='Button')
        self.input_data = TextInput(text='Введите значение (км)')
        # self.input_data.bind(text=self.on_text)  # Добавляем обработчик события

    # Получаем данные и производим их конвертацию
    def on_text(self, *args):
        data = self.input_data.text
        if data.isnumeric():
            self.miles.text = 'Мили: ' + str(float(data) * 0.62)
            self.metres.text = 'Метры: ' + str(float(data) * 1000)
            self.santimetres.text = 'Сантиметры: ' + str(float(data) * 100000)
        else:
            self.input_data.text = ''

    def callback(self):
        print("My button <%s> state is <%s>" % (self, self))

    # Основной метод для построения программы
    def build(self):
        # Все объекты будем помещать в один общий слой
        box = BoxLayout(orientation='vertical')
        box.add_widget(self.button)
        # self.button.bind(on_press=callback)
        box.add_widget(self.label)
        box.add_widget(self.input_data)
        box.add_widget(self.miles)
        box.add_widget(self.metres)
        box.add_widget(self.santimetres)

        return box


# Запуск проекта
if __name__ == "__main__":
    MyApp().run()
