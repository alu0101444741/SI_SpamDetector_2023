# ********************************************************
# *******************   INTERFAZ   ***********************
# ********************************************************
import spamdetection as spam
from tkinter import *

root = Tk()
root.title("Programa ABC")

# MultibinomialNB
# Nearest Neighbors
# Linear SVM
# RBF SVM
# Decision Tree
# Random Forest
# LogisticRegression

global models_activated
global status_labels
global models_buttons
global prediction_results

models_activated = [False, False, False, False, False, False, False]
status_labels = []

for index in range(1, len(models_activated)+1):
    status_labels.append(Label(root, text = "[ ]", width = 5))
    status_labels[index-1].grid(row=index, column=0)

# Funcion para (des)activar los modelos a emplear
def activate_model(index):  
    if (models_activated[index]):
        status_labels[index].config(text="[ ]")
    else:
        status_labels[index].config(text="[X]")
    models_activated[index] = not models_activated[index]

# Botones para cada modelo
models_buttons = []

models_buttons.append(Button(root, text=spam.names[0], width=20, command=lambda: activate_model(0)))
models_buttons[0].grid(row = 1, column = 1)

models_buttons.append(Button(root, text=spam.names[1], width=20, command=lambda: activate_model(1)))
models_buttons[1].grid(row = 2, column = 1)

models_buttons.append(Button(root, text=spam.names[2], width=20, command=lambda: activate_model(2)))
models_buttons[2].grid(row = 3, column = 1)

models_buttons.append(Button(root, text=spam.names[3], width=20, command=lambda: activate_model(3)))
models_buttons[3].grid(row = 4, column = 1)

models_buttons.append(Button(root, text=spam.names[4], width=20, command=lambda: activate_model(4)))
models_buttons[4].grid(row = 5, column = 1)

models_buttons.append(Button(root, text=spam.names[5], width=20, command=lambda: activate_model(5)))
models_buttons[5].grid(row = 6, column = 1)

models_buttons.append(Button(root, text=spam.names[6], width=20, command=lambda: activate_model(6)))
models_buttons[6].grid(row = 7, column = 1)

# Etiqueta donde se muestra el resultado de la predicción para cada modelo
prediction_results = []
prediction_labels = []
label = True
i = 0
for index in range(0, len(models_activated)):
    i = index
    prediction_labels.append(Label(root, text = "Prediction:"))
    prediction_labels[index].grid(row=i+1, column=2)
    i = i + 1
    prediction_results.append(Label(root, text = "-"))
    prediction_results[index].grid(row=i, column=3)

# Campo para escribir un email
input_label = Label(root, text="Write an email:").grid(row = 8 , column = 1, columnspan=2)

input_scrollbar = Scrollbar(root, orient='vertical')
input_scrollbar.grid(row = 9 , column = 1)
input_field = Text(root, width=40, height=10, yscrollcommand=input_scrollbar.set)

input_scrollbar.config(command=input_field.yview)
input_field.grid(row = 9, column = 0, columnspan = 5)

# Botón para ejecutar la predicción
input_button = Button(root, text="Verify", command=lambda: click_verify(input_field.get("1.0",END)))
input_button.grid(row = 10, column = 0)

def click_verify(text):
    test = spam.Vectorizer.transform([text])

    for index in range(0, len(models_activated)):
        if(models_activated[index]):
            actual_model = spam.classifiers[index]
            actual_model.fit(spam.count, spam.targets)
            y_predict = actual_model.predict(test.astype('float32'))
            prediction_results[index].config(text = y_predict)
        else:
            prediction_results[index].config(text = "-")

root.mainloop()