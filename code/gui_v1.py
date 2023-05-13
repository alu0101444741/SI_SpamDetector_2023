# ********************************************************
# *******************   INTERFAZ   ***********************
# ********************************************************
import spamdetection as spam
from tkinter import *

root = Tk()
root.title("Programa ABC")

#Etiquetas iniciales
table_label = Label(root, bg="gray", width = 8, relief=GROOVE).grid(row=0, column=0)
precision_label = Label(root, text="precision", bg="gray", width = 8, relief=GROOVE).grid(row=0, column=1)
recall_label = Label(root, text="recall", bg="gray", width = 8, relief=GROOVE).grid(row=0, column=2)
f1_score_label = Label(root, text="f1_score", bg="gray", width = 8, relief=GROOVE).grid(row=0, column=3)
support = Label(root, text="support", bg="gray", width = 8, relief=GROOVE).grid(row=0, column=4)

ham_label = Label(root, text="ham", bg="gray", width = 8, relief=GROOVE).grid(row=1, column=0)
spam_label = Label(root, text="spam", bg="gray", width = 8, relief=GROOVE).grid(row=2, column=0)
accuracy_label = Label(root, text="accuracy", bg="gray", width = 8, relief=GROOVE).grid(row=3, column=0)
macro_label = Label(root, text="macro", bg="gray", width = 8, relief=GROOVE).grid(row=4, column=0)
weighted_label = Label(root, text="weighted", bg="gray", width = 8, relief=GROOVE).grid(row=5, column=0)

# Campo para escribir un email
input_label = Label(root, text="Write an email:").grid(row = 6 , column = 1, columnspan=2)

input_scrollbar = Scrollbar(root, orient='vertical')
input_scrollbar.grid(row = 7 , column = 1)
input_field = Text(root, width=40, height=10, yscrollcommand=input_scrollbar.set)

input_scrollbar.config(command=input_field.yview)
input_field.grid(row = 7, column = 0, columnspan = 5)

# Etiqueta donde se muestra el resultado de la predicción
global prediction_label
prediction_title = Label(root, text="Prediction:").grid(row = 8 , column = 0)
prediction_label = Label(root, text="-").grid(row = 8 , column = 6)

def click_verify(text):
    test = spam.Vectorizer.transform([text])
    y_predict = spam.clf.predict(test.astype('float32'))
    prediction_label = Label(root, text=str(y_predict)).grid(row = 8 , column = 1)

# Botón para ejecutar la predicción
input_button = Button(root, text="Verify", command=lambda: click_verify(input_field.get("1.0",END)))
input_button.grid(row = 9, column = 0)

# Valores en cada etiqueta
def fill_grid():
    precision = spam.np.zeros(5)
    recall = spam.np.zeros(5)
    f1_score = spam.np.zeros(5)
    support = spam.np.zeros(5, dtype=int)
    counter = 0
    for label in spam.report:        
        if (str(label) != 'accuracy'):
            precision[counter] = (round(spam.report[label]['precision'], 2))
            recall[counter] = (   round(spam.report[label]['recall'], 2))
            f1_score[counter] = ( round(spam.report[label]['f1-score'], 2))
            support[counter] = (  spam.report[label]['support'])
        else:
            precision[counter] = (spam.np.NaN)
            recall[counter] = (spam.np.NaN)
            f1_score[counter] = (round(spam.report[label], 2))
            support[counter] = (spam.report['macro avg']['support'])
        counter += 1

    # Asignación de casillas
    labels = spam.np.zeros((4, 5))

    for rows in range(1,6):        
        labels[0][rows - 1] = Label(root, text = precision[rows - 1], width = 8, relief=GROOVE).grid(row = rows, column = 1)
        labels[1][rows - 1] = Label(root, text = recall[rows - 1]   , width = 8, relief=GROOVE).grid(row = rows, column = 2)
        labels[2][rows - 1] = Label(root, text = f1_score[rows - 1] , width = 8, relief=GROOVE).grid(row = rows, column = 3)
        labels[3][rows - 1] = Label(root, text = support[rows - 1]  , width = 8, relief=GROOVE).grid(row = rows, column = 4)
    
fill_grid()
root.mainloop()