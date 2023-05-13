# ********************************************************
# *******************   INTERFAZ   ***********************
# ********************************************************
from spamdetection import (names, classifiers, Vectorizer)
from tkinter import *
from PIL import ImageTk, Image

root = Tk()
root.title("Spam Detector")

global models_activated
global status_labels
global models_buttons
global prediction_results

models_activated = [False, False, False, False, False, False, False]
status_labels = []

# Formato para etiquetas
font_type = "Rubik"
font_size = 16
frame_background = "#B6BCFE"
button_font = ("Rubik", 12, 'bold')

# Apartados
# MAIN FRAME #97A0FF
root.configure(background='#2E3799')
buttons_frame = LabelFrame (root, text="Classifier selection", font=(font_type, font_size),                       bg=frame_background)
input_frame = LabelFrame   (root, text="Write an email",       font=(font_type, font_size), width=100, height=40, bg=frame_background)
accuracy_frame = LabelFrame(root, text="Accuracy stats",       font=(font_type, font_size), width=100, height=50, bg=frame_background)
matrix_frame = LabelFrame  (root, text="Confusion matrix",     font=(font_type, font_size), width=100, height=50, bg=frame_background)

buttons_frame.grid( row = 0, column = 0,  sticky="NESW", padx = 10, pady = 10)
input_frame.grid(   row = 1 , column = 0, sticky="NESW", padx = 10, pady = 10)
accuracy_frame.grid(row = 0, column = 1,  sticky="NESW", padx = 10, pady = 10)
matrix_frame.grid(  row = 1, column = 1,  sticky="NESW", padx = 10, pady = 10)

# Botones para cada modelo
models_buttons = []

for index in range(0, len(names)):
  models_buttons.append(Button(buttons_frame, text=names[index], font=button_font, width=20, command=lambda i=index: activate_model(i)))
  models_buttons[index].grid(row = index, column = 1, pady = 10)

# Etiqueta que muestra si el clasificador está activado
for index in range(0, len(models_activated)):
  status_labels.append(Label(buttons_frame, text = "X", fg="red", font=button_font, width = 5, bg=frame_background, relief=GROOVE))
  status_labels[index].grid(row = index, column = 0)

# Funcion para (des)activar los modelos a emplear
def activate_model(index):  
  if (models_activated[index]):
    status_labels[index].config(text="X", fg="red")
  else:
    status_labels[index].config(text="\u2713", fg="green")
  models_activated[index] = not models_activated[index]

# Etiqueta donde se muestra el resultado de la predicción para cada modelo
prediction_results = []
prediction_labels = []
label = True
i = 0
for index in range(0, len(models_activated)):
  i = index
  prediction_labels.append(Label(buttons_frame, text = "Prediction:", font=button_font, bg=frame_background))
  prediction_labels[index].grid(row = i, column = 2, padx = 5)
  prediction_results.append(Label(buttons_frame, text = "-", font=button_font, bg=frame_background))
  prediction_results[index].grid(row = i, column = 3, padx = 5)

# Campo para escribir un email
input_scrollbar = Scrollbar(input_frame, orient='vertical')
input_scrollbar.grid(row = 0 , column = 0, rowspan=1)
input_field = Text(input_frame, width = 40, height = 20, bd = 4, yscrollcommand=input_scrollbar.set)

input_scrollbar.config(command=input_field.yview)
input_field.grid(row = 0, column = 1, pady = 10)

# Botón para ejecutar la predicción
input_button = Button(input_frame, text="Verify", font=(font_type, font_size), bg="red", fg="white", command=lambda: click_verify(input_field.get("1.0",END)))
input_button.grid(row = 1, column = 1)

def click_verify(text):
  test = Vectorizer.transform([text])

  for index in range(0, len(models_activated)):
    if(models_activated[index]):
      actual_model = classifiers[index]
      #actual_model.fit(count, targets)
      y_predict = actual_model.predict(test.astype('float32'))
      prediction_results[index].config(text = y_predict)
    else:
      prediction_results[index].config(text = "-")

# Insercion de grafica
accuracy_stats = ImageTk.PhotoImage(Image.open('./images/Accuracy.jpg').resize((450, 300), Image.Resampling.LANCZOS), master=root)

accuracy_label = Label(accuracy_frame, image=accuracy_stats)
accuracy_label.grid(row = 0, column = 0, padx = 10, pady = 10)

# Visualizador de matrices de confusion
global matrix_label
global matrix_all_images
global actual_image_name
global image_counter

matrix_image = ImageTk.PhotoImage(Image.open('./images/' + names[0] + '.jpg').resize((450, 300), Image.Resampling.LANCZOS), master=root)

matrix_label = Label(matrix_frame, image=matrix_image)
matrix_label.grid(row = 0, column = 0, columnspan = 3, padx = 10, pady = 40)

matrix_all_images = []
image_counter = 0
for index in range(0, len(names)):    
  matrix_all_images.append(ImageTk.PhotoImage(Image.open('./images/' + names[index] + '.jpg').resize((450, 300), Image.Resampling.LANCZOS), master=root))

# Botones para mostrar distintas matrices de confusion
actual_image_name = Label(matrix_frame, text=names[image_counter], width=20, font=button_font, bg=frame_background, relief=GROOVE)
actual_image_name.grid(row = 1, column = 1, sticky="N")

back_button = Button(matrix_frame, font=button_font, bg="gray", text="<<", command=lambda: show_next(False))
next_button = Button(matrix_frame, font=button_font, bg="gray", text=">>", command=lambda: show_next(True))
back_button.grid(row = 1, column = 0, sticky="N")
next_button.grid(row = 1, column = 2, sticky="N")

def show_next(next):
  global image_counter
  global matrix_all_images
  global actual_image_name

  if (next):
    image_counter = image_counter + 1
  else:
    image_counter = image_counter - 1
  
  if (image_counter < 0): image_counter = len(matrix_all_images)  - 1
  if (image_counter > len(matrix_all_images) - 1): image_counter = 0

  matrix_label.config(image=matrix_all_images[image_counter])
  actual_image_name.config(text=names[image_counter])

root.mainloop()