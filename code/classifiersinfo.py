# ********************************************************
# *************   CREACION DE IMAGENES   *****************
# ********************************************************
from spamdetection import (plt, names, classifiers, x_test, y, y_test, y_train, count,
                        Vectorizer, classification_report, accuracy_score,
                        confusion_matrix, ConfusionMatrixDisplay)
y_axis = []
counter = 0

for claf in classifiers:
  clf = claf
  targets = y_train.values

  # Aprendizaje
  clf.fit(count, targets)
  
  # Resultados de la prediccion sobre el dataset
  y_predict = clf.predict(Vectorizer.transform(x_test))
  
  # Matriz de confusion del clasificador entrenado
  c_matrix = confusion_matrix(y_test, y_predict, normalize = 'all') 
  c_matrix_display = ConfusionMatrixDisplay(c_matrix, display_labels = ['ham', 'spam'])
  c_matrix_display.plot()

  # Creacion de imagen    
  plt.savefig('./images/' + names[counter] + '.jpg')

  # Precision del clasificador entrenado 
  y_axis.append(accuracy_score(y_test, y_predict) * 100)

  # Reporte de clasificacion del clasificador entrenado
  report = classification_report(y_test, y_predict, output_dict = True)
  
  counter = counter + 1
  plt.clf()

# Creación de gráfica. (Accuracy score)
colors = ['blue', 'maroon', 'yellow', 'red', 'green', 'pink', 'purple']
plt.bar(names, y_axis, color = colors, width = 0.4)
plt.title('Accuracy scores of each classifier')
plt.xlabel('Classifiers')
plt.xticks(fontsize = 5, rotation = 30)
plt.ylabel('Accuracy score')
plt.savefig('./images/Accuracy.jpg')