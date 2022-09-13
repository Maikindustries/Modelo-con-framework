#Miguel Ángel Pérez López A01750145
#Regresión Logística
#Tarea 
#Momento de Retroalimentación: Módulo 2 Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución. (Portafolio Implementación)

#Librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

#Cargar datos
dataset = pd.read_csv('winequality_red.csv')

#Hacer dos pruebas variando los datasets de train y test
resultados = []
test_sizes = [0.2, 0.3]
for test_size in test_sizes:
  print(f"Model with test size of {test_size*100}%")
  #Obtenemos su valores para quitar la columna index y el nombre de las columnas
  X = dataset.iloc[:, :-1].values
  y = dataset.iloc[:, -1].values

  #Separar datos
  #En test_size asignamos que solo sea el 20% del dataset
  #Para que las predicciones de abajo sean ciertas, asignamos un random_state para 
  #que la separación de datos aleatoria sea la misma siempre que se corra
  X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                      test_size=test_size)

  #Estandarizar
  #Esta clase estandariza las características eliminando 
  #la media (u) y escalando a la varianza (s) de la unidad.
  #La fórmula que utiliza esta clase es la siguiente -> z = (x - u) / s
  standard_scaler = StandardScaler()
  #Estandarizamos la X_train y X_test para que el modelo funcione mejor

  #Checar diferencia
  X_train = standard_scaler.fit_transform(X_train)
  X_test = standard_scaler.transform(X_test)

  #Logistic Regression
  log_reg = LogisticRegression(max_iter=1000)

  #Pasamos la x y y de entrenamiento para entrenar el modelo
  log_reg.fit(X_train, y_train)

  #Una vez entrenado el modelo, utilizamos x de prueba para predecir la y
  y_pred = log_reg.predict(X_test)

  #Calculamos la matriz de confusión para ver los falsos positivos y falsos negativos
  confusion_m = confusion_matrix(y_test, y_pred)
  print("Confusion Matrix")
  print(confusion_m)
  #Calculamos su accuracy para obtener la fracción de muestras clasificadas correctamente
  print("Model accuracy_score: ", accuracy_score(y_test, y_pred))

  #Calculamos la precisión es la relación tp / (tp + fp)
  #donde tp es el número de verdaderos positivos y fp el número de falsos positivos.
  print("Model precision_score: ", precision_score(y_test, y_pred))


  #Obtenemos un modelo con una precisión de más del 70% que se considera bueno
  #ya que ninguna vida depende de si se clasifico bien la clase de un vino


  #Predicciones para demostrar la efectividad del modelo
  prediction = log_reg.predict(standard_scaler.transform([[6.7, .58,  .08,
                                                           1.8, .097, 15., 
                                                           65., .9959, 3.28,
                                                           .54, 9.2]]))
  print("Prediction 1: ", list(prediction) == [0])

  prediction = log_reg.predict(standard_scaler.transform([[7.8, .53, .04, 1.7,
                                                           .076, 17., 31.,
                                                           .9964, 3.33, .56, 
                                                           10.]]))
  print("Prediction 2: ", list(prediction) == [1])
  print()
  resultados.append([precision_score(y_test, y_pred), 
                     accuracy_score(y_test, y_pred)])


#Comparación de resultados en con 20% y 30%
if resultados[0][0] > resultados[1][0]:
  print("El modelo con un 20% de test size tuvo un accuracy score más alto")
else:
  print("El modelo con un 30% de test size tuvo un accuracy score más alto")
  
if resultados[0][1] > resultados[1][1]:
  print("El modelo con un 20% de test size tuvo un precission score más alto")
else:
  print("El modelo con un 30% de test size tuvo un precission score más alto")
