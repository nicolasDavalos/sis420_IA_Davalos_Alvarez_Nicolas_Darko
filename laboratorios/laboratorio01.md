# regrecion lineal multivariable
aqui se añadira el laboratorio 1 de la materia sis420
* la url del dataset es: https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset
* Este conjunto de datos se recopila del Servicio Nacional de Seguro de Salud en Corea. Toda la información personal y los datos sensibles fueron excluidos.
* sus variables son; Sexo, Edad, Altura, Peso, vista izquierda, vista derecha, audicion izquierda, audicion derecha, Presión arterial sistólica, Presión arterial diastolica, glucosa en ayunas, colesterol total, colesterol HDL, colesterol LDL, hemoglobinas, proteina en orina, suero, SGOT, ALT, y-glutamilo transpeptidasa, estado de fumador, bebedor o no, y por ultimo trigliceridos
* de los cuales descartamos sexo, bevedor o no
* nuestra y a obtener o predecir es trigliceridos
## librerias del codigo
podemos ver las libreiras que se descargaron para que los siguientes codigos que se vayan a escribir puedan funcionar, igual se añaden unas librerias mas
```bash
# utilizado para manejos de directorios y rutas
import os

# Computacion vectorial y cientifica para python
import numpy as np

# Librerias para graficación (trazado de gráficos)
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # Necesario para graficar superficies 3D

# llama a matplotlib a embeber graficas dentro de los cuadernillos
%matplotlib inline
```
# vinculo hacia google drive
```
from google.colab import drive
drive.mount("/content/gdrive")
```
# carga del dataset usado para este ejerciocio
en esta parte del ejercicio se cargan los datos de m filas y n columnas del dataset, ademas se "separaron" columnas inesesarias para la regresion lineal como los datos no numericos, luego se representan a las variables independientes como la dependiente, separando la dependiente de las demas y asignanodles valoes para el codigo, luego se imprimieron algunsa de las muchas filas del dataset, su m y n
```
# Librería para manejo de datos
import pandas as pd
import numpy as np

# Cargar datos desde Google Drive
data = pd.read_csv('/content/gdrive/MyDrive/IA/smoking_driking_dataset_Ver01.csv/smoking_driking_dataset_Ver01.csv')

# Eliminar columnas no numéricas
data = data.drop(columns=['sex', 'DRK_YN'])

# Variable dependiente (lo que queremos predecir)
y = data['triglyceride'].values

# Variables independientes (todas las demás)
X = data.drop(columns=['triglyceride']).values

# Número de ejemplos
m = y.size

print("Número de ejemplos:", m)

# Mostrar dimensiones del dataset
print("Dimensión de X:", X.shape)
print("Dimensión de y:", y.shape)

print("\nPrimeros 10 ejemplos del dataset:\n")

# Mostrar primeras 10 filas completas
print(data.head(10))
```
# normalizacion de las variables
se normalizan las variables del dataset con el siguiente codigo para que puedan tener mejor comparacion, esto lo hace en la copia del dataset para que no afecte a los valores originales

```
def  featureNormalize(X):

    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma
```

# llama a la funcion de normalizacion
este codigo llama a la funcion de normalizacion para que se aplique y tambien muestra los siguientes datos:
x_norm - matriz de datos normalizados;
mu - vector de la media de cada variable;
sigma - vector con desviacion estandar de cada variable
```
# llama featureNormalize con los datos cargados
X_norm, mu, sigma = featureNormalize(X)

print(X[:5])
print('Media calculada:', mu)
print('Desviación estandar calculada:', sigma)
print(X_norm[:5])
```
# columna de 1
se añade una columa de 1 en todos los datos del dataset para que la formula que se use se unifique y no sean 2 sino 1 formula
```
# Añade el termino de interseccion a X
# (Columna de unos para X0)
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)
```
# ver la nueva matriz de datos
para saber que se haya cargado la columna de 1
```
print(X[:5])
```

# funcion de costo
en esta parte del codigo calcula que tan bien o mas esta el modelo, esta funcion mide el error del modelo

```
def computeCostMulti(X, y, theta):
    # Inicializa algunos valores utiles
    m = y.shape[0] # numero de ejemplos de entrenamiento

    J = 0

    # h = np.dot(X, theta)

    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))

    return J
```

# descenso por gradiente
se aplicara la formula de descenso por gradiete que es una de las que estamos viendo, esta ajusta los valores de theta para minimizar el error del modelo

```
def gradientDescentMulti(X, y, theta, alpha, num_iters):

    # Inicializa algunos valores
    m = y.shape[0] # numero de ejemplos de entrenamiento

    # realiza una copia de theta, el cual será acutalizada por el descenso por el gradiente

    theta = theta.copy()

    J_history = []

    for i in range(num_iters):
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history
```
# seleccion de coeficientes de aprendizaje
aqui se hacen varias cosas.
* hace una grafica que muestra cómo converge el error y usa el modelo para predecir un precio
* indica de que tamaño seran los "pasos" que de en el aprendizaje, es decir de que forma aprendera
* se decide el numero de iteracciones que tendra el modelo
* toma unos datos de ejemplo donde se introdcen las variables independientes para encontrar la variable dependiente que se desea

```
# Elegir algun valor para alpha
alpha = 0.01
num_iters = 200

# inicializa theta
theta = np.zeros(X.shape[1])

# ejecutar descenso por gradiente
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Grafica la convergencia del costo
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Numero de iteraciones')
pyplot.ylabel('Costo J')

# Mostrar parámetros
print('Theta calculado por el descenso por el gradiente:')
print(theta)

# -------------------------------
# EJEMPLO DE PREDICCIÓN
# -------------------------------

# valores ejemplo
X_example = np.array([
45,   # age
170,  # height
75,   # weight
85,   # waistline
1.0,  # sight_left
1.0,  # sight_right
1,    # hear_left
1,    # hear_right
120,  # SBP
80,   # DBP
95,   # BLDS
190,  # tot_chole
55,   # HDL
120,  # LDL
15,   # hemoglobin
1,    # urine_protein
1.0,  # serum_creatinine
25,   # SGOT_AST
30,   # SGOT_ALT
35,   # gamma_GTP
1     # SMK_stat_type_cd
])

# normalizar (solo variables reales)
X_example_norm = (X_example - mu) / sigma

# añadir termino de intersección
X_example_final = np.concatenate(([1], X_example_norm))

# predicción
pred_triglycerides = np.dot(X_example_final, theta)

print("Predicción de triglicéridos:")
print(pred_triglycerides)
```

# ecuacion de la normal
se hizo en un mismo cuaderno de colab asi que por precaucion y evitar un choque de variables nombradas, se nombro nuevas variables.
ecuación de la normal, otro metodo o forma de calcula la regresión lineal del modelo, carga el dataset, separa las variables independientes y de la objetivo, agrega la columna de 1 para x0, y ahora usa el etodo matematico de la ecuacion de la normal, antes usabas el descenso de la gradiente para luego calcular theta, ahora la calculas directamente con algebra matricial, tambien no se requiere normalizacion, al tener un dataset tan grande para poder ver como funciona la ecucacion de la normal reduciremos su numero de m con las que trabajara, para que los calculos matriciales se puedan ejecutar en colab

```
# Librerías
import pandas as pd
import numpy as np

# Cargar dataset
data2 = pd.read_csv('/content/gdrive/MyDrive/IA/smoking_driking_dataset_Ver01.csv/smoking_driking_dataset_Ver01.csv')

# Tomar 30000 filas aleatorias
data2 = data2.sample(n=30000, random_state=42)

# Variable dependiente (lo que queremos predecir)
y2 = data2['triglyceride'].values

# Variables independientes (quitando columnas que no se usan)
X2 = data2.drop(columns=['triglyceride','sex','DRK_YN']).values

# Número de ejemplos
m2 = y2.size
print("Número de ejemplos usados:", m2)

# Añadir columna de intersección
X2 = np.concatenate([np.ones((m2, 1)), X2], axis=1)

print("Shape de X:", X2.shape)
```

# formula de la ecuacion de la normal
se tuvo que ajustar tambien para el ejercicio. 
* se nos mostrara la forma en la que se realizaba (aprendida en clase pero no ejecutda aqui)
```
def normalEqn(X, y):

    theta = np.zeros(X.shape[1])

    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)

    return theta
```
* encontrada con ayuda de la IA
```
def normalEqn(X, y):
    # Usar pseudo-inversa para estabilidad y rapidez
    theta = np.linalg.pinv(X) @ y
    return theta
```

# ejemplo de la normal
en este codigo se nos muestra todas las theta del modelo que es igual al numero de n, y un ejemplo de la normal

```
# Calcula los parámetros con ecuación normal
theta = normalEqn(X2, y2)

# Mostrar parámetros
print('Theta calculado a partir de la ecuación de la normal:')
print(theta)
print("Cantidad de parámetros en theta:", len(theta))

# -----------------------------
# EJEMPLO DE PREDICCIÓN
# -----------------------------

X_array = np.array([
1,      # término de intersección
45,     # age
170,    # height
75,     # weight
85,     # waistline
1.0,    # sight_left
1.0,    # sight_right
1,      # hear_left
1,      # hear_right
120,    # SBP
80,     # DBP
95,     # BLDS
190,    # tot_chole
55,     # HDL_chole
120,    # LDL_chole
15,     # hemoglobin
1,      # urine_protein
1.0,    # serum_creatinine
25,     # SGOT_AST
30,     # SGOT_ALT
35,     # gamma_GTP
1       # SMK_stat_type_cd
])

# predicción
pred = np.dot(X_array, theta)

print("Predicción de triglicéridos con ecuación normal:")
print(pred)
```
