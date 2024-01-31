"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random #<--- Importamos la libreria random que nos ayudará para \\
              #<--- obtener nùmeros pseudo-aleatorios

# Third-party libraries
import numpy as np #<--- Importamos la libreris Numpy y para poder acceder \\
                   #<--- a sus funciones la llamaremos como funciòn.np 

class Network(object): #<--- Creamos la clase llamda "Network" y hereda la clase
                       #<--- base de python "Object", con esta clase vamos a \\
                       #<--- crear diferentes instancias.

    def __init__(self, sizes): #<---En esta parte recibe una lista que va a   \\
                               #<---ser propia de la instancia, es decir      \\
                               #<--- podemos llamar la clase para una red con \\
                               #<--- [2,3,4] dos neuronas de entrada, 3 de la \\
                               #<--- intermedia y 4 de la capa de salida, y   \\
                               #<--- podemos llamar de nuevo a la clase con   \\
                               #<--- una lista diferente. 
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes) #<--- Almacenamos el nùmero de elementos \\
                                     #<--- de la lista 'sizes' en la variable \\
                                     #<--- self.num_layers, es decir, este    \\
                                     #<--- valor es EL NUMERO DE CAPAS de la  \\
                                     #<--- red.
        self.sizes = sizes #<--- Guardamos la lista 'sizes' para cada         \\
                           #<--- instancia creada, podremos acceder a ella. 

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #<---        \\
        #<--- Guardamos en 'self.biases' una lista de matrices en donde las   \\
        #<--- matrices tendràn dimensiones de 'y' filas y 1 columna. 'y' va a \\
        #<--- tener los valores de la cantidad de neuronas que existan en las \\
        #<--- capas intermedias y la final, ya que se omite la capa de entrada\\
        #<--- porque la capa de entrada no se les asigna un bias o sesgo, de  \\
        #<--- hecho tiene sentido, cada capa tendrà un nùmero de neuronas y   \\
        #<--- esas neuronas tendràn asociado un sesgo. Los elementos de las   \\
        #<--- matrices son numeros aleatorios entre 0 y 1.                    \\


        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])] #<--- Estamos \\
        #<--- guardando en 'self.weights' una lista de matrices de con 'y'    \\
        #<--- numero de filas y 'x' numero de columnas. Los valores que puede \\
        #<--- tener 'x' son el número de neuronas de todas las capas excepto  \\
        #<--- la de salida, y los que puede tener 'y' son el numero de        \\
        #<--- neuronas de todas las capas excepto la de entrada.              \\
        #<--- Todos los elementos de las matrices son numeros aleatorios que \\
        #<--- van de 0 a 1                                                    \\
    def feedforward(self, a): #<--- Se define la función que recibe los datos \\
        #<--- de entrada y devuelve la activación, que es el resultado de     \\
        #<--- evaluar la sigmoide en el resultado de hacer el producto entre  \\
        #<--- la activación y el peso y sumamois el bias.                     \\
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights): #<--- Asigna a las        \\
            #<--- variables "b" y "w" una matriz de las lista de matrices de  \\
            #<--- 'self.biases' y 'self.weights'\\
            a = sigmoid(np.dot(w, a)+b) #<--- Hace el producto punto de       \\
            #<--- matrices 'w' y 'a' y suma el vector bias 'b'                \\
            #<--- y el vector lo evalua en la función de activación sigmoide  \\ 
        return a #Regresa el vector de las activaciones

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None): #<--- Definimos la función del stochastic gradient \\
        #<--- descent, tenemos las variables de 'training_data', 'epochs',      \\
        #<--- 'mini_batch', 'eta'\\
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: #<--- Si la variable 'test_data' tiene algún valor, se    \\
            #<--- va a ejecutar lo que está dentro del condicional, estos son   \\
            #<--- los datos de prueba \\

            test_data = list(test_data) #<--- Convierte 'test_data' en una lista\\
            n_test = len(test_data) #<--- Nos da el número de datos de prueba   \\

        training_data = list(training_data) #<--- Convierte en una lista la     \\
        #<--- información que contenga la variable 'training_data', son los     \\
        #<--- datos de entrenamiento\\

        n = len(training_data)#<--- Nos da el numero de datos de entrenamiento  \\

        for j in range(epochs): #<--- Ejecutará este codigo dependiendo del     \\
            #<--- número de epocas que escogamos \\

            random.shuffle(training_data) #<--- Se reorganizan los datos de     \\
            #<--- forma aleatoria, para evitar sesgos                           \\

            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]#<--- Creación de los mini \\
            #<--- batches, guardamos en la variable 'mini_batches' una lista que  \\
            #<--- toma un pedazo de la lista de los datos de entrenamiento, es decir\\
            #<--- de 'training_data', este pedazo de las lista, es la que está en  \\
            #<--- la posición "k" hasta esa posición más 'mini_batch_size', el cual\\
            #<--- establece el tamaño del mini batch y la posición de 'k' varia  \\
            #<--- del primer elemento de la lista hasta el último 'n' es el numero\\
            #<--- de datos de entrenamiento y da saltos dependiendo el tamaño del \\
            #<--- mini batch.\\

            for mini_batch in mini_batches: #<--- La variable 'mini_batch' tomará\\
                #<--- los valores de los mini batches ya definidos, es decir vamos\\
                #<--- a operar cada uno de los mini batches\\

                self.update_mini_batch(mini_batch, eta) #<--- Ejecutamos la función\\
                #<--- definida más abajo, aquí aplicamos la función 'update_mini_batch'\\
                #<--- a cada uno de los mini batches creados e ingresamos el learning \\
                #<--- rate o eta, el cuál nos definirá el tamaño de los pasos que demos\\
                #<---para descender al minimo de la función de costo.              \\

            if test_data: #<--- Si hay datos de prueba, se ejecutará este codigo   \\

                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)) #<--- Vamos a imprimir en \\
                #<--- pantalla el número de epoca en la que estamos, el cual es la  \\
                #<--- variable 'j' ya que corre en el numero de epocas, en el ciclo \\
                #<--- for, muestra lo que arroja la función 'self.evaluate(test_data)'\\
                #<--- la cual está definida más abajo, en general muestra el rendimiento\\
                #<--- de la red neuronal, es decir cuantos acertó dividido por 'n_test'\\
                #<--- el cual es el tamaño de los datos de prueba                    \\

            else: #<--- Si no hay datos de prueba se ejecutará esta parte del codigo\\
                print("Epoch {0} complete".format(j)) #<--- Imprime el cuando se    \\
                #<--- haya completado una epoca e indica el número de epoca que se  \\
                #<--- ha concluido, estamos hablando de epocas de entrenamiento  \\

    def update_mini_batch(self, mini_batch, eta): #<--- Estamos creando una función  \\
        #<--- que recibe al mini batch y el learning rate  \\
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] #<--- Creamos una lista de \\
        #<--- matrices, y cada matriz tendrá la misma forma que las contenidas en la  \\
        #<--- lista de matrices de los bias, y los elementos será cero\\

        nabla_w = [np.zeros(w.shape) for w in self.weights] #<--- Creamos una lista de\\
        #<--- matrices con las mismas dimensiones que las contenidas en la lista de   \\
        #<--- los pesos, los elementos de estas matrices son cero.                    \\

        for x, y in mini_batch: #<--- Iteramos sobre los elementos del mini_batch, en \\
            #<--- donde 'x' es el valor de entrada y 'y' es el valor esperado\\

            delta_nabla_b, delta_nabla_w = self.backprop(x, y) #<--- Se ejecuta la función\\
            #<--- de backpropagation definida más abajo y se guardan en las variables \\
            #<--- 'delta_nabla_b','delta_nabla_w', se guardan los gradientes de los   \\
            #<--- sesgos y de los pesos\\

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #<---Actualizamos\\
            #<--- el valor de los gradientes de los bias, sumandolos con el calculado \\

            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]#<--- De la  \\
            #<--- misma manera, actualizamos el valor de los gradientes de los pesos  \\
            #<--- sumamos los que fueron calculados\\

        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]#<--- Vamos actualizar \\
        #<--- los valores de los pesos, incialmente erán aleatorios, ahora vamos a buscar\\
        #<--- los pesos que hagan minima la función de costo. Esta actualización va a ser\\
        #<--- a cada elemento de los pesos, se les va a restar el learning rate dividido\\
        #<--- por el numero de elmentos del mini batch multiplicado por el nuevo valor  \\
        #<--- del gradiente de los pesos\\

        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]#<--- Vamos actualizar \\
        #<--- los valores de los bias, incialmente erán aleatorios, ahora vamos a buscar\\
        #<--- los bias que hagan minima la función de costo. Esta actualización va a ser\\
        #<--- a cada elemento de los bias, se les va a restar el learning rate dividido\\
        #<--- por el numero de elmentos del mini batch multiplicado por el nuevo valor  \\
        #<--- del gradiente de los bias\\

    def backprop(self, x, y): #<--- Definidmos la función del backpropagation el cual\\
        #<--- recibe el valor 'x' ques el dato y 'y' es el dato esperado\\
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        nabla_b = [np.zeros(b.shape) for b in self.biases] #Creamos una lista de matrices \\
        #<--- cuyos elementos son cero y las matrices tienen las mismas dimensiones   \\
        #<--- que las matrices de los bias\\

        nabla_w = [np.zeros(w.shape) for w in self.weights] #<--- Creamos una lista\\
        #<--- de matrices cuyos elementos son cero y tienen las mismas dimensiones \\
        #<--- que las matrices de los pesos\\

        # feedforward Se realiza la propagación hacia adelante 
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer


        for b, w in zip(self.biases, self.weights): #<--- Tomaremos los valores de \\
            #<--- la lista de matrices de valores aleatorios de los bias y de los  \\
            #<--- pesos, se los vamos a asignar a 'b' y\\

            z = np.dot(w, activation)+b #<--- Hacemos el producto punto de las matrices\\
         #<--- de los pesos y de la activación y sumamos el bias\\   

            zs.append(z) #<--- Agregamos el valor de z a la lista de los vectores z\\
            activation = sigmoid(z) #<--- Calculamos activación para cada z\\
            activations.append(activation) #<--- Agregamos la activación a la lista\\


        # backward pass
            
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1]) #<--- Calcula el error en la capa de salida al \\
        #<--- evaluar la derivada parcial de la función de costo en el último parametro\\
        #<--- y lo multiplica por la derivada de la función de activación y lo evalua\\
        #<--- en el ultimo valor de la capa\\

        nabla_b[-1] = delta #<--- Guarda lo calculado en la última posición de la lista\\
        #<--- de ceros de los gradientes de los bias\\

        nabla_w[-1] = np.dot(delta, activations[-2].transpose())#<--- Guarda en la última\\
        #<--- posición dde la lista de ceros de los gradientes de los pesos, el resultado\\
        #<--- al hacer el producto punto entre la lista 'delta' y la transpuesta de la matriz\\
        #<--- activations de la penultima capa, para garantizar que exista el producto punto\\


        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.


        for l in range(2, self.num_layers): #<--- Se realiza la propagación en las capas\\
            #<--- intermedias de la red, toma desde 2 hasta el numero de capas\\

            z = zs[-l] #<--- inicia en la penultima capa ya que l=-2   \\ 
            #<--- en donde z representa los valores ponderados antes de aplicar   \\
            #<--- la función de activación en la capa actual\\

            sp = sigmoid_prime(z) #<--- Es la derivada de la función de activación sigmoide\\

            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp #<--- Calcula\\
            #<--- el gradiente de los sesgos de la capa actual, haciendo el producto\\
            #<--- punto de la transposición de los pesos y el error de la capa actual\\
            #<--- por la derivada de la función de activación\\

            nabla_b[-l] = delta #<--- Guarda lo calculado en la lista de ceros en la \\
            #<--- posición -l\\

            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())#<--- Calcula\\
            #<--- y asigna el gradiente de los pesos para una capa oculta \\ 
            #<--- específica durante el proceso de retropropagación.\\


        return (nabla_b, nabla_w) #<--- Regresa las listas con los pesos y los bias 

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
