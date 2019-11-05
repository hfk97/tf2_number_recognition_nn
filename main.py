import sys
import importlib
import subprocess

# function that imports a library if it is installed, else installs it and then imports it
def getpack(package):
    try:
        return (importlib.import_module(package))
        # import package
    except ImportError:
        subprocess.call([sys.executable, "-m", "pip", "install", package])
        return (importlib.import_module(package))
        # import package

time=getpack("time")
tf=getpack("tensorflow")
from tensorflow.keras.callbacks import TensorBoard
from tensorboard import program


Name=f"Number_eval_{int(time.time())}" #name for logging
tensorboard=TensorBoard(log_dir=f"logs/{Name}") #setup tensorboard for visual inspection


subprocess.call('rm -rf ./logs/*', shell=True) #remove old logs


#This code works with the mnist dataset included in tf (it can be found here: http://yann.lecun.com/exdb/mnist/)
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


#"pixels" in x_train and x_test have a value between 0-255 we want values between 0-1 for our network
x_train, x_test = x_train / 255.0, x_test / 255.0



#sequential model is chosen because we have one input source one output source and do not have to reuse layers

model = tf.keras.models.Sequential([ #model with 4 layers
  tf.keras.layers.Flatten(input_shape=(28, 28)), #flatten the input array
  tf.keras.layers.Dense(128, activation='relu'), #size of output space = 128 selected through grid search; activation via rectified linear unit (returns max(x,0))
  tf.keras.layers.Dropout(0.2), #dropout layer randomly sets 20% of input units to 0 (to avoid overfitting)
  tf.keras.layers.Dense(10, activation='softmax') #outputspace = 10 (0-9); activation through highest softmax (see https://en.wikipedia.org/wiki/Softmax_function)
])

model.compile(optimizer='adam', #adam combines RMSprop and Momentum for gradient descent and is state of the art (more info https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
              loss='sparse_categorical_crossentropy', #allowse for multi class classification without one hot encoding
              metrics=['accuracy']) #the metric we want to optimize


model.fit(x_train, y_train, epochs=7, callbacks=[tensorboard])#input,output,epochs; if we hadn't specified test and training sets we would use validation_split parameter here

model.evaluate(x_test,  y_test, verbose=2)#verbose specifies output



tb = program.TensorBoard() #program module is used to launch tensorboard (see https://github.com/tensorflow/tensorboard/blob/master/tensorboard/program.py)
tb.configure(argv=[None, '--logdir', "logs"])
url = tb.launch()

input("Press enter to end the program and close the tensorboard session at: " + url) #follow this link to see the tensorboard