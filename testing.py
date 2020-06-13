import os
import tensorflow as tf

workingDirectory = os.getcwd()
workingDirectory = os.path.dirname(os.path.realpath(__file__))
print(workingDirectory)


tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
