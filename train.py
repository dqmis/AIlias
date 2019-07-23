import os
from glob import glob
import ast
import numpy as np 
import pandas as pd
from PIL import Image, ImageDraw 
from tqdm import tqdm
from dask import bag

from tensorflow import keras
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras import backend as K

# getting train files and classnames
classfiles = os.listdir('../data/unzip/')
numstonames = {i: v[:-4].replace(" ", "_") for i, v in enumerate(classfiles)}
class_paths = glob('../data/unzip/*.csv')

# constants for the traning
IM_PER_CLASS = 13000 # images count per class
IM_HEIGHT, IM_WIDTH = 75, 75 # size of image
CLASSES_COUNT = 40 # number of classes to train
EPOCHS = 30
BATCH_SIZE = 2500

# function that draws doodle from csv data
def draw_it(strokes):
	image = Image.new("P", (256,256), color=255)
	image_draw = ImageDraw.Draw(image)
	for stroke in ast.literal_eval(strokes):
		for i in range(len(stroke[0])-1):
			image_draw.line([stroke[0][i], 
							 stroke[1][i],
							 stroke[0][i+1], 
							 stroke[1][i+1]],
							fill=0, width=5)
	image = image.resize((IM_HEIGHT, IM_WIDTH)).convert('RGB')
	return np.array(image)/255.

# function that return traning and validation data
def get_data():
	train_grand = []

	# drawing and storing images to the np.array
	for i,c in enumerate(tqdm(class_paths[0: CLASSES_COUNT])):
		train = pd.read_csv(c, usecols=['drawing', 'recognized'], nrows=IM_PER_CLASS*5//4)
		train = train[train.recognized == True].head(IM_PER_CLASS)
		imagebag = bag.from_sequence(train.drawing.values).map(draw_it) 
		trainarray = np.array(imagebag.compute())
		trainarray = np.reshape(trainarray, (IM_PER_CLASS, -1))	
		labelarray = np.full((train.shape[0], 1), i)
		trainarray = np.concatenate((labelarray, trainarray), axis=1)
		train_grand.append(trainarray)
	train_grand = np.array([train_grand.pop() for i in np.arange(CLASSES_COUNT)])
	train_grand = train_grand.reshape((-1, (3*IM_HEIGHT*IM_WIDTH+1)))

	# deleting to save memory
	del trainarray
	del train

	# validation data split fraction
	valfrac = 0.1
	cutpt = int(valfrac * train_grand.shape[0])

	# splitting the data
	np.random.shuffle(train_grand)
	y_train, X_train = train_grand[cutpt: , 0], train_grand[cutpt: , 1:]
	y_val, X_val = train_grand[0:cutpt, 0], train_grand[0:cutpt, 1:]

	# deleting to save memory
	del train_grand

	# reshaping arrays to img_count, size, size, channels format
	y_train = keras.utils.to_categorical(y_train, CLASSES_COUNT)
	x_train = X_train.reshape(X_train.shape[0], IM_HEIGHT, IM_WIDTH, 3)
	y_val = keras.utils.to_categorical(y_val, CLASSES_COUNT)
	x_val = X_val.reshape(X_val.shape[0], IM_HEIGHT, IM_WIDTH, 3)

	return x_train, x_val, y_train, y_val

def main():
	# getting np arrays of data
	x_train, x_val, y_train, y_val = get_data()

	# constructing the model
	model = Sequential()
	model.add(InceptionResNetV2(
		include_top=False,
		weights='imagenet',
		input_shape=(75,75,3),
		pooling='avg',
		classes=CLASSES_COUNT
	))
	model.add(Flatten())
	model.add(BatchNormalization(trainable=False))
	model.add(Dense(1024, activation='relu'))
	model.add(BatchNormalization(trainable=False))
	model.add(Dense(564, activation='relu'))
	model.add(BatchNormalization(trainable=False))
	model.add(Dense(CLASSES_COUNT, activation='softmax'))

	# freezing top ResNet layers
	model.layers[0].trainable = False
	K.set_learning_phase(1)

	# compiling and fitting the model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	model.fit(X_train, y_train, epochs=30, validation_split=0.13, batch_size=2500)

	# evaluating and saving the model
	model.evaluate(X_val, y_val)
	model.save('doodle_model.h5')

if __name__ == '__main__':
	main()
