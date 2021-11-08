from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

# Define a sequential model here
model = models.Sequential()

# add more layers to the model...
#convolutional layer
model.add(layers.Conv2D(
    filters = 16,
    kernel_size = 4,
    activation = 'relu', # Don't change
    input_shape = (150, 150, 3) # Don't change
))

#maxpooling layer
model.add(layers.MaxPooling2D(
    pool_size = (3, 3), #tuple/list
    padding = "same"
))

#convolutional layer
model.add(layers.Conv2D(
    filters = 16,
    kernel_size = 9,
    activation = 'relu', # Don't change
    input_shape = (150, 150, 3) # Don't change
))

#maxpooling layer
model.add(layers.MaxPooling2D(
    pool_size = (2, 2), #tuple/list
    padding = "same"
))

#convolutional layer
model.add(layers.Conv2D(
    filters = 16,
    kernel_size = 9,
    activation = 'relu', # Don't change
    input_shape = (150, 150, 3) # Don't change
))

#maxpooling layer
model.add(layers.MaxPooling2D(
    pool_size = (2, 2), #tuple/list
    padding = "same"
))

#flatten layer
model.add(layers.Flatten())

#hidden densly connected layer
model.add(layers.Dense(16, activation = 'relu')) # Don't change activation

#final densely connected layer
model.add(layers.Dense(1, activation = 'sigmoid')) # Don't change activation or units

model.summary()

# Then, call model.compile()
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)

# Finally, train this compiled model by running:
# python train.py