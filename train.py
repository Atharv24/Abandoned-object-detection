import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

def make_model(seq_len=15, lstm_hidden=512, lr=1e-4):
    base_model = keras.applications.MobileNetV2(input_shape=(512, 512, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    average_layer = keras.layers.GlobalAveragePooling2D()

    feature_extractor = keras.models.Sequential([base_model, average_layer])
    lstm_layer = keras.layers.LSTM(lstm_hidden, input_shape=(10, 1280), return_sequences=False, return_state=False)
    dense = keras.layers.Dense(1)

    lstm = keras.models.Sequential([lstm_layer, dense])

    inp = keras.Input(shape=(512, 512, 3))
    x = feature_extractor(inp)
    x = keras.layers.Reshape(target_shape=(1, seq_len, 1280))
    x = lstm(x)

    model = keras.Model(inputs=[inp], outputs=[x])
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['sparse_categorical_accuracy'])

    return model

#dataset = np.load('data/dataset.npy')
#labels = np.load('data/labels.npy')

#train_data = tf.data.Dataset.from_tensor_slices((dataset, labels))

def train_step(model, batch):
    with tf.GradientTape() as tape:
        features = fe(batch)
        out = lstm(batch.reshape([tf.newaxis, 15, 1280]))
    return out
model = make_model()
print(model.summary())

