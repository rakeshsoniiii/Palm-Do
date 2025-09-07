# palm_model.py
import tensorflow as tf

IMG_SIZE = 128
EMBED_DIM = 128

def build_embedding_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), embed_dim=EMBED_DIM):
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )
    base.trainable = False  # freeze backbone

    inp = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inp * 255.0)
    x = base(x, training=False)
    x = tf.keras.layers.Dense(embed_dim, use_bias=False)(x)
    x = tf.keras.layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)
    model = tf.keras.Model(inputs=inp, outputs=x, name="palm_embedder")
    return model
