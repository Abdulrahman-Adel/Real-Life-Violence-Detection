import numpy as np
import tensorflow as tf
from tensorflow import keras


class Projection(keras.layers.Layer):
    """
    Custom layer for dividing video frames into patches and implementing linear projection step
    using Tubelet Embedding.

    * Input
    * Conv3D
    * Reshape

    Parameters
    =========

    img_size: int
        Here is assumed that the video hight, width and depth are of the same size

    embed_size: int
        Embedding size.

    patch_size: int
        Size of patch the images with be split into.

    Inputs
    =======

    input shape: shape
        [batch_size, Depth, Hight, Width, Channels]

    Outputs
    =======

    Output shape: shape
        [batch_size, n_patches, embed_size]
    """

    def __init__(self, embed_size, patch_size, img_size, **kwargs):
        super().__init__(**kwargs)
        self.n_patches = 0
        self.patch_size = patch_size
        self.conv = keras.layers.Conv3D(
            filters=embed_size,
            kernel_size=(patch_size, patch_size, 2),
            strides=(patch_size, patch_size, 2),
            padding="VALID",
        )
        self.reshape = keras.layers.Reshape(target_shape=(-1, embed_size))

    def build(self, input_shape):
        bz, d, h, w, c = input_shape
        self.n_patches = (
            d * h * w // self.patch_size**3
        )  # (n_patches) = Hight * width * Depth / (Patch size) ^3

    def call(self, videos):
        x = videos
        x = self.conv(
            x
        )  # shape --> [batch_size, n_patches ** (3/2), n_patches ** (3/2), n_patches ** (3/2), embed_size]

        return self.reshape(x)


class MHA(keras.layers.Layer):
    """
    Class Implementing multi-head self attention

    * head = softmax(q @ k_t // scale) @ v
    * concat(heads)
    * Linear Projection

    parameters
    ==========

    embed_dim: int
        Embedding size.

    n_head: int
        Number of heads of the multi-head self attention.

    Input
    =====

    input shape: shape
        [batch_size, n_patches + 1, embed_size]

    Output
    =====

    output shape: shape
        [batch_size, n_patches + 1, embed_size]
    """

    def __init__(self, embed_size, n_heads, dropout_rate):
        super(MHA, self).__init__()

        self.n_heads = n_heads
        self.head_dim = (
            embed_size // n_heads
        )  # when concatenated will result in embed_size
        self.scale = self.head_dim ** (-0.5)

        self.query = keras.layers.Dense(self.head_dim)
        self.key = keras.layers.Dense(self.head_dim)
        self.value = keras.layers.Dense(self.head_dim)
        self.softmax = keras.layers.Softmax()
        self.drop1 = keras.layers.Dropout(dropout_rate)

        self.proj = keras.layers.Dense(embed_size)
        self.drop2 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        q, k, v = self.query(inputs), self.key(inputs), self.value(inputs)
        k_t = tf.transpose(
            k, perm=[0, 2, 1]
        )  # Transpose --> [batch_size, head_dim, n_patches + 1]

        attn_filter = (q @ k_t) * self.scale
        attn_filter = self.drop1(self.softmax(attn_filter))

        attn_head = attn_filter @ v
        attn_head = tf.expand_dims(
            attn_head, axis=0
        )  # [1, batch_size, n_patches + 1, head_dim]

        heads = tf.concat(
            [attn_head for _ in range(self.n_heads)], axis=0
        )  # [n_heads, batch_size, n_patches + 1, head_dim]
        heads = tf.transpose(
            heads, perm=[1, 2, 3, 0]
        )  # [batch_size, n_patches + 1, head_dim, n_heads]

        bs, n_p, hd, nh = [keras.backend.shape(heads)[k] for k in range(4)]
        heads = tf.reshape(
            heads, [bs, n_p, hd * nh]
        )  # [batch_size, n_patches + 1, embed_dim]

        return self.drop2(self.proj(heads))


class MLP(keras.layers.Layer):
    """
    Class Implementing FeedForward Layer.

    * Linear
    * Activation (GELU)
    * Linear

    parameters
    ==========

    embed_size: int
        Embedding size.

    hidden_size: int
        output dim of first hidden layer

    activation_fn: str
        activation function applied after the first hidden layer

    Input
    =====

    input shape: shape
        [batch_size, n_patches + 1, embed_size]

    Output
    =====

    output shape: shape
        [batch_size, n_patches + 1, embed_size]
    """

    def __init__(self, embed_size, hidden_size, activation_fn="gelu", dropout_rate=0.2):
        super(MLP, self).__init__()

        self.Hidden = keras.layers.Dense(hidden_size)
        self.drop1 = keras.layers.Dropout(dropout_rate)
        self.activation = keras.activations.get(activation_fn)

        self.Linear = keras.layers.Dense(embed_size)
        self.drop2 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        x = inputs
        x = self.Hidden(x)
        x = self.drop1(self.activation(x))

        return self.drop2(self.Linear(x))


class TransformerEncoder(keras.layers.Layer):
    """
    Class for implementing Transformer Encoder Block.

    * Input
    * LayerNorm
    * Multi-head self attention
    * residual connection
    * LayerNorm
    * Multi-layer perceptron
    * residual connection

    parameters
    ==========

    embed_size: int
        Embedding size.

    n_heads: int

    mlpHidden_size: int
        output dim of first hidden layer of the MLP

    mlp_activation: str
        activation function applied after the first hidden layer of the MLP

    Input
    =====

    input shape: shape
        [batch_size, n_patches + 1, embed_size]

    Output
    =====

    output shape: shape
        [batch_size, n_patches + 1, embed_size]
    """

    def __init__(
        self,
        embed_size,
        n_heads,
        mlpHidden_size,
        mlp_activation,
        mlp_dropout,
        attn_dropout,
    ):
        super(TransformerEncoder, self).__init__()

        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-3)
        self.MHA = MHA(embed_size, n_heads, attn_dropout)
        self.MLP = MLP(embed_size, mlpHidden_size, mlp_activation, mlp_dropout)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-3)

    def call(self, inputs):
        x = inputs

        x = self.norm1(x)
        x = self.MHA(x)
        x = x + inputs

        y = self.norm2(x)
        y = self.MLP(y)
        y = y + x

        return y


class VisionTransformer(keras.Model):
    """
    Class for implementing Vision Transformer Architecture.

    * Input
    * Linear Projection
    * prepend cls token then add positional embedding
    * transformer encoder
    * LayerNorm
    * Linear

    parameters
    ==========

    embed_size: int
        Embedding size.

    patch_size: int
        Size of patch the images with be split into.

    n_head: int
        Number of heads of the multi-head self attention.

    mlpHidden_size: int
        output dim of first hidden layer of the MLP

    mlp_activation: str
        activation function applied after the first hidden layer of the MLP

    n_blocks: int
        Number of transformer encoder block

    n_classes: int
        Number of class for our image classification problem

    Input
    =====

    input shape: shape
        [batch_size, Depth, Hight, Width, Channels]

    Output
    =====

    output shape: shape
        [batch_size, n_classes]
    """

    def __init__(
        self,
        n_heads: int = 12,
        n_blocks: int = 12,
        img_size: int = 224,
        n_classes: int = 100,
        patch_size: int = 16,
        embed_size: int = 786,
        mlp_dropout: float = 0.0,
        pos_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        mlpHidden_size: int = 3072,
        mlp_activation: str = "gelu",
        **kwargs
    ):
        super(VisionTransformer, self).__init__(**kwargs)

        self.embed_size = embed_size
        self.proj = Projection(embed_size, patch_size, img_size)
        self.cls_token = tf.Variable(
            tf.zeros(shape=[1, 1, embed_size])
        )  # Will be broadcasted to batch size
        self.pos_embed = tf.Variable(
            tf.zeros(shape=[1, self.proj.n_patches + 1, embed_size])
        )  # Learnable Positional Embedding
        self.drop = keras.layers.Dropout(pos_dropout)

        self.Encoder_blocks = keras.Sequential(
            [
                TransformerEncoder(
                    embed_size,
                    n_heads,
                    mlpHidden_size,
                    mlp_activation,
                    mlp_dropout,
                    attn_dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        self.norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.pooling = keras.layers.GlobalAvgPool1D()
        self.Linear = keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        batch_size, depth, hight, width, channel = inputs.shape

        linear_embed = self.proj(
            inputs
        )  # shape --> [batch_size, n_patches, embed_size]

        broadcast_shape = tf.where(
            [True, False, False],
            keras.backend.shape(tf.expand_dims(linear_embed[:, 0], axis=1)),
            [0, 1, self.embed_size],
        )  # for broadcasting to a dynamic shape [None,  1, embed_size]
        cls_token = tf.broadcast_to(
            self.cls_token, shape=broadcast_shape
        )  # Found solution here --> (https://stackoverflow.com/questions/63211206/how-to-broadcast-along-batch-dimension-with-tensorflow-functional-api)

        assert cls_token.shape[0] == linear_embed.shape[0]
        linear_proj = tf.concat(
            [cls_token, linear_embed], axis=1
        )  # shape --> [batch_size, n_patches + 1, embed_size]
        linear_proj = linear_proj + self.pos_embed

        x = self.Encoder_blocks(self.drop(linear_proj))
        x = self.norm(x)

        cls_token_final = self.pooling(x)  # [
        #             :, 0
        #         ]  # Only the output of the final cls token should be considered
        return self.Linear(cls_token_final)


if __name__ == "__main__":
    rnd_vid = tf.random.uniform(shape=[1, 16, 224, 224, 3], dtype=tf.float32)

    model = VisionTransformer(
        n_heads=8,
        n_classes=2,
        img_size=224,
        mlp_dropout=0.1,
        pos_dropout=0.0,
        attn_dropout=0.0,
        embed_size=198,
        patch_size=8,
        n_blocks=4,
        mlpHidden_size=198 * 4,
    )

    output = model(rnd_vid)  # shape --> [batch_size, n_classes]
    print(output)
