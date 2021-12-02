# Import TF 2.X and make sure we're running eager.
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
assert tf.executing_eagerly()

import tensorflow_hub as hub
import numpy as np

# Load the module and run inference.
module = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/2')
# `wav_as_float_or_int16` can be a numpy array or tf.Tensor of float type or
# int16. The sample rate must be 16kHz. Resample to this sample rate, if
# necessary.
wav_as_float_or_int16 = np.sin(np.linspace(-np.pi, np.pi, 128), dtype=np.float32)
print(wav_as_float_or_int16)
print(type(wav_as_float_or_int16))
print(wav_as_float_or_int16.shape)
emb_dict = module(samples=wav_as_float_or_int16, sample_rate=16000)
# For a description of the difference between the two endpoints, please see our
# paper (https://arxiv.org/abs/2002.12764), section "Neural Network Layer".
emb = emb_dict['embedding']
emb_layer19 = emb_dict['layer19']
# Embeddings are a [time, feature_dim] Tensors.
emb.shape.assert_is_compatible_with([None, 512])
emb_layer19.shape.assert_is_compatible_with([None, 12288])
    
print(emb)
print(emb_layer19)
