# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf
import numpy as np
import scipy.misc 
import matplotlib.pyplot as plt
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(name=tag, data=value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        with self.writer.as_default():
            # Ensure images are in the format TensorFlow expects, [batch_size, height, width, channels].
            # Convert to [0, 255] uint8 format.
            for i, img in enumerate(images):
                # Convert image to [0, 1] range
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                plt.savefig(f"{tag}_{i}.png", bbox_inches='tight', pad_inches=0.0)
                plt.close()
                
                img = tf.io.read_file(f"{tag}_{i}.png")
                img = tf.image.decode_png(img, channels=4)
                img = tf.expand_dims(img, 0)  # Add batch dimension.
                
                tf.summary.image(name=f'{tag}/{i}', data=img, step=step)
            self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        with self.writer.as_default():
            tf.summary.histogram(name=tag, data=values, step=step, buckets=bins)
            self.writer.flush()

