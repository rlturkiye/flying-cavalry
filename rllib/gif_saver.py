import tempfile
import moviepy.editor as mpy
import os
import tensorflow as tf

def encode_gif_bytes(im_thwc, fps=4):
  with tempfile.NamedTemporaryFile() as f: fname = f.name + '.gif'
  clip = mpy.ImageSequenceClip(list(im_thwc), fps=fps)
  clip.write_gif(fname, verbose=False, progress_bar=False)

  with open(fname, 'rb') as f: enc_gif = f.read()
  os.remove(fname)

  return enc_gif

def gif_summary(im_thwc, fps=4):
  """
  Given a 4D numpy tensor of images (TxHxWxC), encode a gif into a tf v1 Summary protobuf.
  Note that the tensor must be in the range [0, 255] as opposed to the usual small float values.
  """
  # create a tensorflow image summary protobuf:
  thwc = im_thwc.shape
  im_summ = tf.compat.v1.Summary.Image()
  im_summ.height = thwc[1]
  im_summ.width = thwc[2]
  im_summ.colorspace = 3 # fix to 3 for RGB
  im_summ.encoded_image_string = encode_gif_bytes(im_thwc, fps)

  # create a serialized summary obj:
  summ = tf.compat.v1.Summary()
  summ.value.add(image=im_summ)
  return summ.SerializeToString()