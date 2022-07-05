# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import infer
from DB import Database
import matplotlib.pyplot as plt
from PIL import Image

from color import Color
from daisy import Daisy
from edge  import Edge
from gabor import Gabor
from HOG   import HOG
from vggnet import VGGNetFeat
from resnet import ResNetFeat
from fusion import FeatureFusion

depth = 5
d_type = 'jensenshannon'
query_idx = 1300

if __name__ == '__main__':
  db = Database()

  # retrieve by color
  method = Color()
  samples = method.make_samples(db)
  query = samples[query_idx]
  print('query image:', query['img'])
  _, result, perf = infer(query, samples=samples, depth=depth, d_type=d_type)
  print("Color:",result)
  print("Performance:",perf)

  # Plot the result
  fig, axes = plt.subplots(2,depth)
  fig.suptitle('Color-feature based search')
  axes[0][depth//2].imshow(Image.open(query['img']))
  axes[0][depth//2].set_title('Query image')
  axes[0][depth//2].set_axis_off()

  for i in range(depth):
    if i==depth//2:
        continue
    axes[0][i].set_frame_on(False)
    axes[0][i].set_axis_off()

  for i, r in enumerate(result):
    axes[1][i].imshow(Image.open(r['img']))
    axes[1][i].set_title('Image {}'.format(i+1))
    axes[1][i].set_axis_off()
    axes[1][i].set_xlabel('Distance: {}'.format(r['dis']))
  
  plt.show()

#   # retrieve by daisy
#   method = Daisy()
#   samples = method.make_samples(db)
#   query = samples[query_idx]
#   _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
#   print(result)

  # retrieve by edge
  method = Edge()
  samples = method.make_samples(db)
  query = samples[query_idx]
  _, result, perf = infer(query, samples=samples, depth=depth, d_type=d_type)
  print("Edge:",result)
  print("Performance:",perf)

  # Plot the result
  fig, axes = plt.subplots(2,depth)
  fig.suptitle('Edge-feature based search')
  axes[0][depth//2].imshow(Image.open(query['img']))
  axes[0][depth//2].set_title('Query image')
  axes[0][depth//2].set_axis_off()

  for i in range(depth):
    if i==depth//2:
        continue
    axes[0][i].set_frame_on(False)
    axes[0][i].set_axis_off()

  for i, r in enumerate(result):
    axes[1][i].imshow(Image.open(r['img']))
    axes[1][i].set_title('Image {}'.format(i+1))
    axes[1][i].set_axis_off()
    axes[1][i].set_xlabel('Distance: {}'.format(r['dis']))
  
  plt.show()

  # retrieve by gabor
  method = Gabor()
  samples = method.make_samples(db)
  query = samples[query_idx]
  _, result, perf = infer(query, samples=samples, depth=depth, d_type=d_type)
  print("Texture:",result)
  print("Performance:",perf)

  # Plot the result
  fig, axes = plt.subplots(2,depth)
  fig.suptitle('Texture-feature based search')
  axes[0][depth//2].imshow(Image.open(query['img']))
  axes[0][depth//2].set_title('Query image')
  axes[0][depth//2].set_axis_off()

  for i in range(depth):
    if i==depth//2:
        continue
    axes[0][i].set_frame_on(False)
    axes[0][i].set_axis_off()

  for i, r in enumerate(result):
    axes[1][i].imshow(Image.open(r['img']))
    axes[1][i].set_title('Image {}'.format(i+1))
    axes[1][i].set_axis_off()
    axes[1][i].set_xlabel('Distance: {}'.format(r['dis']))
  
  plt.show()

  # retrieve by fusion of Texture, and Color features
  method = FeatureFusion(features=['gabor', 'color'])
  samples = method.make_samples(db)
  query = samples[query_idx]
  _, result, perf = infer(query, samples=samples, depth=depth, d_type=d_type)
  print("Fusion:",result)
  print("Performance:",perf)

  # Plot the result
  fig, axes = plt.subplots(2,depth)
  fig.suptitle('Texture and Color Fusion based search')
  axes[0][depth//2].imshow(Image.open(query['img']))
  axes[0][depth//2].set_title('Query image')
  axes[0][depth//2].set_axis_off()

  for i in range(depth):
    if i==depth//2:
        continue
    axes[0][i].set_frame_on(False)
    axes[0][i].set_axis_off()
  
  for i, r in enumerate(result):
    axes[1][i].imshow(Image.open(r['img']))
    axes[1][i].set_title('Image {}'.format(i+1))
    axes[1][i].set_axis_off()
    axes[1][i].set_xlabel('Distance: {}'.format(r['dis']))
  
  plt.show()

#   # retrieve by HOG
#   method = HOG()
#   samples = method.make_samples(db)
#   query = samples[query_idx]
#   _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
#   print(result)

#   # retrieve by VGG
#   method = VGGNetFeat()
#   samples = method.make_samples(db)
#   query = samples[query_idx]
#   _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
#   print(result)

#   # retrieve by resnet
#   method = ResNetFeat()
#   samples = method.make_samples(db)
#   query = samples[query_idx]
#   _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
#   print(result)