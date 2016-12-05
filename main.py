import train
import scipy.misc
import numpy as np

style = scipy.misc.imread("test_style.jpg").astype(np.float)
content = scipy.misc.imread("test_content.jpg").astype(np.float)
img = np.clip(style, 0, 255).astype(np.uint8)
img = train.stylize(style, content, 1, 1, 1000, "imagenet-vgg-verydeep-19.mat")
scipy.misc.imsave("output.jpg", img)
