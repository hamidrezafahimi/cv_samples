import numpy as np
import matplotlib.pyplot as plt

vision = np.zeros([8,8,3],dtype=np.uint8)
data = np.random.random((8,8))

vision[:, :, 2] = np.uint8(np.random.uniform(low=0.3, high=0.7, size=(8,8))*255)
vision[1, 1, :] = (255,255,255)

img = plt.imshow(vision)
img.set_cmap('hot')
plt.axis('off')
plt.show()