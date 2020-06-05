import numpy as np
import matplotlib.pyplot as plt

def view_annotated(tensor, plot=True):
    temp = tensor.numpy()
    plt.imshow(rgb)
    plt.show()
  

def decode_image(tensor):
    inp = tensor.numpy().transpose((1, 2, 0))
    return inp

def view_image(tensor):
    inp = decode_image(tensor)
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()
