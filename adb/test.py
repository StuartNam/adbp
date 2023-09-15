from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

clean_data = 'db_dataset/5/set_B/412.jpg'
pertubed_data = 'outputs/ASPL/5/adversarial/noise-ckpt/50/50_noise_412.jpg'

clean_data = np.array(Image.open(clean_data))
pertubed_data = np.array(Image.open(pertubed_data))
noise = np.abs(pertubed_data - clean_data)

noise = Image.fromarray(noise)
noise.save('./noise.jpg')