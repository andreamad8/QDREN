from tqdm import tqdm
import os
import imageio



images = []
for filename in tqdm(sorted(os.listdir('data/plot/task1_REN/'), key=lambda name: int(name[2:-4]))):
    images.append(imageio.imread('data/plot/task1_REN/'+filename))

imageio.mimsave('data/test.gif', images,duration = 0.4)
