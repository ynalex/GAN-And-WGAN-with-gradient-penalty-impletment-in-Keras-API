import numpy as np
import cv2
import os
import random

def compress_image(photo_array, mutiplier):
    photo_array = np.array(photo_array)

    h = int(photo_array.shape[0])
    w = int(photo_array.shape[1])
    new_h = int(h/mutiplier)
    new_w = int(w/mutiplier)

    print("Compressing images with size".format(size) + " ({},".format(h) + "{}".format(w) + ").")
    
    new_image = np.zeros([new_h, new_w, 3])

    for i in range(int(h/mutiplier)):
        for j in range(int(w/mutiplier)):
                for z in range(3):
                    new_image[int(i),int(j),z] = photo_array[i*mutiplier,j*mutiplier,z]

    return new_image

img_folder_path = 'Image'
size = 10000
os.makedirs('Image2', exist_ok=True)
save_path = os.path.join(img_folder_path, 'Image2')

for file in sorted(os.listdir(img_folder_path)):
        image_path = os.path.join(img_folder_path, file)
        image = cv2.imread(image_path,cv2.COLOR_RGB2BGR)
        image = np.array(image)
        image = compress_image(image,4)
        save_name = os.path.join('Image2', file)
        print("Saving {}.".format(file))
        cv2.imwrite(save_name, image)
