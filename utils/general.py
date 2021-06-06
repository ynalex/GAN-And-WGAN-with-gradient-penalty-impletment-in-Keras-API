import numpy as np
def save_generated_image(generated_image_list, row=4,col=4):
    generated_image_list = generated_image_list.numpy()
    generated_image_list = (generated_image_list + 1) / 2

    b,h,w,_ = generated_image_list.shape()

    combined_image = np.zeros(shape=(h*col, w*row, 3))

    for y in range(col):
        for x in range(row):
            combined_image[y*h:(y+1)*h, x*w:(x+1)*w] = generated_image_list[x+y*row]

    return combined_image 
