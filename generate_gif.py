import imageio
import os
import sys

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def main():
    images_dir = sys.argv[1]
    images = os.listdir(images_dir)
    images.sort()
    images = [os.path.join(images_dir, image) for image in images]
    images = images[::50]
    gif_name = 'progressive_training.gif'
    duration = 0.05
    create_gif(images, gif_name, duration)


if __name__ == '__main__':
    main()