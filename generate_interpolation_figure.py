import os
import argparse
import numpy as np
from PIL import Image

import jittor as jt

from models.GAN import Generator
from generate_grid import adjust_dynamic_range


def draw_interpolation_figure(png, gen, out_depth, src_seeds, dst_seeds, n_steps):
    assert len(src_seeds) == len(dst_seeds)
    n_col = n_steps + 1
    n_row = len(dst_seeds)
    w = h = 2 ** (out_depth + 2)
    with jt.no_grad():
        latent_size = gen.g_mapping.latent_size
        src_latents_np = np.stack([np.random.RandomState(seed).randn(latent_size, ) for seed in src_seeds])
        dst_latents_np = np.stack([np.random.RandomState(seed).randn(latent_size, ) for seed in dst_seeds])
        src_latents = jt.array(src_latents_np.astype(np.float32))
        dst_latents = jt.array(dst_latents_np.astype(np.float32))
        
        src_dlatents = gen.g_mapping(src_latents)  # [seed, layer, component]
        dst_dlatents = gen.g_mapping(dst_latents)  # [seed, layer, component]
        src_images = gen.g_synthesis(src_dlatents, depth=out_depth, alpha=1)
        dst_images = gen.g_synthesis(dst_dlatents, depth=out_depth, alpha=1)

        src_dlatents_np = src_dlatents.numpy()
        dst_dlatents_np = dst_dlatents.numpy()
        # print(src_dlatents_np.shape, dst_dlatents_np.shape)
        # 5,12,512 5,12,512
        canvas = Image.new('RGB', (w * n_col, h * n_row), 'white')
        # for col, src_image in enumerate(list(src_images)):
        #     src_image = adjust_dynamic_range(src_image)
        #     src_image = (src_image*255).clamp(0, 255).int8().permute(1, 2, 0).numpy()
        #     canvas.paste(Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
        for row in range(n_row):
            src_image = adjust_dynamic_range(src_images[row])
            src_image = (src_image*255).clamp(0, 255).int8().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(src_image, 'RGB'), (0, row * h))
            
            dst_image = adjust_dynamic_range(dst_images[row])
            dst_image = (dst_image*255).clamp(0, 255).int8().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(dst_image, 'RGB'), ((n_col-1)*w, row * h))
            dlatents = np.stack([src_dlatents_np[row]] * (n_steps-1))
            for i in range(1, n_steps):
                dlatents[i-1] += (dst_latents_np[row]-src_dlatents_np[row]) * i / n_steps

            dlatents = jt.array(dlatents)
            images = gen.g_synthesis(dlatents, depth=out_depth, alpha=1)
            for col, image in enumerate(list(images)):
                image = adjust_dynamic_range(image)
                image = (image*255).clamp(0, 255).int8().permute(1, 2, 0).numpy()
                canvas.paste(Image.fromarray(image, 'RGB'), ((col + 1) * w, row * h))    
        
        canvas.save(png)


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """

    from config import cfg as opt

    opt.merge_from_file(args.config)
    opt.freeze()

    print("Creating generator object ...")
    # create the generator object
    gen = Generator(resolution=opt.dataset.resolution,
                    num_channels=opt.dataset.channels,
                    structure=opt.structure,
                    **opt.model.gen)

    print("Loading the generator weights from:", args.generator_file)
    # load the weights into it
    gen.load_state_dict(jt.load(args.generator_file))

    # path for saving the files:
    # generate the images:
    # src_seeds = [639, 701, 687, 615, 1999], dst_seeds = [888, 888, 888],
    draw_interpolation_figure(os.path.join('interpolation.png'), gen,
                             out_depth=5, src_seeds=[639, 1995, 687, 615, 1999], dst_seeds=[1010, 233, 960, 10, 88], n_steps=10)
    print('Done.')


def parse_arguments():
    """
    default command line argument parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./configs/sample_race_256.yaml')
    parser.add_argument("--generator_file", action="store", type=str,
                        help="pretrained weights file for generator", required=True)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_arguments())
