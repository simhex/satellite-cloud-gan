import os
from typing import Tuple
from tqdm import tqdm
import numpy as np
import argparse
from utils import get_filenames, patchify, save_patches
from PIL import Image

def build_argparser() -> argparse.ArgumentParser:
    """
    Build the argument parser for command line arguments.
    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = argparse.ArgumentParser(description="Process real clouds dataset.")
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing the images to process."
    )
    parser.add_argument(
        "--input_mask", type=str, required=True,
        help="Directory containing the mask."
    )
    parser.add_argument(
        "--patch_size", type=int, nargs=2, required=True,
        help="Size of the patches (height, width)."
    )
    parser.add_argument(
        "--crop", type=int, nargs=4, required=True,
        help="Coordinates for cropping (left, upper, right, lower)."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save the patches."
    )
    parser.add_argument(
        "--max_files", type=int, default=-1,
        help="Maximum number of files to process. Default is -1 (all files)."
    )
    return parser

# The Perlin noise implementation is from: https://pvigier.github.io/2018/06/13/perlin-noise-numpy.html

def generate_perlin_noise_2d(shape, res):
    def f(t): return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * np.pi * np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)

    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0]-1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0]-1, grid[:, :, 1]-1)) * g11, 2)

    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (int(frequency * res[0]), int(frequency * res[1])))
        frequency *= 2
        amplitude *= persistence
    return noise

def generate_synthetic_clouds(
        shape: Tuple[int, int],
        res: Tuple[int, int],
        octaves: int) -> np.ndarray:
    """Generate a 2D numpy array of noise.
    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        octaves: The number of octaves in the noise.
    Returns:
        A numpy array of shape shape with the generated noise.
    """
    noise = generate_fractal_noise_2d(shape, res, octaves)
    return noise


def apply_synthetic_clouds_to_mask(
        noise: np.ndarray,
        mask: np.ndarray) -> np.ndarray:
    """Apply the generated noise to the mask. First, normalize the
    noise to be between 0 and 255, then add the noise to the mask.
    The mask is assumed to be in the range [0, 255]. Clip the output
    to be in the range [0, 255] and convert it to uint8.
    Args:
        noise: The generated noise.
        mask: The mask to apply the noise to.
    Returns:
        A numpy array of the same shape as the mask with the noise
        applied.
    """
    noise_normalized = 255 * (noise - noise.min()) / (noise.max() - noise.min())
    result = mask.astype(np.float32) + noise_normalized
    result_clipped = np.clip(result, 0, 255)
    return result_clipped.astype(np.uint8)


def process(
        N: int, 
        input_mask: str,
        patch_size: Tuple[int, int],
        crop: Tuple[int, int, int, int],
        output_dir: str) -> None:
    """
    Process the synthetic clouds by creating the synthetic images, patchifying them,
    and saving the patches to the output directory.
    Args:
        N (int): Number of images to process.
        input_mask (str): Path to the input mask.
        patch_size (Tuple[int, int]): Size of the patches (height, width).
        crop (Tuple[int, int, int, int]): Coordinates for cropping (left, upper, right, lower).
        output_dir (str): Directory to save the patches.
    Returns:
        None
    """
    # Load mask
    mask = np.array(Image.open(input_mask).convert("L"))

    print(f"----- Size of N:{N} ----- \n")

    for i in tqdm(range(N), desc="Processing"):

        # Generate noise
        synthetic_cloud = generate_synthetic_clouds(shape=mask.shape, res=(8, 8), octaves=4)

        # Apply clouds to mask
        synthetic_cloud = apply_synthetic_clouds_to_mask(synthetic_cloud, mask)

        # Split into patches
        patches = patchify(synthetic_cloud, patch_size)
        
        # Save patches
        print(f"Start index: {i * len(patches)}")
        save_patches(patches, output_dir, starting_index=i * len(patches))

def main():
    """
    Main function to execute the script.
    """
    parser = build_argparser()
    args = parser.parse_args()
    filenames = get_filenames(args.input_dir)
    if args.max_files > 0:
        filenames = filenames[:args.max_files]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    process(len(filenames), args.input_mask, args.patch_size, args.crop, args.output_dir)


if __name__ == "__main__":
    main()
