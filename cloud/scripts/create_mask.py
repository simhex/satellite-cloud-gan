from utils import get_filenames, load_image
import numpy as np
from PIL import Image

def create_border_mask(img_path):
  """
    Create a mask for the country borders based on these conditions:
     1. Green is the largest value.
     2. Green is greater than 100.
     3. Green - mean(Red, Blue) > 20.
  """ 
  img = load_image(img_path, crop = (21, 144, 1045, 656))
        
  # Create a mask based on the conditions
  red = img[:, :, 0]
  green = img[:, :, 1]
  blue = img[:, :, 2]
  mean_rb = (red + blue) / 2  
  mask = (green > red) & (green > blue) & (green > 80) & (green - mean_rb > 20)

  # Return the mask and the original image
  return mask, img

def create_and_save_mask(image_path, save_path):
    mask, img_rgb = create_border_mask(image_path)

    # Convert the mask to a grayscale image (0 = black, 255 = white)
    mask_image = np.zeros_like(img_rgb[:, :, 0]) 
    mask_image[mask] = 255  # Set the mask area to white

    # Convert the numpy array back to an image and save it
    mask_pil = Image.fromarray(mask_image)
    mask_pil.save(save_path)

    print(f"Mask saved to {save_path}")


image_path = get_filenames("sat_images")[4] # select one of the images
create_and_save_mask(image_path, 'mask.png')