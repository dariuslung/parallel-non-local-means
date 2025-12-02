import numpy as np
from PIL import Image

def process_image():
    try:
        input_filename = "noisy_lena_512.png"
        output_filename = "noisy_lena_512_normalized.csv"

        # Load the image and convert to grayscale (L mode)
        img = Image.open(input_filename).convert('L')
        
        # Convert to numpy array
        pixel_data = np.asarray(img)
        
        # Normalize the pixel values to 0-1 range
        # floating point division by 255.0
        normalized_data = pixel_data / 255.0
        
        # Save to CSV
        # fmt='%.6f' ensures precision for the 0-1 float values
        np.savetxt(output_filename, normalized_data, delimiter=",", fmt='%.6f')
        
        print(f"Successfully saved {output_filename}")

    except Exception as error_message:
        print(f"An error occurred: {error_message}")

if __name__ == "__main__":
    process_image()