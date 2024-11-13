import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def fft2_image(image):
    """Apply FFT to an image and shift zero frequency component to the center."""
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

def process_images(folder_path):
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("No images found in the folder.")
        return

    # Initialize a list to hold the frequency domain images
    freq_images = []

    for image_name in images:
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Skipping {image_name} as it could not be read.")
            continue

        freq_image = fft2_image(image)
        freq_images.append(freq_image)

    if not freq_images:
        print("No valid frequency domain images found.")
        return

    # Calculate the average of all frequency domain images
    avg_freq_image = np.mean(freq_images, axis=0)
    
    # Save the result
    output_filename = os.path.join(folder_path, f"{os.path.basename(folder_path)}_frequency.png")
    plt.imsave(output_filename, avg_freq_image, cmap='gray')
    print(f"Saved average frequency domain image to {output_filename}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    process_images(folder_path)

