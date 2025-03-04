# Image Processing Interface

This project is an interactive image processing interface built using Python, OpenCV, Tkinter, and other libraries. It allows users to load an image and apply various processing effects—such as brightness/contrast adjustment, convolution filtering, adding Gaussian noise, and applying the Kuwahara filter—directly from a graphical user interface.

---

## Example: 
![]("test1.png")

## Features

- **Image Loading**  
  Load an image from your computer and automatically resize it (maximum dimension of 400 pixels) for faster processing.

- **Brightness/Contrast Adjustment**  
  Use sliders to adjust the brightness (beta) and contrast (k) of the image in real time.

- **Convolution Filters**  
  Apply various convolution filters (e.g., Blur 3x3, Horizontal/Vertical Derivative, Sobel filters) to process the image.

- **Gaussian Noise**  
  Add Gaussian noise to the image by adjusting the noise standard deviation.

- **Kuwahara Filter**  
  Apply the Kuwahara filter to the image with a configurable window size. A warning message is displayed if the selected window size is above a certain threshold (indicating potential slow processing).

- **User-Friendly Interface**  
  The interface features a side panel with buttons to select the desired effect. Each effect displays its own controls and results within the same window.

---

## Installation

### Prerequisites

- Python 3.x

### Required Libraries

- OpenCV  
- NumPy  
- Pillow (PIL)  
- SciPy  
- Tkinter (usually comes pre-installed with Python)

### Installation via pip

Use the following command to install the necessary libraries:

```bash
pip install opencv-python numpy pillow scipy
```

---

## Usage

1. **Clone or Download the Repository**  
   Clone this repository or download the source code to your local machine.

2. **Run the Script**  
   Execute the main script to start the interface:

   ```bash
   python your_script_name.py
   ```

3. **Interact with the Interface**  
   - **Load Image:** Click the "Load Image" button in the side panel and select an image file. The image will be automatically resized for faster processing.
   - **Select Effect:** Use the side buttons to switch between effects:
     - *Brightness/Contrast*: Adjust brightness and contrast using the sliders.
     - *Convolution*: Choose a convolution filter from the dropdown and apply it.
     - *Gaussian Noise*: Adjust the noise level and add Gaussian noise.
     - *Kuwahara Filter*: Set the window size for the filter. If a high value is selected, a red warning message will appear indicating that processing may be slow.
   - **View Result:** The processed image will be displayed in the main panel of the interface.

---

## Project Structure

The main script contains both the image processing functions and the GUI implementation:

- **Image Processing Functions:**
  - `adjust_brightness_contrast`: Adjusts image brightness and contrast.
  - `convolution_filter`: Applies a convolution filter to the image.
  - `add_gaussian_noise`: Adds Gaussian noise to the image.
  - `kuwahara_filter`: Applies the Kuwahara filter to the image.
  - `resize_image`: Resizes the image to a maximum dimension for improved processing speed.

- **Graphical Interface:**
  - The interface is built using Tkinter and features a side panel with buttons to switch between different effects.
  - Each effect panel contains specific controls (e.g., sliders, dropdowns) and a display area for the resulting image.


---

## License: GNU

