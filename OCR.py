import cv2
import numpy as np
import pytesseract

# Path to the Tesseract executable (update this based on your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Abhijeet Karwade\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image loaded correctly
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Resize if needed (improves OCR for small images)
    height, width = image.shape[:2]
    if width < 1000:
        scale_factor = 2
        image = cv2.resize(image, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Adaptive Histogram Equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Apply Bilateral Filter to reduce noise while keeping edges sharp
    gray = cv2.bilateralFilter(gray, 9, 75, 80)

    # Adaptive Thresholding for better OCR
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 25, 10)

    # Use dilation to make text clearer
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Perform OCR with improved settings
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789$.:/"'
    extracted_text = pytesseract.image_to_string(binary, config=custom_config, lang='eng')

    return extracted_text