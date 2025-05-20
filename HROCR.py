import cv2
import numpy as np
import pytesseract
import os
import datetime
# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Abhijeet Karwade\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

def order_points(pts):
    """Arrange points in a consistent order: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

def four_point_transform(image, pts):
    """Applies a perspective transform to get a top-down view."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width and height of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Define the destination points
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)

    return warped

def transform_image(image_path, points=None, use_automatic=True):
    """Transform an image into a straight image using automatic detection or user-selected points."""
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to load the image. Check the file path.")

    if points is not None and len(points) == 4:
        pts = np.array(points, dtype="float32")
    else:
        if use_automatic:
            image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=2)
            edged = cv2.Canny(dilated, 30, 100)
            cv2.imwrite("static/debug_edges.jpg", edged)
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            receipt_contour = None
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4:
                    receipt_contour = approx
                    break

            if receipt_contour is not None:
                pts = receipt_contour.reshape(4, 2)
            else:
                raise ValueError("Could not detect receipt contour automatically. Please select corners manually.")
        else:
            raise ValueError("No points provided and automatic detection is disabled.")

    transformed_image = four_point_transform(image, pts)
    transformed_path = "static/transformed_only_image.jpg"
    cv2.imwrite(transformed_path, transformed_image)
    return transformed_path

def correct_inclined_image(image_path, points=None, use_automatic=True):
    """Corrects the perspective of an inclined receipt image."""
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to load the image. Check the file path.")

    if points is not None and len(points) == 4:
        pts = np.array(points, dtype="float32")
    else:
        if use_automatic:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            receipt_contour = None
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4:
                    receipt_contour = approx
                    break

            if receipt_contour is not None:
                pts = receipt_contour.reshape(4, 2)
            else:
                raise ValueError("Could not detect receipt contour automatically. Please select corners manually.")
        else:
            raise ValueError("No points provided and automatic detection is disabled.")

    pts = order_points(pts)
    width, height = 500, 700
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    corrected_image = cv2.warpPerspective(image, M, (width, height))
    corrected_path = f"static/corrected_inclined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(corrected_path, corrected_image)
    return corrected_path

def extract_text_from_image(image_path):
    """Extract text from an image using perspective transformation and OCR, returning cropped and transformed images."""
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to load the image. Check the file path.")

    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    edged = cv2.Canny(dilated, 30, 100)
    cv2.imwrite("static/debug_edges.jpg", edged)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    receipt_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            receipt_contour = approx
            break

    if receipt_contour is not None:
        x, y, w, h = cv2.boundingRect(receipt_contour)
        cropped_image = image[y:y+h, x:x+w]
        transformed_image = four_point_transform(image, receipt_contour.reshape(4, 2))
        cropped_path = "static/cropped_image.jpg"
        transformed_path = "static/transformed_image.jpg"
        cv2.imwrite(cropped_path, cropped_image)
        cv2.imwrite(transformed_path, transformed_image)
        scanned_gray = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(scanned_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite("static/debug_binary.jpg", binary)
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789$. '
        extracted_text = pytesseract.image_to_string(binary, config=custom_config, lang='eng')
        return extracted_text, cropped_path, transformed_path
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789$. '
        extracted_text = pytesseract.image_to_string(binary, config=custom_config, lang='eng')
        return f"[Fallback Mode - No receipt contour detected]\n{extracted_text}", None, None