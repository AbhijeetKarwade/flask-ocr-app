from flask import Flask, request, jsonify, session
import cv2
import os
import json
import pickle
import re
from datetime import datetime
from OCR import extract_text_from_image as ocr_extract
from HROCR import extract_text_from_image as hrocr_extract, transform_image
from flask_session import Session

app = Flask(__name__)

# Configure session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with a secure key
Session(app)

# Configure static and data folders
STATIC_DIR = 'static'
DATA_DIR = 'data'
for directory in [STATIC_DIR, DATA_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# File to store the last bill number
BILL_COUNTER_FILE = os.path.join(DATA_DIR, 'bill_counter.pkl')

def get_next_bill_number():
    """Get the next sequential bill number, persisting it to a file."""
    try:
        if os.path.exists(BILL_COUNTER_FILE):
            with open(BILL_COUNTER_FILE, 'rb') as f:
                counter = pickle.load(f)
        else:
            counter = 0
        counter += 1
        with open(BILL_COUNTER_FILE, 'wb') as f:
            pickle.dump(counter, f)
        return counter
    except Exception as e:
        raise Exception(f"Failed to generate bill number: {str(e)}")

def parse_bill_text(text):
    text = text.lower()
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    bill_data = {
        'vendor_name': None,
        'items': [],
        'subtotal': None,
        'tax': None,
        'total': None,
        'payment_type': None
    }

    # Detect vendor name (first line or early lines)
    vendor_name_patterns = [
        re.compile(r'^\s*(?:receipt\s+from|invoice\s+from|bill\s+from)?\s*([a-zA-Z0-9\s&\.\-]+)\s*$'),
        re.compile(r'^([a-zA-Z0-9\s&\.\-]+)(?:\s+store|\s+shop|\s+mart|\s+inc|\s+llc)?$', re.IGNORECASE)
    ]

    for i in range(min(5, len(lines))):  # Check top 5 lines
        line = lines[i]
        for pattern in vendor_name_patterns:
            match = pattern.search(line)
            if match:
                name = match.group(1).strip().title()
                if len(name) > 2:
                    bill_data['vendor_name'] = name
                    break
        if bill_data['vendor_name']:
            break

    # Regex patterns
    price_pattern = re.compile(r'\b(?:\$?\s*)(\d+\.\d{2})\b')
    # Stricter item pattern to avoid matching total-like lines
    item_pattern = re.compile(
        r'^(?:(\d+)\s+)?'  # Optional quantity
        r'([a-zA-Z\s\-\(\)&/,]+?)'  # Item name (no numbers in name to avoid codes like "104251742")
        r'\s*(?:\$?\s*)(\d+\.\d{2})\s*$',  # Price
        re.IGNORECASE
    )
    subtotal_pattern = re.compile(r'(?:^|\s+)subtotal[:\s\.]*\$?(\d+\.\d{2})\b')
    tax_pattern = re.compile(r'(?:^|\s+)tax[:\s\.]*\$?(\d+\.\d{2})\b')
    # Updated total pattern to handle formats like "04/2517:42 TOTAL: 69.25"
    total_pattern = re.compile(
        r'(?:^|\s+)(?:total|tot*|lotal|jotal|amount|balance|grand\s+total)[:\s]*(?:\$?\s*)(\d+\.\d{2})\b',
        re.IGNORECASE
    )
    payment_pattern = re.compile(r'\b(cash|credit|debit|card|visa|mastercard|amex)\b')
    # Stricter exclude pattern to block lines with "total" or numbers
    exclude_pattern = re.compile(r'\b(subtotal|total|tax|discount|change|server|check|table|we[kl]|amount|balance|grand|\d{5,})\b')

    item_section = False

    # First pass: Extract total, subtotal, tax, and payment type
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if subtotal_pattern.search(line):
            bill_data['subtotal'] = subtotal_pattern.search(line).group(1)
        if tax_pattern.search(line):
            bill_data['tax'] = tax_pattern.search(line).group(1)
        if total_pattern.search(line):
            bill_data['total'] = total_pattern.search(line).group(1)
        if payment_pattern.search(line):
            bill_data['payment_type'] = payment_pattern.search(line).group(1).capitalize()

    # Second pass: Extract items
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect start of item section
        if item_pattern.match(line) and not exclude_pattern.search(line):
            item_section = True

        # Process item lines only if they don't contain excluded keywords
        item_match = item_pattern.match(line)
        if item_match and not exclude_pattern.search(line) and item_section:
            quantity = int(item_match.group(1)) if item_match.group(1) else 1
            item_name = item_match.group(2).strip()
            # Clean and format item name
            item_name = re.sub(r'\s+', ' ', item_name)  # Normalize spaces
            item_name = re.sub(r'(?<=[a-z])([A-Z])', r' \1', item_name).title()
            price = item_match.group(3)
            bill_data['items'].append({
                'quantity': quantity,
                'name': item_name,
                'price': price
            })

    return bill_data

def save_to_json(data, mode, image_path, cropped_path=None, transformed_path=None):
    """Save extracted bill information to a JSON file with only the specified fields."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_filename = f"ocr_result_{timestamp}.json"
    json_path = os.path.join(DATA_DIR, json_filename)

    # Parse bill details from extracted text
    bill_data = parse_bill_text(data.get('text', ''))

    # Create JSON with only the requested fields
    json_data = {
        'bill_no': get_next_bill_number(),
        'vendor_name': bill_data['vendor_name'],
        'items': bill_data['items'],
        'subtotal': bill_data['subtotal'],
        'tax': bill_data['tax'],
        'total': bill_data['total'],
        'payment_type': bill_data['payment_type']
    }

    # Save to JSON file
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        return json_path
    except Exception as e:
        raise Exception(f"Failed to save JSON: {str(e)}")

@app.route('/')
def index():
    return app.send_static_file('index1.html')

@app.route('/crop', methods=['POST'])
def crop_image():
    """Crop the uploaded image based on provided coordinates."""
    temp_path = os.path.join(STATIC_DIR, 'temp_image.jpg')
    try:
        if 'image' not in request.files or 'crop_coords' not in request.form:
            return jsonify({'error': 'Image and crop coordinates are required'}), 400

        file = request.files['image']
        crop_coords = json.loads(request.form['crop_coords'])

        if not file or file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        # Validate crop coordinates
        required_keys = ['x1', 'y1', 'x2', 'y2']
        if not all(key in crop_coords for key in required_keys):
            return jsonify({'error': 'Invalid crop coordinates'}), 400

        x1, y1, x2, y2 = crop_coords['x1'], crop_coords['y1'], crop_coords['x2'], crop_coords['y2']
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
            return jsonify({'error': 'Invalid crop dimensions'}), 400

        file.save(temp_path)
        img = cv2.imread(temp_path)
        if img is None:
            os.remove(temp_path)
            return jsonify({'error': 'Invalid or corrupted image file'}), 400

        # Ensure crop coordinates are within image bounds
        h, w = img.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            os.remove(temp_path)
            return jsonify({'error': 'Crop dimensions too small'}), 400

        # Crop the image
        cropped_img = img[y1:y2, x1:x2]
        cropped_path = os.path.join(STATIC_DIR, f'cropped_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
        cv2.imwrite(cropped_path, cropped_img)

        return jsonify({'cropped_image': cropped_path})

    except Exception as e:
        return jsonify({'error': f'Cropping failed: {str(e)}'}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/transform', methods=['POST'])
def transform_image_only():
    """Transform an uploaded image using automatic or manual corner detection."""
    temp_path = os.path.join(STATIC_DIR, 'temp_image.jpg')
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        points = request.form.get('points')
        use_automatic = request.form.get('use_automatic', 'true').lower() == 'true'

        if not file or file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        points_list = None
        if points:
            try:
                points_list = eval(points)
                if not isinstance(points_list, list) or len(points_list) != 4:
                    return jsonify({'error': 'Points must be a list of exactly four [x, y] coordinates'}), 400
                for point in points_list:
                    if not isinstance(point, list) or len(point) != 2:
                        return jsonify({'error': 'Each point must be a list of [x, y] coordinates'}), 400
            except Exception as e:
                return jsonify({'error': f'Invalid points format: {str(e)}'}), 400

        file.save(temp_path)
        img = cv2.imread(temp_path)
        if img is None:
            os.remove(temp_path)
            return jsonify({'error': 'Invalid or corrupted image file'}), 400

        transformed_path = transform_image(temp_path, points=points_list, use_automatic=use_automatic)
        json_path = save_to_json({'text': ''}, 'transform_only', temp_path, transformed_path=transformed_path)

        response = {
            'transformed_image': transformed_path,
            'json_path': json_path
        }
        session['last_data'] = {'text': '', 'transformed_image': transformed_path}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Transformation failed: {str(e)}'}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/process', methods=['POST'])
def process_image():
    """Process an uploaded image to extract text, with standard or high-res mode."""
    temp_path = os.path.join(STATIC_DIR, 'temp_image.jpg')
    try:
        if 'image' not in request.files or 'mode' not in request.form:
            return jsonify({'error': 'Image and mode are required'}), 400

        file = request.files['image']
        mode = request.form['mode']
        if not file or file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        if mode not in ['standard', 'high-res']:
            return jsonify({'error': 'Invalid mode. Use "standard" or "high-res"'}), 400

        file.save(temp_path)
        img = cv2.imread(temp_path)
        if img is None:
            os.remove(temp_path)
            return jsonify({'error': 'Invalid or corrupted image file'}), 400

        if mode == 'standard':
            text = ocr_extract(temp_path)
            response = {'text': text}
        else:
            try:
                result = hrocr_extract(temp_path)
                if len(result) != 3:
                    raise ValueError(f"Expected 3 values from hrocr_extract, got {len(result)}")
                text, cropped_path, transformed_path = result
                response = {
                    'text': text,
                    'cropped_image': cropped_path,
                    'transformed_image': transformed_path
                }
            except ValueError as ve:
                return jsonify({'error': f'Processing failed: {str(ve)}'}), 500

        session['last_data'] = response
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/save_json', methods=['POST'])
def save_json():
    """Save the provided text data to a JSON file in the data folder."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Invalid JSON data: "text" field is required'}), 400

        # Use the existing save_to_json function
        json_path = save_to_json(
            data,
            'manual_save',  # Mode for manual JSON save
            'static/temp_image.jpg'  # Placeholder, not used since image isn't reprocessed
        )
        return jsonify({'json_path': json_path})

    except Exception as e:
        return jsonify({'error': f'Failed to save JSON: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)