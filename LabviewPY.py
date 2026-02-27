import cv2
import imutils
import numpy as np
import easyocr
import os
import time

# --- Step 1: Set paths ---
input_folder = "C:/Users/shanm/OneDrive/Documents/img"
output_file = "C:/Users/shanm/OneDrive/Documents/Detected_Plates.txt"

# --- Step 2: Initialize OCR Reader ---
reader = easyocr.Reader(['en'])

# --- Step 3: Create a set to track already processed files ---
processed_files = set()

print("🚦 Smart Traffic Number Plate Recognition System Started")
print(f"Monitoring folder: {input_folder}\n")

# --- Step 4: Run continuously ---
while True:
    # Get list of image files
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for img_name in images:
        if img_name not in processed_files:
            img_path = os.path.join(input_folder, img_name)
            print(f"\n📷 New image detected: {img_path}")

            # --- Step 5: Read and preprocess the image ---
            image = cv2.imread(img_path)
            if image is None:
                continue  # Skip if unreadable

            image = imutils.resize(image, width=600)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Extra preprocessing for better OCR detection
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

            # --- Step 6: Find plate contour (optional filtering) ---
            edged = cv2.Canny(gray, 30, 200)
            cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
            plate = None

            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.018 * peri, True)
                if len(approx) == 4:
                    plate = approx
                    break

            mask = np.zeros(gray.shape, np.uint8)
            if plate is not None:
                new_image = cv2.drawContours(mask, [plate], 0, 255, -1)
                new_image = cv2.bitwise_and(image, image, mask=mask)
                roi = cv2.boundingRect(plate)
                x, y, w, h = roi
                cropped = image[y:y+h, x:x+w]
            else:
                cropped = thresh  # fallback if contour not found

            # --- Step 7: OCR Detection ---
            result = reader.readtext(cropped, detail=0)
            if result:
                plate_text = " ".join(result)
                print(f"✅ Detected Number Plate: {plate_text}")
            else:
                plate_text = "No plate detected"
                print("⚠️ No plate detected")

            # --- Step 8: Save to file safely (UTF-8 encoding) ---
            with open(output_file, 'a', encoding='utf-8') as file:
                file.write(f"{img_name} -> {plate_text}\n")

            # --- Step 9: Display result (optional) ---
            cv2.imshow("Detected Image", image)
            cv2.waitKey(1000)

            # Mark as processed
            processed_files.add(img_name)

    # --- Step 10: Wait before checking again ---
    time.sleep(2)  # check every 2 seconds

cv2.destroyAllWindows()
