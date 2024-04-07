import cv2
from PIL import Image
from colorthief import ColorThief
import matplotlib.pyplot as plt

def capture_photo(camera_index=0, photo_filename="captured_photo.jpg"):
    
    cap = cv2.VideoCapture(camera_index)

    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    
    print("Press the 'Space' key to capture a photo...")
    while True:
        ret, frame = cap.read()
        cv2.imshow("Capture Photo", frame)

        
        key = cv2.waitKey(1) & 0xFF
        if key == 32: 
            cv2.imwrite(photo_filename, frame)
            break

    
    cap.release()
    cv2.destroyAllWindows()

def crop_photo(image_path, crop_filename="cropped_photo.jpg"):
    
    img = cv2.imread(image_path)

    
    roi = cv2.selectROI(img)


    cropped_img = img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

    
    cv2.imwrite(crop_filename, cropped_img)

    
    cv2.imshow("Cropped Image", cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def analyze_skin_tone(image_path):
    
    color_thief = ColorThief(image_path)

    
    dominant_color = color_thief.get_color(quality=1) 


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))


    img = Image.open(image_path)
    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax1.axis("off")


    color_swatch = [[dominant_color]]
    ax2.imshow(color_swatch)
    ax2.set_title(f"Skin Tone is: {dominant_color}")
    ax2.axis("off")

    plt.show()

if __name__ == "__main__":
    
    capture_photo()

    
    crop_photo("captured_photo.jpg")

    
    analyze_skin_tone("cropped_photo.jpg")
