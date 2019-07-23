from cv2 import cv2
import numpy as np
from tensorflow import keras
from PIL import Image
import operator

def main():
    # size of camera output
    cam_size = 600

    # label's config
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN
    text_background = (0, 0, 0)
    text_offset_x = 10
    text_offset_y = cam_size - 25

    # getting all class names
    with open('class_names.txt', 'r') as f:
        class_names = f.read().splitlines()

    # loading the model
    model = keras.models.load_model('doodle_model.h5')

    # starting cv2 video capture
    cap = cv2.VideoCapture(0)
    while True:
        # getting middle of cropped camera output
        crop_size = int(cam_size / 2)

        _, frame = cap.read()
        
        # setting white backgound for lines to draw on to
        img = 255 * np.ones(shape=frame.shape, dtype=np.uint8)
        
        # line detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 75, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, maxLineGap=10)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 5)

        # cropping the image for setted output
        mid_h = int(img.shape[0] / 2)
        mid_w = int(img.shape[1] / 2)
        img = img[mid_h-crop_size:mid_h+crop_size, mid_w-crop_size:mid_w+crop_size]

        # converting and normalizing image to array
        # also expanding dims for further keras prediction
        im = Image.fromarray(img, 'RGB')
        im = im.resize((75, 75))
        img_array = np.array(im)/255.
        img_array = np.expand_dims(img_array, axis=0)

        # classifying the doodle
        pred = model.predict(img_array) * 100

        # getting class name and score of the best prediction
        index, _ = max(enumerate(pred[0]), key=operator.itemgetter(1))

        # generating output text
        text = '{} {}%'.format(class_names[index], int(pred[0][index]))

        # generating text box    
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))

        # drawing text box, text and showing the lines for better camera adjustment
        cv2.rectangle(img, box_coords[0], box_coords[1], text_background, cv2.FILLED)
        cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(255, 255, 255), thickness=1)
        cv2.imshow('AIlias', img)

        key = cv2.waitKey(1)
        if key == 27:
            break

    # ending cv2 cam capture        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
