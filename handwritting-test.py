import cv2
import numpy as np
import gradio as gr

def underline_boxes(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 8 and y > 100 and y < img.shape[0] - 100:
            boxes.append((x, y, w, h))

    # Group boxes into lines
    boxes.sort(key=lambda b: b[1])
    lines = []
    current_line = []

    for box in boxes:
        x, y, w, h = box
        if not current_line:
            current_line.append(box)
            continue
        _, prev_y, _, _ = current_line[-1]
        if abs(y - prev_y) < 25:
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]
    if current_line:
        lines.append(current_line)

    # Group into words
    for line in lines:
        line = sorted(line, key=lambda b: b[0])
        word = [line[0]]

        for i in range(1, len(line)):
            prev = word[-1]
            curr = line[i]

            prev_right = prev[0] + prev[2]
            gap = curr[0] - prev_right
            same_baseline = abs(curr[1] - prev[1]) < 12

            if gap <= 1:
                word.append(curr)
            elif gap < 25 or (curr[2] < 15 and gap < 50 and same_baseline):
                word.append(curr)
            else:
                x_min = min(b[0] for b in word)
                x_max = max(b[0] + b[2] for b in word)
                y_min = min(b[1] for b in word)
                y_max = max(b[1] + b[3] for b in word)
                cv2.rectangle(img, (x_min - 2, y_min - 2), (x_max + 2, y_max + 2), (0, 0, 255), 2)
                word = [curr]

        # Final box
        if word:
            x_min = min(b[0] for b in word)
            x_max = max(b[0] + b[2] for b in word)
            y_min = min(b[1] for b in word)
            y_max = max(b[1] + b[3] for b in word)
            cv2.rectangle(img, (x_min - 2, y_min - 2), (x_max + 2, y_max + 2), (0, 0, 255), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Gradio app
gr.Interface(
    fn=underline_boxes,
    inputs=gr.Image(type="filepath", label="Upload a handwriting image"),
    outputs=gr.Image(type="numpy", label="Boxed Output"),
    title="Box Handwritten Words",
    description="Draws one red box per full word — even for cursive or lightly spaced handwriting."
).launch()
