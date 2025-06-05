import cv2
import numpy as np
import gradio as gr

def underline_lines(image_path):
    # Load and preprocess
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Detect text blobs
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 8 and y > 150 and y < img.shape[0] - 100:
            boxes.append((x, y, w, h))

    # Group boxes by Y (lines)
    boxes.sort(key=lambda b: b[1])
    lines = []
    current_line = []

    for box in boxes:
        x, y, w, h = box
        if not current_line:
            current_line.append(box)
            continue
        _, prev_y, _, prev_h = current_line[-1]
        if abs(y - prev_y) < 20:
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]
    if current_line:
        lines.append(current_line)

    # Draw box instead of underline
    for line in lines:
        line = sorted(line, key=lambda b: b[0])
        segment = [line[0]]

        for i in range(1, len(line)):
            prev_box = segment[-1]
            curr_box = line[i]
            gap = curr_box[0] - (prev_box[0] + prev_box[2])

            if gap < 25:
                segment.append(curr_box)
            else:
                x_min = min(b[0] for b in segment)
                x_max = max(b[0] + b[2] for b in segment)
                y_max = max(b[1] + b[3] for b in segment)

                # Find box top by scanning upward
                top_y = y_max
                for y in range(y_max, max(y_max - 60, 0), -1):
                    row = thresh[y, x_min:x_max]
                    if np.count_nonzero(row) == 0:
                        top_y = y
                        break

                cv2.rectangle(img, (x_min, top_y), (x_max, y_max + 5), (0, 0, 255), 2)
                segment = [curr_box]

        # Draw last segment
        if segment:
            x_min = min(b[0] for b in segment)
            x_max = max(b[0] + b[2] for b in segment)
            y_max = max(b[1] + b[3] for b in segment)
            top_y = y_max - 10
            for y in range(y_max, max(y_max - 50, 0), -1):
                row = thresh[y, x_min:x_max]
                if np.count_nonzero(row) == 0:
                    top_y = y
                    break
            cv2.rectangle(img, (x_min, top_y), (x_max, y_max + 5), (0, 0, 255), 2)

    return img

# Gradio app
gr.Interface(
    fn=underline_lines,
    inputs=gr.Image(type="filepath", label="Upload a handwriting image"),
    outputs=gr.Image(type="numpy", label="Underlined Output"),
    title="Underline Handwritten Lines",
    description="Upload an image of handwritten text. The app will automatically underline each line."
).launch()
