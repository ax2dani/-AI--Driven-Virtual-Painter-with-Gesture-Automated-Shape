import cv2
import numpy as np
import os
from handTracker import findDistances, findError, MediapipeHands
from sketch_corrector import SketchCorrector
from quickdraw_integration import QuickDrawRecognizer
import json
import pickle
import time
import sys
import torch
from quickdraw import QuickDrawData

# Base directory for resolving resource paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize global variables
Stroke = []
current_stroke = []
suggestions = []
recognition_result = None
last_draw_state = 'Standby'  # Track the last drawing state
detected_shape = None  # Track the detected shape for auto-correction
forced_shape = None  # Track if a specific shape is forced via gesture
forced_shape_confirmed = False  # True when gesture stable across frames
_recent_gestures = []  # history of recent detected gestures
_GESTURE_CONFIRM_WINDOW = 8   # frames to consider
_GESTURE_CONFIRM_MIN = 5      # minimum matches in window
_GESTURE_CLEAR_NONE = 20      # consecutive None frames to clear
_none_gesture_streak = 0
active_stroke_shape = None  # shape actually used during current stroke preview
_draw_on_counter = 0
_draw_off_counter = 0
_MIN_STROKE_DURATION_S = 0.18
_MIN_STROKE_PATH_PX = 80.0
_MIN_STROKE_BBOX_MIN_SIDE = 20
_MIN_POINT_STEP_PX = 2
_stroke_start_time = None
_stroke_path_len = 0.0

# Letters mode
letters_mode = False
letters_toggle_btn = None  # (x1, y1, x2, y2)
_letter_templates = {}
_letter_hu = {}

def _compute_hu(img_bin):
    # img_bin is single-channel 0/255 image
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments)
    # Log scale to stabilize
    with np.errstate(divide='ignore'):
        hu = -np.sign(hu) * np.log10(np.abs(hu))
    return hu.flatten()

def _generate_letter_templates(size=64):
    global _letter_templates
    global _letter_hu
    _letter_templates = {}
    _letter_hu = {}
    for ch in [chr(c) for c in range(ord('A'), ord('Z') + 1)]:
        img = np.zeros((size, size), dtype=np.uint8)
        # Choose font and compute scale to fill most of the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Initial scale; we will adjust with thickness
        scale = 1.5
        thickness = 3
        text_size, _ = cv2.getTextSize(ch, font, scale, thickness)
        # Adjust scale if too big/small
        target = int(size * 0.8)
        if text_size[0] > 0:
            scale = scale * target / max(text_size)
        text_size, _ = cv2.getTextSize(ch, font, scale, thickness)
        text_x = (size - text_size[0]) // 2
        text_y = (size + text_size[1]) // 2
        cv2.putText(img, ch, (text_x, text_y), font, scale, 255, thickness, cv2.LINE_AA)
        _letter_templates[ch] = img
        hu = _compute_hu(img)
        if hu is not None:
            _letter_hu[ch] = hu

def _render_stroke_bitmap(stroke, size=64, thickness=8):
    if not stroke or len(stroke) < 2:
        return np.zeros((size, size), dtype=np.uint8)
    # Normalize stroke to fit within size x size
    pts = np.array(stroke, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    if w == 0 or h == 0:
        return np.zeros((size, size), dtype=np.uint8)
    scale = min((size - 8) / float(w), (size - 8) / float(h))
    canvas = np.zeros((size, size), dtype=np.uint8)
    prev = None
    for p in pts:
        tx = int((p[0] - x) * scale) + 4
        ty = int((p[1] - y) * scale) + 4
        if prev is not None:
            cv2.line(canvas, prev, (tx, ty), 255, thickness, cv2.LINE_AA)
        prev = (tx, ty)
    return canvas

def _recognize_letter_from_stroke(stroke):
    # Ensure templates
    if not _letter_templates:
        _generate_letter_templates()
    stroke_img = _render_stroke_bitmap(stroke, size=64)
    # Binarize (already 0/255), but ensure uint8
    bin_img = (stroke_img > 0).astype(np.uint8) * 255
    stroke_hu = _compute_hu(bin_img)
    best_ch = None
    best_dist = float('inf')
    if stroke_hu is not None and _letter_hu:
        for ch, tmpl_hu in _letter_hu.items():
            dist = float(np.linalg.norm(stroke_hu - tmpl_hu))
            if dist < best_dist:
                best_dist = dist
                best_ch = ch
    # Fallback to correlation if HU unavailable
    if best_ch is None:
        best_score = -1.0
        for ch, tmpl in _letter_templates.items():
            res = cv2.matchTemplate(stroke_img, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(res[0][0])
            if score > best_score:
                best_score = score
                best_ch = ch
        return (best_ch if best_score >= 0.2 else None), best_score
    # Return HU-based with pseudo-confidence (inverse distance)
    confidence = 1.0 / (1.0 + best_dist)
    return best_ch, confidence

# Load settings
def load_settings(settings_path):
    defaults = {
        'window_height': 720,
        'window_width': 1280,
        'camera_port': 0,
        'fps': 30,
        'fullscreen': False,
        'fpsfilter': 0.9,
        'model_complexity': 1,
        'min_detection_confidence': 0.7,
        'min_tracking_confidence': 0.7,
        'confidence': 0.8,
        'keypoints': 21,
        'brush_size': [5, 10, 15, 20, 25, 30],
        'color_swatches': {
            'red': (0, 0, 255),
            'orange': (0, 165, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'cyan': (255, 255, 0),
            'blue': (255, 0, 0),
            'purple': (255, 0, 255),
            'pink': (203, 192, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0)
        },
        'command_hand': 'Left',
        'brush_hand': 'Right'
    }
    if os.path.exists(settings_path):
        try:
            with open(settings_path, 'r') as f:
                user_settings = json.load(f)
            if isinstance(user_settings, dict):
                defaults.update(user_settings)
        except Exception as e:
            print(f"Failed to load settings from {settings_path}: {e}. Using defaults.")
    else:
        print(f"settings.json not found at {settings_path}. Using defaults.")
    return defaults

settings = load_settings(os.path.join(BASE_DIR, 'settings.json'))

# Initialize canvas after loading settings
permanent_canvas = np.zeros([settings['window_height'], settings['window_width'], 3], dtype=np.uint8)

drawState = 'Standby'
# Initialize state variables
color = 'white'
brush_size = 20
show_suggestions = False
auto_correct = False

show_gesture_guide = False
last_selection_time = 0
selection_cooldown = 0.5  # Cooldown time in seconds between selections

# UI toggles
show_color_palette = True
color_toggle_btn = None  # (x1, y1, x2, y2)

# Initialize camera
camera = cv2.VideoCapture(settings['camera_port'], cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['window_height'])
camera.set(cv2.CAP_PROP_FRAME_WIDTH, settings['window_width'])
camera.set(cv2.CAP_PROP_FPS, settings['fps'])
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Initialize AI components
corrector = SketchCorrector(device='cuda' if torch.cuda.is_available() else 'cpu')
recognizer = QuickDrawRecognizer()

# Initialize window
cv2.namedWindow('OpenCV Paint', cv2.WINDOW_NORMAL)
if settings['fullscreen']:
    cv2.setWindowProperty('OpenCV Paint', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Load gesture data
gesturenames = []
knowngestures = []
fps = 0
fpsfilter = settings['fpsfilter']
starttime = time.time()
savetime = -1
run = True

gesture_path = os.path.join(BASE_DIR, 'gesture_data.pkl')
if os.path.exists(gesture_path):
    with open(gesture_path, 'rb') as f:
        gesturenames = pickle.load(f)
        knowngestures = pickle.load(f)
else:
    print(f'No gesture data found at {gesture_path}')
    sys.exit()

# Initialize hand tracker
findhands = MediapipeHands(
    model_complexity=settings['model_complexity'],
    min_detection_confidence=settings['min_detection_confidence'],
    min_tracking_confidence=settings['min_tracking_confidence']
)

# Initialize color settings
threshold = settings['confidence']
keypoints = settings['keypoints']
color_idx = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink', 'white', 'black']

def convert_toBNW(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame

def clearcanvas():
    global permanent_canvas, current_stroke, suggestions, recognition_result, detected_shape, forced_shape
    permanent_canvas = np.zeros([settings['window_height'], settings['window_width'], 3], dtype=np.uint8)
    current_stroke = []
    suggestions = []
    recognition_result = None
    detected_shape = None
    forced_shape = None

def draw_strokes(strokes, canvas, color=(255, 255, 255), thickness=2, padding=6):
    """Draw one or more strokes (list of points) onto a square canvas.

    strokes can be:
      - list[(x, y)]
      - list[list[(x, y)]]
    """
    if strokes is None:
        return canvas

    # Normalize to list[list[tuple]]
    if len(strokes) > 0 and isinstance(strokes[0], tuple):
        stroke_list = [strokes]
    else:
        stroke_list = strokes

    # Flatten points to compute bounds
    points = [pt for stroke in stroke_list for pt in stroke if pt is not None]
    if not points:
        return canvas

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)

    h, w = canvas.shape[:2]
    scale = min((w - 2 * padding) / width, (h - 2 * padding) / height)

    # Centering offsets
    scaled_w = width * scale
    scaled_h = height * scale
    offset_x = int((w - scaled_w) / 2)
    offset_y = int((h - scaled_h) / 2)

    def transform(pt):
        x = int((pt[0] - min_x) * scale) + offset_x
        y = int((pt[1] - min_y) * scale) + offset_y
        return (x, y)

    for stroke in stroke_list:
        for i in range(len(stroke) - 1):
            if stroke[i] is None or stroke[i + 1] is None:
                continue
            p1 = transform(stroke[i])
            p2 = transform(stroke[i + 1])
            cv2.line(canvas, p1, p2, color, thickness)
    return canvas

def saveimage():
    global savetime
    filename = ''
    for i in range(6):
        filename += f'{time.localtime()[i]}'
    pictures_dir = os.path.join(BASE_DIR, 'pictures')
    os.makedirs(pictures_dir, exist_ok=True)
    cv2.imwrite(os.path.join(pictures_dir, filename + f'.jpeg'), frame)
    savetime = time.time()

def process_stroke():
    global current_stroke, suggestions, recognition_result, permanent_canvas, detected_shape, forced_shape
    
    if len(current_stroke) < 3:
        return

    # Safety: validate stroke duration, movement, and size
    global _stroke_start_time, _stroke_path_len
    duration_ok = False
    if _stroke_start_time is not None:
        duration_ok = (time.time() - _stroke_start_time) >= _MIN_STROKE_DURATION_S
    bbox_ok = False
    pts_np = np.array(current_stroke, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts_np)
    bbox_ok = (min(w, h) >= _MIN_STROKE_BBOX_MIN_SIDE)
    path_ok = (_stroke_path_len >= _MIN_STROKE_PATH_PX)

    if not (duration_ok and bbox_ok and path_ok):
        # Too small/short/noisy: just discard without drawing letters/shapes
        current_stroke = []
        _stroke_start_time = None
        _stroke_path_len = 0.0
        return
    
    # Letters mode takes priority when enabled
    if letters_mode:
        ch, conf = _recognize_letter_from_stroke(current_stroke)
        if ch is not None:
            # Render recognized letter into the bounding box area of the stroke
            pts = np.array(current_stroke, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            # Choose a font scale to fit inside the box
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.0
            thickness = max(2, brush_size // 2)
            text_size, _ = cv2.getTextSize(ch, font, scale, thickness)
            # Adjust scale to fit width/height
            if text_size[0] > 0:
                scale_w = max(0.1, (w * 0.85) / text_size[0])
                scale_h = max(0.1, (h * 0.85) / text_size[1])
                scale = max(0.4, min(scale_w, scale_h))
                text_size, _ = cv2.getTextSize(ch, font, scale, thickness)
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2
            cv2.putText(permanent_canvas, ch, (text_x, text_y), font, scale,
                        settings['color_swatches'][color], thickness, cv2.LINE_AA)
            current_stroke = []
            _stroke_start_time = None
            _stroke_path_len = 0.0
            return
        # If no letter recognized, draw the original stroke and return (avoid shape fallback)
        for i in range(len(current_stroke) - 1):
            cv2.line(permanent_canvas, tuple(current_stroke[i]), tuple(current_stroke[i + 1]),
                    settings['color_swatches'][color], brush_size)
        current_stroke = []
        _stroke_start_time = None
        _stroke_path_len = 0.0
        return

    # Decide which shape (if any) to apply (when letters mode not active)
    shape_to_apply = None
    if auto_correct:
        # Prefer a confirmed gesture shape
        if forced_shape and forced_shape_confirmed:
            shape_to_apply = forced_shape
        else:
            # Fall back to auto-detection when no stable gesture
            detected = corrector.detect_shape(current_stroke)
            if detected and detected != 'unknown':
                shape_to_apply = detected

    if shape_to_apply:
        corrected_stroke = corrector.correct_stroke(current_stroke, shape_to_apply)
        for i in range(len(corrected_stroke) - 1):
            cv2.line(permanent_canvas, tuple(corrected_stroke[i]), tuple(corrected_stroke[i + 1]),
                    settings['color_swatches'][color], brush_size)
    else:
        # Draw the original stroke
        for i in range(len(current_stroke) - 1):
            cv2.line(permanent_canvas, tuple(current_stroke[i]), tuple(current_stroke[i + 1]),
                    settings['color_swatches'][color], brush_size)
    
    current_stroke = []
    _stroke_start_time = None
    _stroke_path_len = 0.0

def draw_suggestions(canvas):
    if not show_suggestions or not suggestions:
        return canvas
    
    suggestion_height = 100
    margin = 10
    for i, suggestion in enumerate(suggestions[:3]):
        y_start = 60 + i * (suggestion_height + margin)
        suggestion_canvas = np.zeros((suggestion_height, suggestion_height, 3), dtype=np.uint8)
        draw_strokes(suggestion, canvas=suggestion_canvas, color=(255, 255, 255))
        canvas[y_start:y_start + suggestion_height, 
               settings['window_width'] - suggestion_height - margin:settings['window_width'] - margin] = suggestion_canvas
    
    return canvas

def draw_recognition_info(canvas):
    if recognition_result:
        text = f"{recognition_result['category']} ({recognition_result['confidence']:.2f})"
        cv2.putText(canvas, text, (settings['window_width'] - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas

def get_selection_from_right_hand(right_hand):
    """Get color or brush size based on right hand position"""
    # Get index finger position
    right_index = right_hand[8]  # Right hand index finger tip
    
    # Check if right hand is in color selection area
    if show_color_palette and right_index[1] > 0 and right_index[1] < 60:
        color_index = min(int(right_index[0] / (settings['window_width'] / 10)), 9)
        return color_idx[color_index], brush_size
    
    # Check if right hand is in brush size selection area
    if right_index[0] > 0 and right_index[0] < 60 and right_index[1] > 60 and right_index[1] < settings['window_height'] - 60:
        diff = (settings['window_height'] - 120) // 6
        size_index = min(int((right_index[1] - 60) / diff), 5)
        return color, settings['brush_size'][size_index]
    
    return color, brush_size

def preprocess(frame, drawState, fps):
    frameleft = frame[60:settings['window_height'] - 60, :80]
    objectframeleft = np.zeros([settings['window_height'] - 120, 80, 3], dtype=np.uint8)
    frameleft = cv2.addWeighted(frameleft, .6, objectframeleft, .9, 0)
    frame[60:settings['window_height'] - 60, :80] = frameleft
    framebottom = frame[settings['window_height'] - 60:, :]
    objectframebottom = np.zeros([60, settings['window_width'], 3], dtype=np.uint8)
    framebottom = cv2.addWeighted(framebottom, .6, objectframebottom, .9, 0)
    frame[settings['window_height'] - 60:, :] = framebottom
    cv2.line(frame, (0, 60), (settings['window_width'], 60), (10, 10, 10), 2)
    
    # Draw color palette (toggleable)
    if show_color_palette:
        cntr = 0
        for x in range(0, settings['window_width'], settings['window_width'] // 10):
            pt1 = (x, 0)
            pt2 = (x + settings['window_width'], 0)
            pt4 = (x, 60)
            pt3 = (x + settings['window_width'], 60)
            cv2.fillPoly(frame, [np.array([pt1, pt2, pt3, pt4])], settings['color_swatches'][color_idx[cntr]])
            cntr += 1
    
    # Draw brush sizes
    cntr = 0
    for x in range((settings['window_height'] - 120) // 6, settings['window_height'] - 60,
                   (settings['window_height'] - 120) // 6):
        cv2.circle(frame, (40, x), settings['brush_size'][cntr], (255, 255, 255), -1)
        cntr += 1
    
    cv2.line(frame, (80, 60), (80, settings['window_height'] - 60), (10, 10, 10), 1)
    cv2.line(frame, (0, settings['window_height'] - 60),
             (int(3.4 * settings['window_width'] // 5), settings['window_height'] - 60), (10, 10, 10), 1)
    cv2.putText(frame, f'{drawState}', (20, settings['window_height'] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    cv2.line(frame, (settings['window_width'] // 8, settings['window_height'] - 60),
             (settings['window_width'] // 8, settings['window_height']), (10, 10, 10), 1)
    pt1 = (settings['window_width'] // 7, settings['window_height'] - 50)
    pt2 = (2 * settings['window_width'] // 7, settings['window_height'] - 50)
    pt3 = (2 * settings['window_width'] // 7, settings['window_height'] - 10)
    pt4 = (settings['window_width'] // 7, settings['window_height'] - 10)
    cv2.fillPoly(frame, [np.array([pt1, pt2, pt3, pt4])], settings['color_swatches'][color])
    if color == 'black':
        cv2.putText(frame, f'Eraser', (int(1.22 * settings['window_width'] // 7), settings['window_height'] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.line(frame, (int(1.8 * settings['window_width'] // 6), settings['window_height'] - 60),
             (int(1.8 * settings['window_width'] // 6), settings['window_height']), (10, 10, 10), 1)
    if brush_size == 30:
        cv2.circle(frame, (int(2 * settings['window_width'] // 6), settings['window_height'] - 30), brush_size - 4,
                   (255, 255, 255), -1)
    else:
        cv2.circle(frame, (int(2 * settings['window_width'] // 6), settings['window_height'] - 30), brush_size,
                   (255, 255, 255), -1)
    cv2.line(frame, (int(2.2 * settings['window_width'] // 6), settings['window_height'] - 60),
             (int(2.2 * settings['window_width'] // 6), settings['window_height']), (10, 10, 10), 1)
    cv2.putText(frame, f'C to clear', (int(2.3 * settings['window_width'] // 6), settings['window_height'] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.line(frame, (int(3.15 * settings['window_width'] // 6), settings['window_height'] - 60),
             (int(3.15 * settings['window_width'] // 6), settings['window_height']), (10, 10, 10), 1)
    cv2.putText(frame, f'S to save', (int(3.25 * settings['window_width'] // 6), settings['window_height'] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.line(frame, (int(3.4 * settings['window_width'] // 5), settings['window_height'] - 60),
             (int(3.4 * settings['window_width'] // 5), settings['window_height']), (10, 10, 10), 1)
    cv2.putText(frame, f'Q to quit', (int(3.48 * settings['window_width'] // 5), settings['window_height'] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.line(frame, (int(3.3 * settings['window_width'] // 4), settings['window_height'] - 60),
             (int(3.3 * settings['window_width'] // 4), settings['window_height']), (10, 10, 10), 1)
    cv2.putText(frame, f'Eraser', (int(2.74 * settings['window_width'] // 3), 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    
    # Add shape-correction controls info (move to left/top for full visibility)
    left_ui_x = 90
    bottom_ui_y = settings['window_height'] - 20
    cv2.putText(frame, f'A: Toggle Shape-correct', (left_ui_x, bottom_ui_y - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'H: Help   G: Guide', (left_ui_x, bottom_ui_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'{int(fps)} FPS', (left_ui_x, bottom_ui_y - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # Status block near top-left of canvas
    status_x = 90
    status_y = 90
    cv2.putText(frame, f'Shape-correct: {"ON" if auto_correct else "OFF"}', (status_x, status_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if auto_correct else (200, 200, 200), 2)
    cv2.putText(frame, f'Suggestions: {"ON" if show_suggestions else "OFF"}', (status_x, status_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Colors toggle button near status
    global color_toggle_btn
    btn_x1 = status_x
    btn_y1 = status_y + 70
    btn_x2 = btn_x1 + 180
    btn_y2 = btn_y1 + 30
    color_toggle_btn = (btn_x1, btn_y1, btn_x2, btn_y2)
    btn_color = (60, 60, 60) if show_color_palette else (30, 30, 30)
    border_color = (0, 255, 0) if show_color_palette else (0, 0, 255)
    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), btn_color, -1)
    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), border_color, 2)
    label = f'Colors: {"ON" if show_color_palette else "OFF"}'
    cv2.putText(frame, label, (btn_x1 + 10, btn_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Letters toggle button below
    global letters_toggle_btn
    lbtn_x1 = status_x
    lbtn_y1 = btn_y2 + 10
    lbtn_x2 = lbtn_x1 + 180
    lbtn_y2 = lbtn_y1 + 30
    letters_toggle_btn = (lbtn_x1, lbtn_y1, lbtn_x2, lbtn_y2)
    lbtn_color = (60, 60, 60) if letters_mode else (30, 30, 30)
    lborder_color = (0, 255, 0) if letters_mode else (0, 0, 255)
    cv2.rectangle(frame, (lbtn_x1, lbtn_y1), (lbtn_x2, lbtn_y2), lbtn_color, -1)
    cv2.rectangle(frame, (lbtn_x1, lbtn_y1), (lbtn_x2, lbtn_y2), lborder_color, 2)
    llabel = f'Letters: {"ON" if letters_mode else "OFF"}'
    cv2.putText(frame, llabel, (lbtn_x1 + 10, lbtn_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Display forced shape if gesture is active
    if forced_shape:
        cv2.putText(frame, f'Gesture shape: {forced_shape.capitalize()}', (status_x, status_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display gesture guide if enabled
    if show_gesture_guide:
        instructions = get_gesture_instructions()
        y_offset = 160
        for shape, gesture in instructions.items():
            cv2.putText(frame, f'{shape.capitalize()}: {gesture}', 
                        (int(3.3 * settings['window_width'] // 4), y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
    
    return frame

def mouseclick(event, xpos, ypos, *args, **kwargs):
    global color, brush_size, run, show_suggestions, auto_correct, detected_shape, show_gesture_guide, show_color_palette, letters_mode
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Toggle Colors button
        if color_toggle_btn is not None:
            x1, y1, x2, y2 = color_toggle_btn
            if xpos >= x1 and xpos <= x2 and ypos >= y1 and ypos <= y2:
                show_color_palette = not show_color_palette
                return

        # Toggle Letters button
        if letters_toggle_btn is not None:
            x1, y1, x2, y2 = letters_toggle_btn
            if xpos >= x1 and xpos <= x2 and ypos >= y1 and ypos <= y2:
                letters_mode = not letters_mode
                return

        # Color selection (only when palette visible)
        if show_color_palette and ypos > 0 and ypos < 60:
            if xpos > 0 and xpos < settings['window_width'] // 10:
                color = 'red'
            elif xpos > settings['window_width'] // 10 and xpos < 2 * settings['window_width'] // 10:
                color = 'orange'
            elif xpos > 2 * settings['window_width'] // 10 and xpos < 3 * settings['window_width'] // 10:
                color = 'yellow'
            elif xpos > 3 * settings['window_width'] // 10 and xpos < 4 * settings['window_width'] // 10:
                color = 'green'
            elif xpos > 4 * settings['window_width'] // 10 and xpos < 5 * settings['window_width'] // 10:
                color = 'cyan'
            elif xpos > 5 * settings['window_width'] // 10 and xpos < 6 * settings['window_width'] // 10:
                color = 'blue'
            elif xpos > 6 * settings['window_width'] // 10 and xpos < 7 * settings['window_width'] // 10:
                color = 'purple'
            elif xpos > 7 * settings['window_width'] // 10 and xpos < 8 * settings['window_width'] // 10:
                color = 'pink'
            elif xpos > 8 * settings['window_width'] // 10 and xpos < 9 * settings['window_width'] // 10:
                color = 'white'
            else:
                color = 'black'
        
        # Brush size selection
        if xpos > 0 and xpos < 60 and ypos > 60 and ypos < settings['window_height'] - 60:
            diff = (settings['window_height'] - 120) // 6
            if ypos > 60 and ypos < 60 + diff:
                brush_size = 5
            elif ypos > 60 + diff and ypos < 60 + 2 * diff:
                brush_size = 10
            elif ypos > 60 + 2 * diff and ypos < 60 + 3 * diff:
                brush_size = 15
            elif ypos > 60 + 3 * diff and ypos < 60 + 4 * diff:
                brush_size = 20
            elif ypos > 60 + 4 * diff and ypos < 60 + 5 * diff:
                brush_size = 25
            else:
                brush_size = 30
        
        # Control buttons
        if xpos > 0 and xpos < int(3.3 * settings['window_width'] // 4) and ypos > settings['window_height'] - 60:
            if xpos > 0 and xpos < int(3.15 * settings['window_width'] // 6):
                clearcanvas()
            elif xpos > int(3.15 * settings['window_width'] // 6) and xpos < int(3.4 * settings['window_width'] // 5):
                saveimage()
            else:
                run = False

cv2.setMouseCallback('OpenCV Paint', mouseclick)

def check_drawing_gesture(left_hand, right_hand):
    """Check if the hands are in drawing position"""
    # Get index finger positions
    left_index = left_hand[8]  # Left hand index finger tip
    right_index = right_hand[8]  # Right hand index finger tip
    
    # Check if right hand is in drawing area (main canvas)
    if right_index[0] > 80 and right_index[0] < settings['window_width'] and \
       right_index[1] > 60 and right_index[1] < settings['window_height'] - 60:
        return True
    return False

def is_left_fist_closed(left_hand):
    """Check if left hand is in fist position"""
    # Get finger tip positions
    thumb_tip = left_hand[4]
    index_tip = left_hand[8]
    middle_tip = left_hand[12]
    ring_tip = left_hand[16]
    pinky_tip = left_hand[20]
    
    # Get finger base positions
    index_base = left_hand[5]
    middle_base = left_hand[9]
    ring_base = left_hand[13]
    pinky_base = left_hand[17]
    
    # Check if all fingers are curled (tips are below their bases)
    fingers_closed = (
        index_tip[1] > index_base[1] and  # Index finger
        middle_tip[1] > middle_base[1] and  # Middle finger
        ring_tip[1] > ring_base[1] and  # Ring finger
        pinky_tip[1] > pinky_base[1]  # Pinky
    )
    
    return fingers_closed

def detect_shape_gesture(left_hand):
    """Detect shape-specific gestures with left hand"""
    # Get finger tip positions
    thumb_tip = left_hand[4]
    index_tip = left_hand[8]
    middle_tip = left_hand[12]
    ring_tip = left_hand[16]
    pinky_tip = left_hand[20]
    
    # Get finger base positions
    thumb_base = left_hand[2]
    index_base = left_hand[5]
    middle_base = left_hand[9]
    ring_base = left_hand[13]
    pinky_base = left_hand[17]
    
    # Check finger states (extended or closed)
    thumb_extended = thumb_tip[1] < thumb_base[1]
    index_extended = index_tip[1] < index_base[1]
    middle_extended = middle_tip[1] < middle_base[1]
    ring_extended = ring_tip[1] < ring_base[1]
    pinky_extended = pinky_tip[1] < pinky_base[1]
    
    # Shape gestures:
    # Circle: Thumb + Index (OK sign)
    if thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return 'circle'
    
    # Square: Thumb + Index + Middle (3 fingers)
    elif thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
        return 'square'
    
    # Triangle: Thumb + Index + Middle + Ring (4 fingers)
    elif thumb_extended and index_extended and middle_extended and ring_extended and not pinky_extended:
        return 'triangle'
    
    # Rectangle: All 5 fingers extended AND spread (reduce false positives)
    elif thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
        # Ensure fingers are horizontally spread apart sufficiently
        xs = [thumb_tip[0], index_tip[0], middle_tip[0], ring_tip[0], pinky_tip[0]]
        xs_sorted = sorted(xs)
        min_gap = min([xs_sorted[i+1] - xs_sorted[i] for i in range(len(xs_sorted)-1)])
        if min_gap > 8:  # pixel threshold; tweakable
            return 'rectangle'
    
    # Diamond: Index + Middle (2 fingers, no thumb)
    elif not thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
        return 'diamond'
    
    # Star: Index + Middle + Ring (3 fingers, no thumb)
    elif not thumb_extended and index_extended and middle_extended and ring_extended and not pinky_extended:
        return 'star'
    
    # Heart: Index + Pinky (peace sign)
    elif not thumb_extended and index_extended and not middle_extended and not ring_extended and pinky_extended:
        return 'heart'
    
    # Oval: Thumb + Pinky (hang loose)
    elif thumb_extended and not index_extended and not middle_extended and not ring_extended and pinky_extended:
        return 'oval'
    
    return None

def get_gesture_instructions():
    """Return instructions for shape gestures"""
    return {
        'circle': 'Thumb + Index (OK sign)',
        'square': 'Thumb + Index + Middle (3 fingers)',
        'triangle': 'Thumb + Index + Middle + Ring (4 fingers)',
        'rectangle': 'All 5 fingers extended',
        'diamond': 'Index + Middle (2 fingers, no thumb)',
        'star': 'Index + Middle + Ring (3 fingers, no thumb)',
        'heart': 'Index + Pinky (peace sign)',
        'oval': 'Thumb + Pinky (hang loose)'
    }

try:
    while run:
        dt = time.time() - starttime
        starttime = time.time()
        currentfps = 1 / dt
        fps = fps * fpsfilter + (1 - fpsfilter) * currentfps
        
        _, frame = camera.read()
        frame = cv2.flip(frame, 1)
        
        # Create a copy of the permanent canvas for this frame
        canvas = permanent_canvas.copy()
        
        # Process hand tracking
        handlandmarks, handstype = findhands.handsdata(frame)
        
        # Check for selection gesture (left fist open + right fingers up)
        left_hand_found = False
        right_hand_found = False
        left_hand_data = None
        right_hand_data = None
        
        for idx, handtype in enumerate(handstype):
            if handtype == settings['command_hand']:
                left_hand_found = True
                left_hand_data = handlandmarks[idx]
                frame = findhands.drawLandmarks(frame, [handlandmarks[idx]], False)
            elif handtype == settings['brush_hand']:
                right_hand_found = True
                right_hand_data = handlandmarks[idx]
                frame = findhands.drawLandmarks(frame, [handlandmarks[idx]], False)
        
        # If both hands are detected
        if left_hand_found and right_hand_found:
            # Check for shape gesture first (only when left hand is open)
            shape_gesture = None
            if not is_left_fist_closed(left_hand_data):
                shape_gesture = detect_shape_gesture(left_hand_data)

            # Maintain rolling gesture history for stability
            _recent_gestures.append(shape_gesture)
            if len(_recent_gestures) > _GESTURE_CONFIRM_WINDOW:
                _recent_gestures.pop(0)

            # Count most frequent non-None gesture in the window
            stable_shape = None
            if _recent_gestures:
                counts = {}
                for g in _recent_gestures:
                    if g is None:
                        continue
                    counts[g] = counts.get(g, 0) + 1
                if counts:
                    stable_shape, max_count = max(counts.items(), key=lambda kv: kv[1])
                    if max_count >= _GESTURE_CONFIRM_MIN:
                        forced_shape = stable_shape
                        forced_shape_confirmed = True
                    else:
                        forced_shape_confirmed = False

            # Clear forced shape only when not drawing and after consecutive None detections
            if drawState != 'Draw':
                if shape_gesture is None:
                    _none_gesture_streak += 1
                else:
                    _none_gesture_streak = 0
                if _none_gesture_streak >= _GESTURE_CLEAR_NONE:
                    forced_shape = None
                    forced_shape_confirmed = False
            
            # Check if left hand is in fist position
            if is_left_fist_closed(left_hand_data):
                # Drawing mode - left fist closed + right fingers up
                if check_drawing_gesture(left_hand_data, right_hand_data):
                    drawState = 'Draw'
                else:
                    drawState = 'Standby'
            else:
                # Selection mode - left fist open + right fingers up
                drawState = 'Standby'
                # When not drawing, we rely on stability rules above to clear shapes
                current_time = time.time()
                if current_time - last_selection_time > selection_cooldown:
                    new_color, new_size = get_selection_from_right_hand(right_hand_data)
                    if new_color != color or new_size != brush_size:
                        if new_color != color:
                            color = new_color
                            print(f"Selected color: {color}")
                        if new_size != brush_size:
                            brush_size = new_size
                            print(f"Selected brush size: {brush_size}")
                        last_selection_time = current_time
        else:
            drawState = 'Standby'
            forced_shape = None  # Reset forced shape when no hands detected
        
        # Process drawing
        for idx, handtype in enumerate(handstype):
            if handtype == settings['brush_hand']:
                current_point = (handlandmarks[idx][8][0], handlandmarks[idx][8][1])
                
                # Update current stroke with debounce and jitter filter
                if drawState == 'Draw':
                    # Start stroke timing/path tracking
                    if not current_stroke:
                        _stroke_start_time = time.time()
                        _stroke_path_len = 0.0
                        current_stroke.append(current_point)
                    else:
                        last_pt = current_stroke[-1]
                        dx = current_point[0] - last_pt[0]
                        dy = current_point[1] - last_pt[1]
                        step = (dx * dx + dy * dy) ** 0.5
                        if step >= _MIN_POINT_STEP_PX:
                            _stroke_path_len += step
                            current_stroke.append(current_point)
                    # Draw current point
                    cv2.circle(frame, current_point, brush_size, settings['color_swatches'][color], -1)
                
                # Draw current stroke with correction preview only if a stable gesture is active
                if len(current_stroke) > 1:
                    if auto_correct and forced_shape and forced_shape_confirmed and len(current_stroke) > 3:
                        temp_corrected = corrector.correct_stroke(current_stroke, forced_shape)
                        active_stroke_shape = forced_shape
                        for i in range(len(temp_corrected) - 1):
                            cv2.line(frame, temp_corrected[i], temp_corrected[i + 1],
                                    (0, 255, 0), max(1, brush_size // 2))
                    else:
                        # Draw original stroke in current color (no preview overlay)
                        for i in range(len(current_stroke) - 1):
                            cv2.line(frame, current_stroke[i], current_stroke[i + 1],
                                    settings['color_swatches'][color], brush_size)
                
                # Process stroke when switching to standby
                if drawState == 'Standby' and last_draw_state == 'Draw' and len(current_stroke) > 0:
                    process_stroke()
                    active_stroke_shape = None
        
        # Update last draw state
        last_draw_state = drawState
        
        # Update display
        frame = preprocess(frame, drawState, fps)
        
        # Draw the permanent canvas on top of the frame
        frame[60:settings['window_height'] - 60, 80:] = cv2.addWeighted(
            frame[60:settings['window_height'] - 60, 80:], 0.7,
            canvas[60:settings['window_height'] - 60, 80:], 0.3, 0
        )
        
        cv2.imshow('OpenCV Paint', frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            run = False
        elif key == ord('c'):
            clearcanvas()
        elif key == ord('s'):
            saveimage()
        elif key == ord('a'):  # Toggle shape-correction
            auto_correct = not auto_correct
            print(f"Shape-correction: {'ON' if auto_correct else 'OFF'}")
        elif key == ord('l'):  # Toggle letters mode
            letters_mode = not letters_mode
            print(f"Letters mode: {'ON' if letters_mode else 'OFF'}")
        elif key == ord('p'):  # Toggle color palette
            show_color_palette = not show_color_palette
            print(f"Colors palette: {'ON' if show_color_palette else 'OFF'}")
        elif key == ord('h'):  # Show help with available shapes
            print("Available shapes for shape-correction:")
            for shape in corrector.get_available_shapes():
                print(f"  - {shape.capitalize()}")
            print("Press 'A' to toggle shape-correction")
            print("\nHand Gestures for Shapes:")
            instructions = get_gesture_instructions()
            for shape, gesture in instructions.items():
                print(f"  {shape.capitalize()}: {gesture}")
            print("\nUsage: Make gesture with left hand, then draw with right hand")
        elif key == ord('g'):  # Toggle gesture guide
            show_gesture_guide = not show_gesture_guide
            print(f"Gesture guide: {'ON' if show_gesture_guide else 'OFF'}")

except Exception as e:
    print(f"Error: {e}")

finally:
    camera.release()
    cv2.destroyAllWindows()
