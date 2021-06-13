NUM_WORKERS = 1
IGNORE_LABEL = 255
PIXELS_SCALE = 255
OTHER_LABEL = 0
BAD_IOU = 0.3

LABEL_TO_COLOR = {
    'other': (0, 0, 0),
    'land': (29, 101, 181),
    'trees': (0, 255, 0),
    'buildings': (180, 180, 180),
    'tents': (0, 20, 100),
    'fences': (80, 100, 30),
    'sky': (255, 248, 240),
    'road': (50, 150, 250),
    'country_road': (0, 200, 200),
    'windows': (200, 40, 130),
    'cars': (250, 0, 0),
    'people': (0, 0, 255),
}

NUM_CLASSES = len(LABEL_TO_COLOR)
