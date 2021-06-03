NUM_WORKERS = 1
IGNORE_LABEL = 255

LABEL_TO_COLOR = [
    [0, 0, 0],  # other
    [0, 50, 50],  # land
    [0, 255, 0],  # trees
    [50, 200, 0],  # buildings
    [0, 150, 100],  # tents
    [80, 100, 30],  # fences
    [150, 255, 40],  # sky
    [0, 30, 200],  # road
    [90, 90, 200],  # country road
    [200, 40, 130],  # windows
    [250, 160, 100],  # cars
    [200, 40, 0],  # people
]

NUM_CLASSES = len(LABEL_TO_COLOR)
