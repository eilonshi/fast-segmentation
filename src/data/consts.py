BUILDING_LABEL = 3
OTHER_LABEL = 0

IRRELEVANT_LABELS_ADE = [27, 62, 91, 114, 129]
MAPPING_DICT_ADE = {
    1: [10, 14, 17, 30, 35, 47, 69, 95],
    2: [5, 18, 67, 73],
    3: [1, 2, 4, 6, 26, 49, 85],
    4: [87, 89, 115],
    5: [33],
    6: [3],
    7: [7, 55],
    8: [12, 53, 92],
    9: [9, 15, 19],
    10: [21, 84, 104, 117, 128, 134],
    11: [13],
}

IRRELEVANT_LABELS_CITYSCAPES = []
MAPPING_DICT_CITYSCAPES = {
    1: [6, 22],
    2: [21],
    3: [11],
    4: [],
    5: [12, 13, 14],
    6: [23],
    7: [7, 9],
    8: [8],
    9: [],
    10: [26, 27, 28, 29, 30, 31, 32, 33, -1],
    11: [24, 25],
}

IRRELEVANT_LABELS_BARAK = []
MAPPING_DICT_BARAK = {
    1: [[107, 142, 35]],
    2: [[0, 91, 46]],
    3: [[70, 70, 70]],
    4: [],
    5: [],
    6: [[180, 130, 70]],
    7: [],
    8: [],
    9: [],
    10: [],
    11: [],
}

TEVEL_SHAPE = (700, 560)

TRAIN_PERCENT = 0.7
VAL_PERCENT = 0.2
TEST_PERCENT = 0.1
