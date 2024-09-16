import numpy as np



# JUST BLACK PIXEL TO COMPARING
BLACK = np.array([0, 0, 0], dtype=np.uint8)


# COLOR GRADIENTS FOR BINARY TRANSFORMATION OF AN AREA WITH SQUARE WIDTH IN METERS
METER_LOW = np.array([0, 0, 0], dtype=np.uint8)
METER_HIGH = np.array([18, 18, 18], dtype=np.uint8)
# METER_HIGH = np.array([37, 38, 38], dtype=np.uint8)


# COLOR GRADIENTS TO DETECT ME AND A YELLOW POINT ON THE MAP FROM A DRONE
ME_LOW = np.array([60, 151, 72], dtype=np.uint8)
ME_HIGH = np.array([86, 215, 103], dtype=np.uint8)
POINT_LOW = np.array([5, 153, 155], dtype=np.uint8)
POINT_HIGH = np.array([7, 216, 216], dtype=np.uint8)


MIN_BORDER_COLOR_MINIMAP = np.array([97, 97, 97], dtype=np.uint8)
MAX_BORDER_COLOR_MINIMAP = np.array([150, 150, 150], dtype=np.uint8)

IS_BIG_MAP = True
