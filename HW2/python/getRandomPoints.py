import numpy as np

def get_random_points(I, alpha):

    # -----fill in your implementation here --------
    img_size = I.shape
    x_pos = np.random.random_integers(0, img_size[0], alpha)
    y_pos = np.random.random_integers(0, img_size[1], alpha)

    # points = np.vstack([x_pos, y_pos])
    points = []
    points.append([x_pos,y_pos])
    # ----------------------------------------------

    return points
