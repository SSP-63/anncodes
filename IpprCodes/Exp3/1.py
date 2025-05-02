import numpy as np

def translate(point, tx, ty):
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]])
    p = np.array([point[0], point[1], 1])
    new_p = np.dot(T, p)
    return new_p[:2]

point = (2, 3)
translated_point = translate(point, 4, 5)
print("Translated Point:", translated_point)
