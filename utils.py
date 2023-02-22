import math
import numpy as np


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_of_vectors(a, b, c, d):
    angle = 1
    dotProduct = a * c + b * d
    # for three dimensional simply add dotProduct = a*c + b*d  + e*f
    modOfVector1 = math.sqrt(a * a + b * b) * math.sqrt(c * c + d * d)
    if modOfVector1 != 0:
        angle = dotProduct / modOfVector1

    angleInDegree = math.degrees(math.acos(angle))
    return angleInDegree


def vector_to_deg(vector):
    x, y = vector
    angle = math.atan2(y, x)
    angle = math.degrees(angle)
    if angle < 0:
        angle += 360
    return angle


def deg_to_unive(degree):
    radian = math.radians(degree)
    x = -math.cos(radian)
    y = math.sin(radian)
    vector = [x, y]
    return vector


def cord_inf(deg, val, arc):
    value = val
    if arc == 'x':
        if 90 > deg > 0:
            value = val
        elif 180 > deg > 90:
            value = -val
        elif 270 > deg > 180:
            value = -val
        else:
            value = val
    elif arc == 'y':
        if 180 > deg > 0:
            value = val
        else:
            value = -val
    return value


def plane_dist(travel):
    vehicle = travel
    interval = math.sqrt((vehicle[1] - vehicle[3]) ^ 2 +
                         (vehicle[2] - vehicle[3]) ^ 2)
    return interval


def in_zone(travel, zone):
    # [track_id, x_cur, y_cur, x_pre, y_pre, -1, 0]

    vehicle = travel
    # presents = False
    x, y = vehicle[1], vehicle[2]
    x1, y1 = zone[0]
    x2, y2 = zone[1]
    x3, y3 = zone[3]
    x4, y4 = zone[2]
    # print(x1, y1, x2, y2, x3, y3, x4, y4)
    # print(x, y)
    if (x1 < x < x2) and (y3 < y < y1):
        presents = True
    elif (x2 < x < x3) and (y3 < y < y2):
        presents = True
    elif (x3 < x < x4) and (y4 < y < y3):
        presents = True
    elif (x4 < x < x1) and (y4 < y < y1):
        presents = True
    else:
        presents = False

    return presents


def point_in_polygon(travel, poly):

    vehicle = travel
    x, y = vehicle[1], vehicle[2]
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# print(deg_to_unive(120))
