def calculate_shorter_arc(circle_center, radius, point1, point2):
    def to_polar(point):
        x, y = point - circle_center
        return np.arctan2(y, x)

    theta1 = to_polar(point1)
    theta2 = to_polar(point2)

    if theta1 > theta2:
        theta1, theta2 = theta2, theta1

    arc_length1 = theta2 - theta1
    arc_length2 = 2 * np.pi + theta1 - theta2

    if arc_length1 < arc_length2:
        return theta1, theta2
    else:
        return theta2, theta1 + 2 * np.pi