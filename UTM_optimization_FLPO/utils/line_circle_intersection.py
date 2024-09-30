def line_circle_intersection(circle_center, radius, line_start, line_end):
    circle_center = np.array(circle_center)
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    d = line_end - line_start
    f = line_start - circle_center
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius**2
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None

    discriminant = np.sqrt(discriminant)
    t1 = (-b - discriminant) / (2 * a+1e-8)
    t2 = (-b + discriminant) / (2 * a+1e-8)

    if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
        return None

    intersection1 = line_start + t1 * d
    intersection2 = line_start + t2 * d

    return intersection1, intersection2