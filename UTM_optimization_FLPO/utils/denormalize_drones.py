def denormalize_drones(normalized_drones, min_max_values):
    lb, ub = min_max_values

    # Denormalize coordinates
    denormalized_drones = [(((x1 * (ub - lb) + lb), (y1 * (ub - lb) + lb)),
                             ((x2 * (ub - lb) + lb), (y2 * (ub - lb) + lb)),
                             charge)
                            for ((x1, y1), (x2, y2), charge) in normalized_drones]

    return denormalized_drones