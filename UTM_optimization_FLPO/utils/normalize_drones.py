def normalize_drones(drones):
    lb=np.min([np.min((drone[0],drone[1])) for drone in drones])
    ub=np.max([np.max((drone[0],drone[1])) for drone in drones])
    normalized_drones = [(((x1 - lb) / (ub - lb), (y1 - lb) / (ub - lb)),
                          ((x2 - lb) / (ub - lb), (y2 - lb) / (ub - lb)),
                          charge)
                         for ((x1, y1), (x2, y2), charge) in drones]

    return normalized_drones, (lb,ub)