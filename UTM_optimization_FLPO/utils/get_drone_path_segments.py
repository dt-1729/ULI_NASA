def get_drone_path_segments(drone, route, stations):
    path_segments = []
    current_location = drone[0]

    for station_index in route:
        if station_index == len(stations):
            next_location = drone[1]
        else:
            next_location = stations[station_index]

        path_segments.append((current_location, next_location))
        current_location = next_location

    return path_segments