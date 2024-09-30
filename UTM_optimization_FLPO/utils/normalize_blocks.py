def normalize_blocks(blocks,min_max_values):
    lb, ub = min_max_values
    normalized_blocks = [(((x - lb) / (ub - lb), (y - lb) / (ub - lb)),
                          r/(ub - lb))
                         for ((x, y), r) in blocks]
    return normalized_blocks