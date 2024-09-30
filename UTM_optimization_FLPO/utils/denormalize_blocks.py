def denormalize_blocks(blocks,min_max_values):
    lb, ub = min_max_values
    denormalized_blocks = [(((x * (ub - lb) + lb), (y * (ub - lb) + lb)),

                             r* (ub - lb))
                            for ((x, y), r) in blocks]
    return denormalized_blocks