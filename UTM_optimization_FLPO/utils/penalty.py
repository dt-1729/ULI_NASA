def penalty(x):
    mask = x<=0
    y = my_inf*(x**2)
    y[mask] = 0
    return y