def my_log(m):
    return log(m, out=np.zeros_like(m), where=(m>0))