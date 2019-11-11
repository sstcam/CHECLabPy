from numba import njit


@njit(fastmath=True)
def welfords_online_algorithm(sample, mean, count, m2):
    """
    Obtain the next iteration for Welford's online algorithm to stably
    calculate the running mean and standard deviation

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    Parameters
    ----------
    sample : float
        Sample to add to the calculation
    mean : float
        Latest value for the mean
    count : int
        Amount of values contributed so far
    m2 : float
        Sum of squares of differences from the current mean

    Returns
    -------
    mean : float
    count : int
    m2 : float
    """
    delta = sample - mean
    count += 1
    mean += delta / count
    delta2 = sample - mean
    m2 += delta * delta2
    return mean, count, m2
