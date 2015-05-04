from scipt.interpolate import interp1d

def season_interpolation(s1, s2):
    """
    Linearily interpolate two seasons arrays
    """

    x = np.arange(10)
