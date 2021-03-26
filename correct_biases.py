def correct_biases(ft_meas, ft_bias, ang_bias, gravity_bias):
    '''
    Inputs:
    ft_meas = measured ft data, (6, N)
    ft_bias = sensor bias (6, )
    ang_bias = correction ang_bias (deg) (scalar)
    gravity_bias = static forces after ang_bias correction (6, )

    Output:
    ft_meas = measured ft data with ft_bias and gravity_bias removed, ang_bias corrected and converted to normal coordinate frame
    '''

    import numpy as np

    # column vectors and stuff
    ft_bias = ft_bias.reshape((6, 1))
    ang_bias = np.squeeze(ang_bias) * np.pi / 180
    gravity_bias = gravity_bias.reshape((6, 1))

    ft_meas = ft_meas - ft_bias  # remove bias in sensor coordinates

    # rotate to offset sensor coordinates by ang_bias
    # note that this rotation matrix is actually the transpose of the normal one
    rotz_angle = [[np.cos(ang_bias), np.sin(ang_bias), 0], [-np.sin(ang_bias), np.cos(ang_bias), 0], [0, 0, 1]]
    ft_meas = np.dot([[rotz_angle, np.zeros((3, 3))], [np.zeros((3, 3)), rotz_angle]], ft_meas)
