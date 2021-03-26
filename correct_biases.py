def correct_biases(ft_meas, ft_bias, ang_bias, gravity_bias):
    '''
    Inputs:
    ft_meas = measured ft data, (6, N)
    ft_bias = sensor bias (6, )
    ang_bias = correction angle (deg) (scalar)
    gravity_bias = static forces after ang_bias correction (6, )

    Output:
    ft_meas = measured ft data with ft_bias and gravity_bias removed, angle corrected and converted to normal coordinate frame
    '''
    