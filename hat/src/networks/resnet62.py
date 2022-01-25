from .wide_resnet import wide_resnet62


def Net(inputsize, taskcla):
    c_in, _, _ = inputsize
    return wide_resnet62(c_in, taskcla, 1, zero_init_residual=True)
