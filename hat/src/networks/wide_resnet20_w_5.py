from .wide_resnet import wide_resnet20


def Net(inputsize, taskcla):
    c_in, _, _ = inputsize
    return wide_resnet20(c_in, taskcla, 5, zero_init_residual=True)
