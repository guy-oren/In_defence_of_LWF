from .wide_resnet import wide_resnet32


def Net(inputsize, taskcla):
    c_in, _, _ = inputsize
    return wide_resnet32(c_in, taskcla, 1, zero_init_residual=True)
