import numpy as np

def xscale(features):
    # scale parameter.
    # scale_para = list()
    sfeatures = np.zeros_like(features)
    for i in range(len(features)):
        # max and min.
        ma = np.max(features[i])
        mi = np.min(features[i])
        # standar.
        feature = (features[i] - mi) / (ma - mi)
        # save.
        # scale_para.append()
        sfeatures[i] = feature

    return sfeatures

def yscale(content):
    scontend = np.zeros_like(content)
    ma = np.max(content)
    mi = np.min(content)
    scontend = (content - mi) / (ma - mi)

    return (scontend, (ma, mi))
