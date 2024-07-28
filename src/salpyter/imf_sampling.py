"""Routine for generating samples from a certain IMF"""

def imf_samples(num_samples, imf, params, logmmin=-3, logmmax=2):
    """Uses rejection sampling to sample from an IMF"""
    x = np.random.rand(num_samples)
    logm = logmmin + (logmmax - logmmin) * x
    imf_val = imf(logm, params)
    imf_max = imf_val.max()
    y = imf_max * np.random.rand(num_samples)
    return 10 ** logm[y < imf_val]
