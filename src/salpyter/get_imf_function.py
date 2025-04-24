from . import imfs
import numpy as np


def get_imf_function(imf_model: str):
    """Returns the IMF function associated with the input model name"""

    modelname = imf_model + "_imf"
    if hasattr(imfs, modelname):
        return getattr(imfs, modelname)

    elif "_bounds" in modelname and hasattr(imfs, modelname.replace("_bounds", "")):
        imf0 = getattr(imfs, modelname.replace("_bounds", ""))

        def imf_func(logm, params, logmmin=-np.inf, logmmax=4):
            return imfs.imf_with_bounds_params(logm, params, imf0)

        setattr(imfs, modelname, imf_func)  # add the _bounds imf function
        return getattr(imfs, modelname)


def list_implemented_imfs():
    imf_list = [k.replace("_imf", "") for k in imfs.__dict__.keys() if k[-4:] == "_imf"]
    imf_list += [k + "_bounds" for k in imf_list]
    return imf_list
