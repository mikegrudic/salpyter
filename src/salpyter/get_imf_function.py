from . import imfs
from .default_imf_params import DEFAULT_LOGMMIN, DEFAULT_LOGMMAX


def get_imf_function(imf_model: str):
    """Returns the IMF function associated with the input model name"""

    modelname = imf_model + "_imf"
    if hasattr(imfs, modelname):
        return getattr(imfs, modelname)

    elif "_bounds" in modelname and hasattr(imfs, modelname.replace("_bounds", "")):
        imf0 = getattr(imfs, modelname.replace("_bounds", ""))

        def imf_func(logm, params, logmmin=DEFAULT_LOGMMIN, logmmax=DEFAULT_LOGMMAX):
            return imfs.imf_with_bounds_params(logm, params, imf0)

        setattr(imfs, modelname, imf_func)  # add the _bounds imf function
        return getattr(imfs, modelname)


IMF_LIST = [k.replace("_imf", "") for k in imfs.__dict__.keys() if k[-4:] == "_imf"]
IMF_LIST += [k + "_bounds" for k in IMF_LIST]
