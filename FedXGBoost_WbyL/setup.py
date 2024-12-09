from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

ext_options = {
    "compiler_directives": {
        "profile": True,
        'boundscheck': False,
        'wraparound': False
    },
    "annotate": True
}

extensions = [
    Extension("model.loss_function", ["model/loss_function.pyx"]),
]
setup(
    ext_modules = cythonize(extensions,
                            **ext_options),
    include_dirs=[numpy.get_include()]
)

