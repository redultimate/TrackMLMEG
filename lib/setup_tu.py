from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension('tracking_utility_c', ['tracking_utility_c.pyx'], include_dirs = [numpy.get_include()])]   #assign new.pyx module in setup.py.

setup(
      name        = 'tracking_utility_c app',
      cmdclass    = {'build_ext':build_ext},
      ext_modules = ext_modules
      )
