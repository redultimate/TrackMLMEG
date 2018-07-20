from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension('seed_finder_loop3', ['seed_finder_loop3.pyx'], include_dirs = [numpy.get_include()])]   #assign new.pyx module in setup.py.

setup(
      name        = 'seed_finder_loop3 app',
      cmdclass    = {'build_ext':build_ext},
      ext_modules = ext_modules
      )
