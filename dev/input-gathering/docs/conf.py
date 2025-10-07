# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinxcontrib.katex as katex
from importlib.metadata import version
import senxor
#import sphinx_bootstrap_theme
#sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'PySenXor'
copyright = '2020-2022, Stanislav Markov, Meridian Innovation Ltd. HK'
author = 'Stanislav Markov, Meridian Innovation Ltd. HK'

# The full version, including alpha/beta/rc tags
version = version('pysenxor')
# for example take major/minor
release = version

# -- General configuration ---------------------------------------------------

# Ensure we have automatically numbered figure references
# so we can use :numref:`label`
numfig = True
numfig_format = {'figure': 'Fig. %s'}
# The following  will yield Fig. X.i, where X is section number, if 
# section numbering is enabled via the :numbered: option of the toctree
# directive; to avoid section numbers set to 0.
numfig_secnum_depth = 1

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.coverage',
    'sphinxcontrib.katex',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.ifconfig',
    'sphinx.ext.intersphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store'] 

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
