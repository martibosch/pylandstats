"""Docs config."""

import os
import sys

# add module to path
sys.path.insert(0, os.path.abspath(".."))

import pylandstats as pls  # noqa: E402

# -- Project information -----------------------------------------------------
project = "pylandstats"
author = "Martí Bosch"

__version__ = pls.__version__
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "m2r2",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.imgmath",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The master toctree document.
master_doc = "index"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_rtd_theme'
html_theme = "pydata_sphinx_theme"
# html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "github_url": "https://github.com/martibosch/pylandstats",
    "twitter_url": "https://twitter.com/mortybosch",
    "pygment_light_style": "tango",
}

# exclude patterns from sphinx-build
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "pylandstats.tex",
        "pylandstats documentation",
        "Martí Bosch",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "pylandstats", "pylandstats documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "pylandstats",
        "PyLandStats Documentation",
        author,
        "pylandstats",
        "An open-source Pythonic library to compute landscape metrics",
        "Miscellaneous",
    ),
]
