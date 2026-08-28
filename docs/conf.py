"""Docs config."""

import re

from sphinxemoji.sphinxemoji import load_emoji_codes

# add module to path
# sys.path.insert(0, os.path.abspath(".."))
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
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    # mathjax renders the math in the browser, i.e., unlike imgmath, it does not
    # require latex/dvipng to be installed in the docs environment
    "sphinx.ext.mathjax",
    "myst_nb",
    # the changelog headings use the github emoji shortcodes (e.g., `:sparkles:`) that
    # the changelog action writes, which sphinx does not render on its own
    "sphinxemoji.sphinxemoji",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    # myst-nb writes the executed notebooks here, which would otherwise be
    # picked up as (orphan) source documents
    "jupyter_execute",
    ".jupyter_cache",
]

# The master toctree document.
master_doc = "index"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for MyST markdown -----------------------------------------
# dollarmath/amsmath render the `$...$` and `$$...$$` math in the markdown pages
myst_enable_extensions = ["amsmath", "dollarmath"]
myst_heading_anchors = 3

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
    "navigation_with_keys": False,
}

# -- Options for the user guide notebooks ------------------------------
# the notebooks are saved with the pixi kernel (so that they can be run from
# jupyter lab), which does not exist within the environment that builds the docs
nb_kernel_rgx_aliases = {"^pixi-kernel-python3$": "python3"}

nb_execution_mode = "cache"
nb_execution_timeout = 300
# ACHTUNG: the annexes from a02 on are rendered from their stored outputs instead of
# being executed, since they either require data that is not shipped with the docs (the
# performance benchmarks) or belong to the preprocessing pipeline of the
# pylandstats-notebooks repository. The a01 FRAGSTATS comparison IS executed, as it
# checks that the computed metrics match the FRAGSTATS reference values.
# ACHTUNG: keep in sync with the `run-notebooks` task of pyproject.toml and with the
# `nbstripout` exclude of .pre-commit-config.yaml
nb_execution_excludepatterns = ["**/a0[2-9]-*.ipynb"]

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


# -- Rendering of the changelog emoji ----------------------------------
# the changelog is generated by the `requarks/changelog-action`, whose headings use the
# github emoji shortcodes (e.g., `:sparkles:`). Github renders them, but sphinx does not
# (the `sphinxemoji` extension only provides the `|:sparkles:|` substitution form), so
# substitute them with the actual emoji, reusing the `sphinxemoji` code table.
# ACHTUNG: this is restricted to the changelog because a blanket substitution would also
# rewrite legitimate colon-delimited text elsewhere, e.g., the `:members:` option of the
# autodoc directives
EMOJI_DOCNAMES = ("changelog",)
EMOJI_CODE_RE = re.compile(r":[a-z0-9_+-]+:")


def _substitute_emoji_shortcodes(app, docname, source):
    """Replace the github emoji shortcodes of the changelog with the actual emoji."""
    if docname not in EMOJI_DOCNAMES:
        return
    emoji_codes = load_emoji_codes()
    source[0] = EMOJI_CODE_RE.sub(
        lambda match: emoji_codes.get(match.group(), match.group()), source[0]
    )


def setup(app):
    """Set up the sphinx application."""
    app.connect("source-read", _substitute_emoji_shortcodes)
