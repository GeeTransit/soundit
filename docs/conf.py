"""Configuration for the Sphinx documentation builder"""

import os
import re
import subprocess
import sys

if sys.version_info >= (3, 8):
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# - Project config

# Repo root relative to this file's directory
root = ".."

# Helper function to return text of file relative to the repo root
def read(filename: str) -> str:
    with open(os.path.join(root, filename)) as file:
        return file.read()

# Single source the project name from pyproject.toml
project = tomllib.loads(read("pyproject.toml"))["project"]["name"]

# Single source copyright. The format is ([] denoting optional parts):
# Copyright [(c)] [2022[-present]] John Doe [<email>] [(website)]
year, holder = re.search(
    r'''
        Copyright[ ]*  # Locate "Copyright"
        (?:(?:\(c\)|\N{COPYRIGHT SIGN})[ ]*)?  # Optional sign
        # Year ranges separated by commas
        (?:
            (
                (?:\d+(?:[ ]*-[ ]*(?:\d+))?,[ ]*)*
                \d+(?:[ ]*-[ ]*(?:\d+|present))?
            )
            ,?
        )?
        [ ]*
        (?:by[ ]*)?
        (
            [^ \n,.]+
            (?:
                [ ]*[,.]?[ ]*
                (?!
                    [^ @]+@  # Don't match emails (have @ in them)
                    |All rights reserved  # Don't match boilerplate text
                    |[^ :]+://  # Don't match websites
                    |[<(]  # Don't match <email> or (website)
                )
                [^ \n,.]+
            )*
            [.]?
        )
    ''',
    read("LICENSE"),
    re.MULTILINE | re.VERBOSE | re.IGNORECASE,
).groups()
author = holder.strip()
copyright = f"{year}, {author}" if year else author

# Single source the project version from the installed package's version
version = importlib_metadata.version(project)
try:
    # Try getting the package's version from Hatch's CLI if possible. The
    # version is different when a new commit is added but the installed package
    # isn't updated (editable installs don't have dynamic versions).
    version = subprocess.run(
        ["hatch", "version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()[-1]
except subprocess.CalledProcessError:
    pass
release = version

# - Sphinx config

# Allow less verbose references (e.g. `list` instead of :py:class:`list`)
default_role = "any"

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# - Extension config

extensions = []

# Custom extensions are located in _extensions
sys.path.append(os.path.abspath("_extensions"))

# View code from the docs
extensions += ["sphinx.ext.viewcode"]

# Set section permalink
extensions += ["sphinx_better_subsection"]

# Include output from programs (like `gitchangelog`)
extensions += ["program_include"]

# Redirect relocated pages
extensions += ["sphinx_reredirects"]
redirect_html_template_file = "_templates/reredirects/template.html"
redirects = {
    "_generated/CHANGELOG/index": "../../changelog/",
    "_generated/api/soundit/index": "../../../api/soundit/",
}

# Recursively generate docs using AutoAPI
extensions += ["autoapi.extension"]
autoapi_root = "api"  # start documentation from api/
autoapi_dirs = ["../src"]  # directory to document
autoapi_template_dir = "_templates/autoapi"
# Document most members
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]

# Generate docs from docstrings
extensions += ["sphinx.ext.autodoc"]

# Remove :meta: fields to prevent empty field lists
# Note: Must come after sphinx.ext.autodoc
extensions += ["remove_meta_fields"]

# Support Google style docstrings
extensions += ["sphinx.ext.napoleon"]

# Get last updated date from Git
extensions += ["sphinx_last_updated_by_git"]
git_last_updated_timezone = "America/Toronto"
git_untracked_show_sourcelink = True
git_untracked_check_dependencies = False

# - HTML output config

html_theme = "alabaster"
html_static_path = ["_static"]
html_css_files = [
    # Contains some minor CSS improvements to Alabaster
    "custom.css",
]
html_theme_options = {
    # Allow larger windows to have wider text (default is a hardcoded width)
    "page_width": "auto",
    "body_max_width": "auto",
    # Don't have a minimum width
    "body_min_width": "0",
    # Add GitHub "Watch" button
    "github_button": True,
    "github_user": "GeeTransit",
    "github_repo": "soundit",
    "github_type": "star",
}
html_permalinks_icon = "#"  # More consistent with other sites
html_last_updated_fmt = "%B %d, %Y"  # strftime format
