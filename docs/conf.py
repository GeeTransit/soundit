"""Configuration for the Sphinx documentation builder"""

import os
import re
import subprocess
import sys

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
    r"^Copyright \(c\) ([^ ]+) ([^\n]+)$",
    read("LICENSE"),
    re.MULTILINE | re.IGNORECASE,
).groups()
author = holder.strip()
copyright = f"{year}, {author}"

# Single source the project version from the Hatch CLI. The
# version is different when a new commit is added but the installed package
# isn't updated (editable installs don't have dynamic versions).
version = subprocess.run(
    ["hatch", "version"],
    capture_output=True,
    encoding="utf-8",
    check=True,
).stdout.splitlines()[-1]
release = version

# - Sphinx config

# Allow less verbose references (e.g. `list` instead of :py:class:`list`)
default_role = "any"

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# - Extension config

extensions = []

# Path to custom extensions
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
autoapi_keep_files = True
# Document most members
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]

# Clean leftover docs in AutoAPI folder (such as from removing a module)
extensions += ["clean_autoapi"]

# Generate docs from docstrings
extensions += ["sphinx.ext.autodoc"]

# Remove :meta: fields to prevent empty field lists
# Note: Must come after sphinx.ext.autodoc
extensions += ["remove_meta_fields"]

# Support Google style docstrings
extensions += ["sphinx.ext.napoleon"]
# Most themes have styles for special admonitions
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# Get last updated date from Git
extensions += ["sphinx_last_updated_by_git"]
git_last_updated_timezone = "America/Toronto"
git_untracked_show_sourcelink = True
git_untracked_check_dependencies = False

# - HTML output config

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = [
    # Contains some minor CSS improvements
    "custom.css",
]
html_theme_options = {
    # Allow navigating using arrow keys
    "navigation_with_keys": True,
}
html_context = {
    # Show link to GitHub repo
    "display_github": True,
    "github_user": "GeeTransit",
    "github_repo": "soundit",
}
html_permalinks_icon = "#"  # More consistent with other sites
html_last_updated_fmt = "%B %d, %Y"  # strftime format
