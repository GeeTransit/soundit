"""Configuration for the Sphinx documentation builder"""

import os
import re
import subprocess
import sys

from sphinx.util.docfields import GroupedField
from sphinx import addnodes

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
version = "".join(version.partition(".dev")[0:2])
release = version

# - Sphinx config

# Allow less verbose references (e.g. `list` instead of :py:class:`list`)
default_role = "any"

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# Less clutter in API docs
add_module_names = False
python_use_unqualified_type_names = True
maximum_signature_line_length = 80

def setup(app):
    # Adapted from:
    # https://github.com/sphinx-doc/sphinx/blob/4c664ae0b873af91b030a8da253959c0727e1c7a/doc/conf.py
    app.add_object_type(
        "confval", "confval",
        objname="configuration value",
        indextemplate="pair: %s; configuration value",
    )
    def parse_event(env, text, sig):
        m = re.match(r"([a-zA-Z-]+)\s*\((.*)\)", text)
        if not m:
            sig += addnodes.desc_name(text, text)
            return text
        name, args = m.groups()
        sig += [addnodes.desc_name(name, name)]
        params = addnodes.desc_parameterlist()
        for arg in map(str.strip, args.split(",")):
            params += [addnodes.desc_parameter(arg, arg)]
        sig += [params]
        return name
    app.add_object_type(
        "event", "event",
        "pair: %s; event",
        parse_event,
        doc_field_types=[GroupedField(
            "parameter",
            label="Parameters",
            names=["param"],
            can_collapse=True,
        )],
    )

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

# Rebuild docs when static HTML files are modified
extensions += ["static_refresh"]

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

# Process :meta: fields in AutoAPI (mimick autodoc behaviour)
extensions += ["autoapi_meta_fields"]

# Remove :meta: fields to prevent empty field lists
# Note: Must come after sphinx.ext.autodoc
extensions += ["remove_meta_fields"]

# Show strings in `Literal` annotations correctly in AutoAPI
extensions += ["autoapi_literal_annotations"]

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

# Link to other docs
extensions += ["sphinx.ext.intersphinx"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}

# Specify dependencies of a document
extensions += ["rebuild_when"]

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
