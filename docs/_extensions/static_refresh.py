"""Rebuild Sphinx docs when static HTML files are modified"""

import pathlib

from sphinx.environment import CONFIG_OK
from sphinx.util import logging

_logger = logging.getLogger("static_refresh")

# Returns a list of docnames to potentially rebuild. Stores info under
# ``env.static_refresh``.
def _static_refresh(app):
    config = app.config
    env = app.env

    # Create set of static files
    files = {
        path.resolve()
        for root in getattr(config, "html_static_path", ())
        for path in pathlib.Path(app.srcdir, root).rglob("*")
        if path.is_file()
    }

    # Get last modification time
    mtime = max((file.stat().st_mtime for file in files), default=-1)

    # Save and replace previous info
    old_info = getattr(env, "static_refresh", None)
    env.static_refresh = {
        "files": sorted(str(file) for file in files),
        "mtime": mtime,
    }

    # Don't rebuild if nothing has changed
    if env.config_status == CONFIG_OK and env.static_refresh == old_info:
        return []

    # Rebuild the root document. This is a workaround because static files are
    # added after a build finishes, and builds only happen when a file has been
    # added or changed. For more info, see:
    # https://github.com/sphinx-doc/sphinx/issues/2090#issuecomment-177129813
    docname = config.root_doc
    _logger.debug(
        f'Rebuilding {docname} because static HTML files were modified',
        type="static_refresh",
    )
    return [docname]

def setup(app):
    """Sphinx extension entry point"""
    app.connect("env-get-outdated", lambda app, *_: _static_refresh(app))
    return {
        # Probably parallel-read safe
        "parallel_read_safe": True,
    }
