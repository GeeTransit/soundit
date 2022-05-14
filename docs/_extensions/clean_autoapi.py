"""Remove unused files in the AutoAPI directory

When using ``autoapi_keep_files``, removing a module leaves an unused document
which errors from trying to access metadata about an object AutoAPI no longer
tracks. This extension removes extra files so these errors doesn't happen.
Empty directories are also removed.

"""
import os
from pathlib import Path

from sphinx.util import logging

_logger = logging.getLogger("clean_autoapi")

# Removes all unknown files in the AutoAPI directory
def _clean_autoapi(app):
    config = app.config
    env = app.env
    assert isinstance(config.clean_autoapi, dict)

    # Absolute path to the AutoAPI directory
    root = Path(app.srcdir, config.autoapi_root).resolve()
    srcdir = Path(app.srcdir).resolve()
    if root in [srcdir, *srcdir.parents]:
        if not config.clean_autoapi.get("unsafe"):
            raise RuntimeError(
                "autoapi_root is at or outside the source directory. no"
                " cleaning will be done since config.clean_autoapi['unsafe']"
                " is not true."
            )
        _logger.warning(
            "autoapi_root is at or outside the source directory",
            type="clean_autoapi",
            subtype="root",
        )

    # Create set of filenames that come from AutoAPI using the same logic:
    # For out_suffix, see:
    # https://github.com/readthedocs/sphinx-autoapi/blob/17ad4b988e6da41d3b9fc4fe253c19adc2fa3dc0/autoapi/extension.py#L156-L162
    # For rendered_paths, see:
    # https://github.com/readthedocs/sphinx-autoapi/blob/17ad4b988e6da41d3b9fc4fe253c19adc2fa3dc0/autoapi/mappers/base.py#L312-L339
    out_suffix = (
        ".rst" if ".rst" in config.source_suffix else
        ".txt" if ".txt" in config.source_suffix else
        next(iter(config.source_suffix))
    )
    rendered_paths = {
        # Assuming no objects get skipped because ``obj.render()`` was empty
        Path(obj.include_dir(root=str(root)), f'index{out_suffix}').resolve()
        for obj in env.autoapi_objects.values()
    }
    if config.autoapi_add_toctree_entry:
        rendered_paths.add(root / "index.rst")

    # Remove unused files
    for path in root.rglob("*"):
        if path.is_file() and path not in rendered_paths:
            _logger.debug(
                f'Removing unknown file under AutoAPI root: {path}',
                type="clean_autoapi",
                subtype="file",
            )
            path.unlink()

    # Remove empty directories (bottom up so we don't need to use recursion)
    for path, dirs, files in os.walk(root, topdown=False):
        if not dirs and not files:
            path = Path(path)
            _logger.debug(
                f'Removing empty directory under AutoAPI root: {path}',
                type="clean_autoapi",
                subtype="directory",
            )
            path.rmdir()

def setup(app):
    """Sphinx extension entry point"""
    # Priority is 600 to run after AutoAPI updates `app.env.autoapi_objects`
    app.connect("builder-inited", _clean_autoapi, priority=600)
    app.add_config_value("clean_autoapi", {}, rebuild=True, types=[dict])
    return {
        # Probably parallel-read safe
        "parallel_read_safe": True,
    }
