"""Remove meta fields

Remove :meta: info fields before being processed by Sphinx so that there's no
empty field list in the output (which results in an empty space).

"""
def _remove_meta_fields(app, what, name, obj, options, lines):
    lines[:] = [line for line in lines if ":meta" not in line]

def setup(app):
    """Sphinx extension entry point"""
    app.connect("autodoc-process-docstring", _remove_meta_fields)
    return {
        # Probably parallel-read safe
        "parallel_read_safe": True,
    }
