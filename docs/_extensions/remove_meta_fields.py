"""Remove meta fields

Replace all ``:meta ...:`` info fields with a comment before being processed by
Sphinx so that there's no empty field list in the output (which results in an
empty space).

"""
# Roughly equivalent to ``reversed(list(enumerate(sequence)))``
def _enumerate_reversed(sequence):
    return ((i, sequence[i]) for i in range(len(sequence))[::-1])

def _remove_meta_fields(app, what, name, obj, options, lines):
    for i, line in _enumerate_reversed(lines):
        if line.startswith(":meta") and line.endswith(":"):
            lines[i:i+1] = ["", f'.. // {line}', ""]

def setup(app):
    """Sphinx extension entry point"""
    app.connect("autodoc-process-docstring", _remove_meta_fields)
    return {
        # Probably parallel-read safe
        "parallel_read_safe": True,
    }
