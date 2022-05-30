"""Check meta fields in AutoAPI

If ``:meta private:`` exists in a docstring, the member is hidden even if it's
public. If ``:meta public:`` exists, the member is shown even if it's private.
Only the last field is used.

"""
_META_HIDE = ":meta private:"
_META_SHOW = ":meta public:"

# Roughly equivalent to ``reversed(list(enumerate(sequence)))``
def _enumerate_reversed(sequence):
    return ((i, sequence[i]) for i in range(len(sequence))[::-1])

# Skips objects with :meta private: and doesn't skip for :meta public:
def _check_meta_fields(lines):
    for i, line in _enumerate_reversed(lines):
        if line.endswith(_META_HIDE):
            return True
        if line.endswith(_META_SHOW):
            return False

# Connect to `autoapi-skip-member`
def _autoapi_skip_member_check(app, what, name, obj, skip, options):
    return _check_meta_fields(obj.docstring.splitlines())

def setup(app):
    """Sphinx extension entry point"""
    app.connect("autoapi-skip-member", _autoapi_skip_member_check)
    return {
        # Probably parallel-read safe
        "parallel_read_safe": True,
    }
