"""Fix `typing.Literal` annotations in AutoAPI

This extension stops strings from being resolved if they are inside a
``Literal`` annotation.

Strings in type annotations usually mean forward references. In ``Literal``\s
however, they are the string literal, not forward references. AutoAPI still
tries to resolve them however.

Warning:
    This monkey patches an internal method of AutoAPI. It may break at any
    time. Only ``sphinx-autoapi==1.8.4`` has been tested to work.

"""
import astroid
import autoapi

# Returns a wrapper function that doesn't resolve types inside Literal[...]
def _patch_resolve_annotation(old_resolve_annotation):
    def _patched_resolve_annotation(annotation):
        if isinstance(annotation, astroid.Subscript):
            value = old_resolve_annotation(annotation.value)
            slice_node = annotation.slice
            if isinstance(slice_node, astroid.Index):
                slice_node = slice_node.value
            if value == "Literal":
                if isinstance(slice_node, astroid.Tuple):
                    slice_ = ", ".join(e.as_string() for e in slice_node.elts)
                else:
                    slice_ = slice_node.as_string()
                # `value` is already resolved so no need to resolve again
                return f'{value}[{slice_}]'
        # Fallback to original resolve_annotation
        return old_resolve_annotation(annotation)
    return _patched_resolve_annotation

def setup(app):
    """Sphinx extension entry point"""
    u = autoapi.mappers.python.astroid_utils
    u._resolve_annotation = _patch_resolve_annotation(u._resolve_annotation)
    return {
        # Probably parallel-read safe
        "parallel_read_safe": True,
    }
