"""Better Sphinx subsections

Specify an explicit ID for a section with normal `reST inline targets`_::

    _`v1.2.3` (2022-03-19)
    ----------------------
    The permalink is ``#v1-2-3`` instead of ``#v1-2-3-2022-03-19``.

Trailing inline targets aren't shown in the section text (useful if the title
doesn't have `text that can be a target`_)::

    1.2.3 - 2022-03-19 _`v1.2.3`
    ----------------------------
    The header is different but the permalink is still ``#v1-2-3``.

If there's multiple targets for some reason, the last one is used::

    _`Released 2022-03-19`, _`v1.2.3`, New Features
    -----------------------------------------------
    There are multiple targest but the permalink is still ``#v1-2-3``.

In case the space between the title and the inline target is causing issues,
you can `escape the space`_ so that it doesn't appear in the final output::

    1.2.3 - 2022-03-19\ _`v1.2.3`
    -----------------------------
    Now there's no space after "2022-03-19" in the title.

Warning:
    This works by monkey-patching docutils' ``RSTState.new_subsection`` and
    probably relies on implementation details. This can and probably will break
    without warning.

.. _reST inline targets: https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#inline-internal-targets
.. _text that can be a target: https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#escaping-mechanism
.. _escape the space: https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#escaping-mechanism

"""
from docutils import nodes
from docutils.nodes import fully_normalize_name as normalize_name
from docutils.parsers.rst.states import RSTState

# Code between [START] and [END] is new. All other code in this function
# (excluding empty lines) is from docutils 0.17.1.
def _better_subsection(self, title, lineno, messages):
    memo = self.memo
    mylevel = memo.section_level
    memo.section_level += 1
    section_node = nodes.section()
    self.parent += section_node
    textnodes, title_messages = self.inline_text(title, lineno)

    # [START]
    # The last inline implicit target in the title sets the section ID
    for target_node in reversed(textnodes):
        if target_node.tagname == "target":
            break
    else:
        target_node = None
    # A trailing inline target specifies the ID (no text content)
    if target_node and target_node is textnodes[-1]:
        target_node.children.clear()
    # [END]

    titlenode = nodes.title(title, '', *textnodes)
    name = normalize_name(titlenode.astext())
    section_node['names'].append(name)
    section_node += titlenode
    section_node += messages
    section_node += title_messages
    self.document.note_implicit_target(section_node, section_node)

    # [START]
    if target_node:
        # Swap section and target node attributes. The section's ID is now the
        # target's ID (permalink is different). This also keeps the old section
        # name referable from older documents.
        for attr in ["ids", "names", "dupnames"]:
            target_node[attr], section_node[attr] = (
                section_node[attr], target_node[attr]
            )
        # Update the ID->node mapping (because they're swapped now)
        for target_id in target_node["ids"]:
            self.document.ids[target_id] = target_node
        for section_id in section_node["ids"]:
            self.document.ids[section_id] = section_node
    # [END]

    offset = self.state_machine.line_offset + 1
    absoffset = self.state_machine.abs_line_offset() + 1
    newabsoffset = self.nested_parse(
          self.state_machine.input_lines[offset:], input_offset=absoffset,
          node=section_node, match_titles=True)
    self.goto_line(newabsoffset)
    if memo.section_level <= mylevel: # can't handle next section?
        raise EOFError              # bubble up to supersection
    # reset section_level; next pass will detect it properly
    memo.section_level = mylevel

def setup(app):
    """Sphinx extension entry point"""
    RSTState.new_subsection = _better_subsection
    return {
        # Probably parallel-read safe (set up happens once)
        "parallel_read_safe": True,
    }
