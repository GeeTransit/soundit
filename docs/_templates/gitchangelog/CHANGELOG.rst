<%
def indent(indent: str, string: str):
    """Indents all lines but the first with the given string"""
    return "".join(
        line if i == 0 else indent + line
        for i, line in enumerate(string.splitlines(keepends=True))
    )
def underline(char: str, string: str):
    """Returns the string followed by a line of the given char"""
    assert len(char) == 1
    return f'{string}\n{char * len(string)}'
%>\
% if data["title"]:
${underline("=", data["title"])}
% endif
% for version in data["versions"]:
<%
if version["tag"]:
    ref = version["tag"]
    title = f'_`{ref}` ({version["date"]})'
else:
    ref = "Unreleased"
    title = f'{opts["unreleased_version_label"]} (_`{ref}`)'
%>\

${underline("-", title)}
% for section in version["sections"]:
% if section["label"] != "Other" or len(version["sections"]) > 1:

${underline("~", f'{section["label"]}\ _`{ref}-{section["label"]}`')}
% endif

.. rst-class:: compact

% for commit in section["commits"]:
- ${indent("  ", commit["subject"])} [${", ".join(commit["authors"])}]
% endfor
% endfor
% endfor
