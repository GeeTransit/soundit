r"""Include a program's output in your Sphinx docs

Provides a new directive named `program-include`. Similar to
sphinxcontrib-programoutput__'s ``program-output`` directive except the output
isn't wrapped in a literal block. See `ProgramIncludeDirective` for more info.

__ https://sphinxcontrib-programoutput.readthedocs.io/en/latest/

Example:

    .. code-block:: rst

        .. program-include:: python --version

        .. program-include:: python -c "print('- unordered\n- list')"

        .. program-include:: python -c "print('1. numbered\n2. list')"

        .. program-include::
            python -c "import os; print(os.environ['SUS'])"
            :env: SUS=some **bold** and *italicized* text

    This outputs the following:

        .. program-include:: python --version

        .. program-include:: python -c "print('- unordered\n- list')"

        .. program-include:: python -c "print('1. numbered\n2. list')"

        .. program-include::
            python -c "import os; print(os.environ['SUS'])"
            :env: SUS=some **bold** and *italicized* text

"""
import subprocess
import os
import shlex

from docutils.nodes import container
from docutils.parsers.rst import directives
from docutils.statemachine import ViewList
from sphinx.util.docutils import SphinxDirective, switch_source_input
from sphinx.util.nodes import nested_parse_with_titles

class ProgramIncludeDirective(SphinxDirective):
    """Include a program's output

    .. rst:directive:: .. program-include:: command

        The output can include titles (thanks to Sphinx__) where all titles are
        treated as different from ones in the document (similar to docstrings
        under autodoc). This means the command can output titles without
        needing to know the document's title decoration preferences.

        __ https://www.sphinx-doc.org/en/master/extdev/markupapi.html#parsing-directive-content-as-rest

        All options are optional.

        .. rst:directive:option:: cwd: current directory

            The current working directory while running the program. Can be
            relative to the current file or absolute from the source directory.
            Defaults to the source directory (``"/"`` or `app.srcdir
            <sphinx.application.Sphinx.srcdir>`).

        .. rst:directive:option:: env: environment variables

            Extra environment variables to set. Each line should be of the form
            ``KEY=VALUE``.

        .. rst:directive:option:: encoding: output encoding

            The encoding of the program output. Defaults to ``"utf-8"``.

        .. rst:directive:option:: shell

            If specified, ``shell=True`` will be passed to `subprocess.run` to
            run the command via the system shell. Default is to pass
            ``shell=False`` and to split the command using `shlex.split`.

    """
    required_arguments = 1
    final_argument_whitespace = True
    option_spec = {
        "cwd": directives.unchanged,
        "env": directives.unchanged,
        "encoding": directives.encoding,
        "shell": directives.flag,
    }

    def run(self):
        options = self.options
        command = self.arguments[0]

        args = command.replace("\n", " ")
        shell = "shell" in options
        cwd = self.env.relfn2path(options.get("cwd", "/"))[1]
        env = os.environ.copy()
        for line in options.get("env", "").splitlines():
            name, sep, value = line.partition("=")
            env[name] = value
        if not shell:
            args = shlex.split(args)

        process = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            shell=shell,
            capture_output=True,
        )
        if process.returncode != 0:
            raise self.error(f'program-include returned {process.returncode}')
        try:
            output = process.stdout.decode(options.get("encoding", "utf-8"))
        except UnicodeDecodeError:
            raise self.error("program-include decode error")

        result = ViewList(output.splitlines(), command)
        with switch_source_input(self.state, result):
            node = container()
            nested_parse_with_titles(self.state, result, node)
            return node.children

def setup(app):
    """Sphinx extension entry point"""
    app.add_directive("program-include", ProgramIncludeDirective)
    return {
        # Probably parallel-read safe
        "parallel_read_safe": True,
    }
