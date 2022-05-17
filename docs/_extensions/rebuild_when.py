"""Rebuild documents when other files change

Provides a `rebuild-when` directive to specify when the current document
will be rebuilt. See `RebuildWhenDirective` for more info.

Example:
    .. code-block:: rst

        .. rebuild-when:: file-change

            gitchangelog_conf.py

        .. rebuild-when:: program-change

            hatch version

"""
import subprocess
import os
import shlex
import json
import collections
import dataclasses
import pathlib
from typing import Any

from docutils.parsers.rst import directives
from sphinx.environment.collectors import EnvironmentCollector
from sphinx.util.docutils import SphinxDirective
from sphinx.util import logging

_logger = logging.getLogger("rebuild_when")

def handle_program_change(app, condition):
    """Handle ``program-change`` rebuilds

    .. code-block:: rst

        .. rebuild-when:: program-change
            :cwd: current directory
            :env: environment variables
            :shell:

            command to run

    The current document is rebuilt when the program's output is different. The
    entire body is treated as a command and split using `shlex.split` (unless
    ``:shell:`` was used). Output is taken from STDOUT and previous output (if
    any) is fed into STDIN. This allows programs to store state.

    ``:cwd:`` specifies the current working directory while running
    the program. Can be relative to the current file or absolute from the
    source directory. Defaults to the source directory (``"/"`` or `app.srcdir
    <sphinx.application.Sphinx.srcdir>`).

    ``:env:`` specifies extra environment variables to set. Each
    line should be of the form ``KEY=VALUE``.

    If ``:shell:`` is specified, ``shell=True`` will be passed to
    `subprocess.run` to run the command via the system shell. Defaults to
    passing ``shell=False`` and splitting the command using `shlex.split`.

    """
    if condition["type"] != "program-change":
        return

    docname = condition["docname"]
    options = condition["options"]
    content = condition["content"]

    def f():
        if "initial" not in condition:
            state = condition["state"]

        else:
            if not content:
                raise ValueError("expected content for ``program-change``")

            args = content.replace("\n", " ")
            shell = "shell" in options
            if shell and options["shell"]:
                raise ValueError("no argument allowed for ``:shell:``")
            cwd = options.get("cwd", "/")
            env = {}
            for line in options.get("env", "").splitlines():
                name, _, value = line.partition("=")
                env[name] = value
            if not shell:
                args = shlex.split(args)

            state = {
                "args": args,
                "shell": shell,
                "cwd": cwd,
                "env": env,
            }

        args = state["args"]
        shell = state["shell"]
        cwd = state["cwd"]
        env = state["env"]
        input_ = state.get("output", b"")

        cwd = app.env.relfn2path(cwd, docname)[1]
        env = {**os.environ, **env}
        process = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            shell=shell,
            input=input_,
            capture_output=True,
            check=True,
        )
        output = process.stdout

        return {
            "rebuild": output != state.get("output"),
            "state": {
                **state,
                "output": output,
            },
        }

    return {"function": f}

def handle_file_change(app, condition):
    """Handle ``file-change`` rebuilds

    .. code-block:: rst

        .. rebuild-when:: file-change

            files to check

    The current document is rebuilt when the specified files are modified. Each
    line should contain a path relative to the current document or absolute
    from the source directory. The last modified time will be checked when
    rebuilding.

    Globs will be expanded and matching files will trigger a rebuild when
    modified. New or deleted files that match a glob will also trigger a
    rebuild.

    """
    if condition["type"] != "file-change":
        return

    docname = condition["docname"]
    content = condition["content"]

    def f():
        if "initial" not in condition:
            state = condition["state"]

        else:
            if not content:
                raise ValueError("expected content for ``file-change``")

            docpath = app.env.doc2path(docname, base=False)
            patterns = []
            for pattern in content.splitlines():
                # Faster way to match everything under a path (``_static/**``)
                if pattern.endswith("**"):
                    pattern = f'{pattern}/*'
                if pattern.startswith("/"):
                    patterns.append(("absolute", pattern[1:]))
                else:
                    patterns.append(("relative", pattern))

            state = {
                "docpath": docpath,
                "patterns": patterns,
            }

        docpath = state["docpath"]
        patterns = state["patterns"]

        root = pathlib.Path(app.srcdir).resolve()
        current = root.joinpath(docpath).resolve().parent

        # Create set of matched files
        files = sorted({
            # Moving docs shouldn't trigger a reload (relative better)
            os.path.relpath(path.resolve(), root)
            for kind, pattern in patterns
            for path in (root if kind == "absolute" else current).glob(pattern)
            if path.is_file()
        })

        # Get last modification time
        mtime = max((
            root.joinpath(file).stat().st_mtime
            for file in files
        ), default=-1)

        return {
            # Don't rebuild if nothing has changed
            "rebuild": (
                files != state.get("files")
                and mtime != state.get("mtime")
            ),
            "state": {
                **state,
                # Save and replace previous info
                "files": files,
                "mtime": mtime,
            },
        }

    return {"function": f}

def handle_always(app, condition):
    """Handle ``always`` rebuilds

    .. code-block:: rst

        .. rebuild-when:: always

    The current document is always rebuilt.

    """
    if condition["type"] != "always":
        return

    def f():
        if "initial" in condition:
            if condition["content"]:
                raise ValueError("expected no content for ``always``")
        return {"rebuild": True}

    return {"function": f}

class UnhandledRebuildCondition(Exception):
    """Raised when no handlers match a rebuild condition"""
    pass

# TODO: make it possible to emit errors alongside a result
def run_condition(app, condition):
    r"""Match and run against a rebuild condition

    Raises:
        UnhandledRebuildCondition: if no handlers match the condition.

    Arguments:
        app: the `Sphinx <sphinx.application.Sphinx>` instance
        condition: the rebuild condition dict

    ``app`` and ``condition`` are passed to `match-rebuild`. ``match`` is
    the return value of a handler.

    The earliest (smallest priority) ``match["function"]`` is called with no
    arguments and should return a dict of the following form (returning `None`
    is treated as returning an empty dict, and no keys are required):

    .. code-block:: python

        result = {
            "rebuild": bool,  # True if the document should be rebuilt
            "state": Any,  # state to be passed into the next invocation
        }

    This dict will be returned.

    All ``match["after-function"]``\s earlier than the ``match["function"]``
    are called with ``result`` and should return a dict of the same form. Later
    after functions are called before earlier ones (larger before smaller
    priority).

    .. event:: match-rebuild(app, condition)

        ``app`` is the `Sphinx <sphinx.application.Sphinx>` instance.
        ``condition`` is the rebuild condition to match and is of the form:

        .. code-block:: python

            condition = {
                "docname": str,  # document name
                "type": str,  # directive type (after the .. rebuild-when::)
                "content": str,  # newline-separated content
                "options": dict[str, str|None],  # dict of directive options
                "state": NotRequired[Any],  # previous result["state"]
                "initial": NotRequired[Literal[True]],  # only for initial call
            }

        Handlers should return a dict of the following form (returning `None`
        is treated as returning an empty dict, and no keys are required):

        .. code-block:: python

            match = {
                "function": Callable[[], dict],
                "after-function": Callable[[dict], dict],
            }

    """
    matches = app.emit("match-rebuild", condition)

    after_functions = []
    for i, match in enumerate(matches):
        if match is None:
            match = {}
        if "function" in match:
            break
        if "after-function" in match:
            after_functions.append(match["after-function"])
    else:
        raise UnhandledRebuildCondition

    result = match["function"]()
    if result is None:
        result = {}
    for function in reversed(after_functions):
        result = function(result)
    return result

class RebuildWhenDirective(SphinxDirective):
    r"""Specify when the current document should be rebuilt

    .. rst:directive:: .. rebuild-when:: type

        For documentation on built-in types, see their handler functions (such
        as `handle_file_change`).

        Most data is part of the rebuild condition for handlers to use. See
        `run_condition` for more info.

    """
    @dataclasses.dataclass
    class _AllDict:
        """Internal dict-like class that maps all keys to one value"""
        value: Any
        def __contains__(self, key):
            return True
        def __getitem__(self, key):
            return self.value

    required_arguments = 1
    has_content = True
    option_spec = _AllDict(directives.unchanged)

    def run(self):
        """Handle an occurrence of this directive

        ``condition["initial"]`` is `True` during this initial call. Handlers
        can depend on this for code that runs only during document processing
        (such as `env.note_dependency
        <sphinx.environment.BuildEnvironment.note_dependency>` and similar).

        ``result["rebuild"]`` is also ignored for this initial call because the
        current document is already being rebuilt.

        """
        env = self.env
        app = env.app

        docname = env.docname
        if hasattr(env, "rebuild_states"):
            states = env.rebuild_states.get(docname)  # Can still be None
        else:
            states = None

        # The condition_json should identify a rebuild-when directive. It
        # should also be unique for different directives (hence why it's JSON
        # encoded for simple equality checking) and consistent between reruns
        # (hence ``sort_keys=True``).
        condition = {
            "docname": docname,
            "type": self.arguments[0],
            "content": "\n".join(self.content),
            "options": self.options,
        }
        condition_json = json.dumps(condition, separators=",:", sort_keys=True)

        if states is not None and condition_json in states:
            _logger.debug(
                f'Skipping identical rebuild condition: {condition_json}',
                type="rebuild_when",
            )
            # Indicate that the condition still exists
            states[condition_json].pop("remove", None)
            return []

        _logger.debug(
            f'Running new rebuild condition: {condition_json}',
            type="rebuild_when",
        )
        condition["initial"] = True
        try:
            result = run_condition(app, condition)
        except UnhandledRebuildCondition:
            env.note_reread()
            raise self.error(f'unhandled match-rebuild condition') from None
        except Exception as e:
            env.note_reread()
            raise self.error(f'unhandled exception: {e!r}') from e

        # Populate dicts only when we're certain they won't be empty. This is
        # why ``env.rebuild_states`` wasn't created if missing earlier in this
        # function.
        if not hasattr(env, "rebuild_states"):
            env.rebuild_states = {}
        if states is None:
            states = env.rebuild_states[docname] = {}
        state = states[condition_json] = {}

        # Store state
        if "state" in result:
            state["state"] = result["state"]

        return []

class RebuildWhenEnvironmentCollector(EnvironmentCollector):
    """Rebuild documents based on `rebuild-when` directives

    State is stored on the environment and persisted between incremental
    builds. It can be accessed in handlers by ``condition["state"]`` and is
    updated by ``result["state"]``.

    """
    @staticmethod
    def get_outdated_docs(app, env, added, changed, removed):
        """Handle the `env-get-outdated` event

        Returns a list of docnames to rebuild based on each document's rebuild
        conditions.

        """
        if not hasattr(env, "rebuild_states"):
            return []

        # Clear conditions for changed and removed docs
        for docname in changed:
            env.rebuild_states.pop(docname, None)
        for docname in removed:
            env.rebuild_states.pop(docname, None)

        # For other docs, run their conditions and rebuild if needed
        docnames_to_rebuild = []
        for docname, states in env.rebuild_states.items():
            rebuild = False
            for condition_json, state in states.items():
                condition = json.loads(condition_json)

                # Restore state
                assert "state" not in condition
                if "state" in state:
                    condition["state"] = state["state"]

                try:
                    result = run_condition(app, condition)
                except UnhandledRebuildCondition:
                    _logger.error(f'unhandled rebuild-when condition')
                    continue
                except Exception as e:
                    _logger.exception(f'unhandled exception: {e!r}')
                    continue

                if result.get("rebuild"):
                    rebuild = True

                # Store state
                state.pop("state", None)
                if "state" in result:
                    state["state"] = result["state"]

            if rebuild:
                docnames_to_rebuild.append(docname)
                # For rebuilt docs, the conditions may still exist, in which
                # case we don't want to run them again. A flag is set which
                # indicates that the condition should be removed later. If the
                # condition appears in the document, the flag is cleared. After
                # parsing, conditions that still have the flag are removed.
                for state in states.values():
                    state["remove"] = True

        return docnames_to_rebuild

    @staticmethod
    def clear_doc(app, env, docname):
        """Handle the `env-purge-doc` event"""
        # For changed and removed docs, all conditions were already removed in
        # ``collector.get_outdated_docs``. For rebuilt docs, the conditions may
        # still exist. We process them in ``collector.process_doc``
        pass

    @staticmethod
    def merge_other(app, env, docnames, other):
        """Handle the `env-merge-info` event"""
        if hasattr(other, "rebuild_states"):
            if not hasattr(env, "rebuild_states"):
                env.rebuild_states = {}
            env.rebuild_states.update(other.rebuild_states)

    @staticmethod
    def process_doc(app, doctree):
        """Handle the `doctree-read` event"""
        env = app.env
        if not hasattr(env, "rebuild_states"):
            return

        # No conditions to remove (new document without rebuild directives)
        docname = env.docname
        if docname not in env.rebuild_states:
            return

        # Remove conditions that were run in ``collector.get_outdated_docs``
        # but don't exist anymore
        states = env.rebuild_states[docname]
        condition_jsons_to_remove = []
        for condition_json, state in states.items():
            if "remove" in state:
                condition_jsons_to_remove.append(condition_json)
        for condition_json in condition_jsons_to_remove:
            del states[condition_json]

        # Remove document key if there are no conditions
        if not len(states):
            del env.rebuild_states[docname]
            if not len(env.rebuild_states):
                del env.rebuild_states

def setup(app):
    """Sphinx extension entry point"""
    app.add_event("match-rebuild")
    app.add_env_collector(RebuildWhenEnvironmentCollector)
    app.add_directive("rebuild-when", RebuildWhenDirective)

    # "Built-in" condition types
    app.connect("match-rebuild", handle_program_change)
    app.connect("match-rebuild", handle_file_change)
    app.connect("match-rebuild", handle_always)

    return {
        # "version": importlib_metadata.version("rebuild_when"),
        "env_version": 1,
        "parallel_read_safe": True,  # Probably parallel-read safe
        "parallel_write_safe": True,
    }
