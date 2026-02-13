# Debugging tools and techniques

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Interactive debugger setup](#interactive-debugger-setup)
    -   [Debugging with LLDB](#debugging-with-lldb)
        -   [MacOS](#macos)
        -   [VS Code integration](#vs-code-integration)
    -   [Debugging with GDB](#debugging-with-gdb)
        -   [VS Code integration](#vs-code-integration-1)
-   [Toolchain-specific debugging techniques](#toolchain-specific-debugging-techniques)
    -   [Dumping objects in interactive debuggers](#dumping-objects-in-interactive-debuggers)
        -   [The LLDB `dump` command](#the-lldb-dump-command)
    -   [Verbose output](#verbose-output)
    -   [Stack traces](#stack-traces)
    -   [Dumping prelude files](#dumping-prelude-files)

<!-- tocstop -->

## Interactive debugger setup

The toolchain codebase should work with either GDB or LLDB, but using GDB
requires a little more setup and may not have quite as many conveniences.

Pass `-c dbg` to `bazel build` in order to compile with debugging enabled. For
example:

```shell
bazel build -c dbg //toolchain
```

This produces a compiled binary with debug symbols at
`bazel-bin/toolchain/carbon`. For the best debugging experience you may want to
pass additional flags when building and/or running the debugger, but the
specifics depend on your environment; see the following sections for details.

If you have an issue that only reproduces with another build mode, you can still
enable debug information in that mode by passing `--feature=debug_info_flags` to
Bazel.

### Debugging with LLDB

The output of a `-c dbg` build should be usable with any version of LLDB as
recent as the installed Clang used for building:

```shell
lldb bazel-bin/toolchain/carbon
```

However, we recommend running LLDB with the `--local-lldbinit` flag, to enable
our preset configuration (which enables the
[`dump` command](#the-lldb-dump-command) among other features). This requires
running from the repository root:

```shell
lldb --local-lldbinit bazel-bin/toolchain/carbon
```

To debug a single `file_test`, use the following command, pointing it to an
actual carbon test file.

```shell
bazel build -c dbg //toolchain/testing:file_test && \
  lldb --local-lldbinit bazel-bin/toolchain/testing/file_test -- \
    --dump_output --file_tests /path/to/some/test.carbon
```

#### MacOS

Bazel sandboxes builds, which on MacOS makes it hard for the debugger to locate
symbols on linked binaries when debugging. See this
[Bazel issue](https://github.com/bazelbuild/bazel/issues/2537#issuecomment-449089673)
for more information. To workaround, provide the `--spawn_strategy=local` option
to Bazel for the debug build, like:

```shell
bazel build --spawn_strategy=local -c dbg //toolchain
```

You should then be able to debug with `lldb`.

If this build command doesn't seem to produce a debuggable binary you might need
to both clear the build disk cache and clean the build. Running
`scripts/clean_disk_cache.sh` may not be enough, you might try deleting all the
files within the disk cache, typically located at
`~/.cache/carbon-lang-build-cache`. Deleting the disk cache, followed by a
`bazel clean` should allow your next rebuild, with the recommended options, to
supply the symbols for debugging.

#### VS Code integration

VS Code's integrated debugger support can be configured to use LLDB, with custom
support for debugging the toolchain tests. To set that up:

1.  In the `.vscode` subdirectory, symlink `lldb_launch.json` to `launch.json`.
    For example: `ln -s lldb_launch.json .vscode/launch.json`
2.  Install the
    [`llvm-vs-code-extensions.lldb-dap` extension](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.lldb-dap).
3.  In VS Code settings, it may be necessary to set `lldb-dap.executable-path`
    to the path of `lldb-dap`.

A typical debug session looks like:

1. `bazel build -c dbg //toolchain/testing:file_test`
2. Open a `.carbon` testdata file to debug. This must be the active file in VS
   Code.
3. Go to the "Run and debug" panel in VS Code.
4. Select and run the `file_test (lldb)` configuration.

For debugging on MacOS using VSCode, some people have had success using the
CodeLLDB extension. In order for LLDB to connect the project source files with
the symbols you will need to add a `"sourceMap": { ".": "${workspaceRoot}" }`
line to the CodeLLDB `launch.json` configuration, for example:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "explorer",
            "type": "lldb",
            "request": "launch",
            "program": "${workspaceRoot}/bazel-bin/explorer/explorer",
            "args": [],
            "cwd": "${workspaceRoot}",
            "sourceMap": {
                ".": "${workspaceRoot}"
            }
        }
    ]
}
```

### Debugging with GDB

If you prefer using GDB, you may want to pass some extra flags to the build:

```shell
bazel build -c dbg --features=-lldb_flags --features=gdb_flags //toolchain
```

Or you can add them to your `user.bazelrc`, they are designed to be safe to pass
at all times and only have effect when building with `-c dbg`:

```shell
echo "build --features=-lldb_flags --features=gdb_flags" >> user.bazelrc
```

Note that on Linux we use Split DWARF and DWARF v5 debug symbols, which means
that GDB version 10.1 or newer is required. If you see an error like this:

```shell
Dwarf Error: DW_FORM_strx1 found in non-DWO CU
```

It means that the version of GDB used is too old, and does not support the DWARF
v5 format.

#### VS Code integration

VS Code's integrated debugger support can be configured to use GDB, with custom
support for debugging the toolchain tests. To set that up:

1.  In the `.vscode` subdirectory, symlink `gdb_launch.json` to `launch.json`.
    For example: `ln -s gdb_launch.json .vscode/launch.json`
2.  Install the
    [`coolchyni.beyond-debug` extension](https://marketplace.visualstudio.com/items?itemName=coolchyni.beyond-debug).

A typical debug session looks like:

1. `bazel build -c dbg --features=-lldb_flags --features=gdb_flags //toolchain/testing:file_test`
2. Open a `.carbon` testdata file to debug. This must be the active file in VS
   Code.
3. Go to the "Run and debug" panel in VS Code.
4. Select and run the `file_test (gdb)` configuration.

## Toolchain-specific debugging techniques

### Dumping objects in interactive debuggers

We provide namespace-scoped `Dump` functions in several components, such as
[check/dump.cpp](/toolchain/check/dump.cpp). These `Dump` functions will print
contextual information about an object to stderr. The files contain details
regarding support.

Objects which inherit from `Printable` also have `Dump` member functions, but
these will lack contextual information.

IDs are dumped in hexadecimal, so it's often convenient to set your interactive
debugger to print integers in hexadecimal as well. In LLDB you can do this with
`type format add --format hex int`.

#### The LLDB `dump` command

Our LLDB configuration includes a `dump` command (see
[above](#debugging-with-lldb) for how to enable that configuration), which
allows you to dump the contents of a value associated with an id. Since most
data in the toolchain is referenced by id, this ends up being a very frequent
task.

The debugger command `dump <context> <id_expr>` gets roughly translated into a
C++ call to `Dump(<context>, <id_expr>)`.

Run the `dump` command without any arguments to see the builtin help on how to
use it.

### Verbose output

The `-v` flag can be passed to trace state, and should be specified before the
subcommand name: `carbon -v compile ...`. `CARBON_VLOG` is used to print output
in this mode. There is currently no control over the degree of verbosity.

To include VLOG output when debugging a file test, add an `ARGS: -v compile %s`
line to the file, such as:

```
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// ARGS: -v compile %s
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
```

This will also include the VLOG output when running the test in an interactive
debugger. Note that using `-v compile` with `autoupdate.py` will deeply mangle
your test file, so avoid doing that.

### Stack traces

While the iterative processing pattern means function stack traces will have
minimal context for how the current function is reached, we use LLVM's
`PrettyStackTrace` to include details about the state stack. The state stack
will be above the function stack in crash output.

You can also use the `--sem-ir-crash-dump=path/to/file` flag to get a raw SemIR
dump in the event of a crash in the check phase. This can be particularly useful
for interpreting IDs you encounter during interactive debugging.

### Dumping prelude files

By default, prelude files are excluded from SemIR dumps by
`--exclude-dump-file-prefix`. To enable dumps for specific files, add
`//@include-in-dumps`. This works for every phase after lex, but may be most
helpful to debug check and lower output. This can also be used to view
cross-file SemIR, such as imports from a prelude, by adding
`//@include-in-dumps` to the prelude file and looking at the SemIR of the
importing file.
