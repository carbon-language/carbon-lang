# Contribution tools

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

The Carbon language project has a number of tools used to assist in preparing
contributions.

<!-- toc -->

## Table of contents

-   [Setup commands](#setup-commands)
    -   [Debian or Ubuntu](#debian-or-ubuntu)
        -   [Installing Bazelisk](#installing-bazelisk)
        -   [Old `clang` versions](#old-clang-versions)
    -   [macOS](#macos)
-   [Tools](#tools)
    -   [Main tools](#main-tools)
        -   [Running pre-commit](#running-pre-commit)
    -   [Optional tools](#optional-tools)
        -   [Using LLDB with VS Code](#using-lldb-with-vs-code)
        -   [Using GDB with VS Code](#using-gdb-with-vs-code)
    -   [Manually building Clang and LLVM (not recommended)](#manually-building-clang-and-llvm-not-recommended)
-   [Troubleshooting build issues](#troubleshooting-build-issues)
    -   [`bazel clean`](#bazel-clean)
    -   [Old LLVM versions](#old-llvm-versions)
    -   [Asking for help](#asking-for-help)
-   [Troubleshooting debug issues](#troubleshooting-debug-issues)
    -   [Debugging with GDB instead of LLDB](#debugging-with-gdb-instead-of-lldb)
    -   [Debugging other build modes](#debugging-other-build-modes)
    -   [Debugging on MacOS](#debugging-on-macos)

<!-- tocstop -->

## Setup commands

These commands should help set up a development environment on your machine.

<!-- google-doc-style-ignore -->
<!-- Need to retain "repo" in "gh repo clone". -->

### Debian or Ubuntu

```shell
# Update apt.
sudo apt update

# Check that the `clang` version is at least 16, our minimum version. That needs
# the number of the `:` in the output to be over 16. For example, `1:16.0-57`.
apt-cache show clang | grep 'Version:'

# Install tools.
sudo apt install \
  clang \
  gh \
  libc++-dev \
  libc++abi-dev \
  lld \
  lldb \
  python3 \
  pipx

# Install pre-commit.
pipx install pre-commit

# Set up git.
# If you don't already have a fork:
gh repo fork --clone carbon-language/carbon-lang
cd carbon-lang
pre-commit install

# Run tests.
./scripts/run_bazelisk.py test //...:all
```

#### Installing Bazelisk

Although the `run_bazelisk` script can make it easy to get started, if you're
frequently building Carbon, it can be a bit much to type. Consider either
aliasing `bazel` to the `run_bazelisk.py` script, or
[downloading a bazelisk release](https://github.com/bazelbuild/bazelisk), adding
it to your `$PATH`, and aliasing `bazel` to it.

#### Old `clang` versions

If the version of `clang` is earlier than 16, you may still have version 16
available. You can use the following install instead:

```shell
# Install explicitly versioned Clang tools.
sudo apt install \
  clang-16 \
  libc++-16-dev \
  libc++abi-16-dev \
  lld-16 \
  lldb-16

# In your Carbon checkout, tell Bazel where to find `clang`. You can also
# export this path as the `CC` environment variable, or add it directly to
# your `PATH`.
echo "build --repo_env=CC=$(readlink -f $(which clang-16))" >> user.bazelrc
```

> NOTE: Most LLVM 16+ installs should build Carbon. If you're having issues, see
> [troubleshooting build issues](#troubleshooting-build-issues).

### macOS

```shell
# Install Homebrew.
/bin/bash -c "$(curl -fsSL \
  https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# IMPORTANT: Make sure `brew` is added to the PATH!

# Install Homebrew tools.
brew install \
  bazelisk \
  gh \
  llvm \
  python@3.10 \
  pre-commit

# IMPORTANT: Make sure `llvm` is added to the PATH! It's separate from `brew`.

# Set up git.
gh repo fork --clone carbon-language/carbon-lang
cd carbon-lang
pre-commit install

# Run tests. Note homebrew makes `bazel` an alias to `bazelisk`.
bazel test //...:all
```

> NOTE: On macOS, you should end up adding rc file lines similar to:
>
> ```
> # For `brew`, `gh`, and other tools:
> export PATH="${HOME}/.brew/bin:${PATH}"
> # For `llvm`:
> export PATH="$(brew --prefix llvm)/bin:${PATH}"
> ```

<!-- google-doc-style-resume -->

## Tools

### Main tools

These tools are essential for work on Carbon.

-   Package managers
    -   `apt` (for Debian or Ubuntu)
        -   To upgrade versions of `apt` packages, it will be necessary to
            periodically run `sudo apt update && sudo apt upgrade`.
    -   [Homebrew](https://brew.sh/) (for macOS)
        -   To upgrade versions of `brew` packages, it will be necessary to
            periodically run `brew upgrade`.
    -   [Python](https://python.org)
        -   Carbon requires Python 3.9 or newer.
        -   To upgrade versions of pip-installed packages, it will be necessary
            to periodically run `pipx list --outdated`, then
            `pipx install -U <package>` to upgrade desired packages.
        -   When upgrading, version dependencies may mean packages _should_ be
            outdated, and not be upgraded.
-   Main tools
    -   [Bazel](https://www.bazel.build/)
        -   [Bazelisk](https://docs.bazel.build/versions/master/install-bazelisk.html):
            Downloads and runs the [configured Bazel version](/.bazelversion).
    -   [Clang](https://clang.llvm.org/) and [LLVM](https://llvm.org/)
        -   NOTE: Most LLVM 14+ installs should build Carbon. If you're having
            issues, see
            [troubleshooting build issues](#troubleshooting-build-issues).
    -   [gh CLI](https://github.com/cli/cli): Helps with GitHub.
    -   [pre-commit](https://pre-commit.com): Validates and cleans up git
        commits.

#### Running pre-commit

[pre-commit](https://pre-commit.com) is typically set up using
`pre-commit install`. When set up in this mode, it will check for issues when
`git commit` is run. A typical commit workflow looks like:

1.  `git commit` to try committing files. This automatically executes
    `pre-commit run`, which may fail and leave files modified for cleanup.
2.  `git add .` to add the automatically modifications done by `pre-commit`.
3.  `git commit` again.

You can also use `pre-commit run` to check pending changes without `git commit`,
or `pre-commit run -a` to run on all files in the repository.

> NOTE: Some developers prefer to run `pre-commit` on `git push` instead of
> `git commit` because they want to commit files as originally authored instead
> of with pre-commit modifications. To switch, run
> `pre-commit uninstall && pre-commit install -t pre-push`.

### Optional tools

These tools aren't necessary to contribute to Carbon, but can be worth
considering if they fit your workflow.

-   [GitHub Desktop](https://desktop.github.com/): A UI for managing GitHub
    repositories.
-   `rs-git-fsmonitor` and Watchman: Helps make `git` run faster on large
    repositories.
    -   **WARNING**: Bugs in `rs-git-fsmonitor` and/or Watchman can result in
        `pre-commit` deleting files. If you see files being deleted, disable
        `rs-git-fsmonitor` with `git config --unset core.fsmonitor`.
-   [vim-prettier](https://github.com/prettier/vim-prettier): A vim integration
    for [Prettier](https://prettier.io/), which we use for formatting.
-   [Visual Studio Code](https://code.visualstudio.com/): A code editor.
    -   We provide [recommended extensions](/.vscode/extensions.json) to assist
        Carbon development. Some settings changes must be made separately:
        -   Python › Formatting: Provider: `black`
    -   **WARNING:** Visual Studio Code modifies the `PATH` environment
        variable, particularly in the terminals it creates. The `PATH`
        difference can cause `bazel` to detect different startup options,
        discarding its build cache. As a consequence, it's recommended to use
        **either** normal terminals **or** Visual Studio Code to run `bazel`,
        not both in combination. Visual Studio Code can still be used for other
        purposes, such as editing files, without interfering with `bazel`.
    -   [DevContainers](https://code.visualstudio.com/docs/remote/containers): A
        way to use Docker for build environments.
        -   After following the
            [installation instructions](https://code.visualstudio.com/docs/remote/containers#_installation),
            you should be prompted to use Carbon's
            [devcontainer](/.devcontainer/devcontainer.json) with "Reopen in
            container".
-   [clangd](https://clangd.llvm.org/installation): An LSP server implementation
    for C/C++.
    -   To ensure that `clangd` reports accurate diagnostics. It needs a
        generated file called `compile_commands.json`. This can be generated by
        invoking the command below:
        ```
        ./scripts/create_compdb.py
        ```
        -   **NOTE**: This assumes you have `python` 3 installed on your system.

#### Using LLDB with VS Code

The required setup for LLDB is:

1.  Install a minimum of LLVM 19 instead of LLVM 16.
    -   The `lldb-dap` tool was added as part of LLVM 19.
2.  In the `.vscode` subdirectory, symlink `lldb_launch.json` to `launch.json`.
    For example: `ln -s lldb_launch.json .vscode/launch.json`
3.  Install the
    [`llvm-vs-code-extensions.lldb-dap` extension](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.lldb-dap).
4.  In VS Code settings, it may be necessary to set `lldb-dap.executable-path`
    to the path of `lldb-dap`.

A typical debug session looks like:

1. `bazel build -c dbg //toolchain/testing:file_test`
2. Open a `.carbon` testdata file to debug. This must be the active file in VS
   Code.
3. Go to the "Run and debug" panel in VS Code.
4. Select and run the `file_test (lldb)` configuration.

#### Using GDB with VS Code

The required setup for GDB is:

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

### Manually building Clang and LLVM (not recommended)

We primarily test against [apt.llvm.org](https://apt.llvm.org) and Homebrew
installations. However, you can build and install LLVM yourself if you feel more
comfortable with it. The essential CMake options to pass in order for this to
work reliably include:

```
-DLLVM_ENABLE_PROJECTS=clang;clang-tools-extra;lld;lldb
-DLLVM_ENABLE_RUNTIMES=compiler-rt;libcxx;libcxxabi;libunwind
-DRUNTIMES_CMAKE_ARGS=-DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF;-DCMAKE_POSITION_INDEPENDENT_CODE=ON;-DLIBCXX_ENABLE_STATIC_ABI_LIBRARY=ON;-DLIBCXX_STATICALLY_LINK_ABI_IN_SHARED_LIBRARY=OFF;-DLIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY=ON;-DLIBCXX_USE_COMPILER_RT=ON;-DLIBCXXABI_USE_COMPILER_RT=ON;-DLIBCXXABI_USE_LLVM_UNWINDER=ON
-DLLDB_ENABLE_PYTHON=ON
```

## Troubleshooting build issues

### `bazel clean`

Changes to packages installed on your system may not be noticed by `bazel`. This
includes things such as changing LLVM versions, or installing libc++. Running
`bazel clean` should force cached state to be rebuilt.

### Old LLVM versions

Many build issues result from the particular options `clang` and `llvm` have
been built with, particularly when it comes to system-installed versions. If you
run `clang --version`, you should see at least version 16. If you see an older
version, please update, or use the special `clang-16` instructions above.

System installs of macOS typically won't work, for example being an old LLVM
version or missing llvm-ar; [setup commands](#setup-commands) includes LLVM from
Homebrew for this reason.

Run [`bazel clean`](#bazel-clean) when changing the installed LLVM version.

### Asking for help

If you're having trouble resolving issues, please ask on
[#build-help](https://discord.com/channels/655572317891461132/824137170032787467),
providing the output of the following diagnostic commands:

```shell
echo $CC
which clang
which clang-16
clang --version
grep llvm_bindir $(bazel info workspace)/bazel-execroot/external/+clang_toolchain_extension+bazel_cc_toolchain/clang_detected_variables.bzl

# If on macOS:
brew --prefix llvm
```

These commands will help diagnose potential build issues by showing which
tooling is in use.

## Troubleshooting debug issues

Pass `-c dbg` to `bazel build` in order to compile with debugging enabled. For
example:

```shell
bazel build -c dbg //toolchain
```

Then debugging works with LLDB:

```shell
lldb bazel-bin/toolchain/carbon
```

Any installed version of LLDB at least as recent as the installed Clang used for
building should work.

### Debugging with GDB instead of LLDB

If you prefer using GDB, you may want to pass some extra flags to the build:

```shell
bazel build -c dbg --features=-lldb_flags --features=gdb_flags //toolchain
```

Or you can add them to your `user.bazelrc`, they are designed to be safe to pass
at all times and only have effect when building with debug information:

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

### Debugging other build modes

If you have an issue that only reproduces with another build mode, you can still
enable debug information in that mode by passing `--feature=debug_info_flags` to
Bazel.

### Debugging on MacOS

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
