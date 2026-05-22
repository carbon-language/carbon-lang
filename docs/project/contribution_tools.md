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
        -   [Jujutsu (`jj`)](#jujutsu-jj)
        -   [AI assistants](#ai-assistants)
    -   [Running tests with AddressSanitizer (ASan)](#running-tests-with-addresssanitizer-asan)
    -   [Manually building Clang and LLVM (not recommended)](#manually-building-clang-and-llvm-not-recommended)
-   [Troubleshooting build issues](#troubleshooting-build-issues)
    -   [`bazel clean`](#bazel-clean)
    -   [Old LLVM versions](#old-llvm-versions)
    -   [Debugging](#debugging)
    -   [Asking for help](#asking-for-help)

<!-- tocstop -->

## Setup commands

These commands should help set up a development environment on your machine.

<!-- google-doc-style-ignore -->
<!-- Need to retain "repo" in "gh repo clone". -->

### Debian or Ubuntu

```shell
# Update apt.
sudo apt update

# Check that the `clang` version is at least 19, our minimum version. That needs
# the number of the `:` in the output to be over 19. For example, `1:19.0-1`.
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

If the version of `clang` is earlier than 19, you may still have version 19
available. You can use the following install instead:

```shell
# Install explicitly versioned Clang tools.
sudo apt install \
  clang-19 \
  libc++-19-dev \
  libc++abi-19-dev \
  lld-19 \
  lldb-19

# In your Carbon checkout, tell Bazel where to find `clang`. You can also
# export this path as the `CC` environment variable, or add it directly to
# your `PATH`.
echo "build --repo_env=CC=$(readlink -f $(which clang-19))" >> user.bazelrc
```

And if it's not available directly from the distribution, you can install Clang
tools on Debian/Ubuntu from <https://apt.llvm.org>.

> NOTE: Most LLVM 19+ installs should build Carbon. If you're having issues, see
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
        -   Carbon requires Python 3.10 or newer.
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
        -   NOTE: Most LLVM 19+ installs should build Carbon. If you're having
            issues, see
            [troubleshooting build issues](#troubleshooting-build-issues).
    -   [gh CLI](https://github.com/cli/cli): Helps with GitHub.
    -   [pre-commit](https://pre-commit.com): Validates and cleans up git
        commits.
    -   `autoupdate_testdata.py`: Updates expected output for tests.
        -   Usage: `./toolchain/autoupdate_testdata.py [files...]`
        -   This is essential when changes affect compiler output (diagnostics,
            SemIR, etc.).

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
    -   We also provide recommended setups for debugging in VS Code with either
        [LLDB](/toolchain/docs/debugging.md#debugging-with-lldb) or
        [GDB]((/toolchain/docs/debugging.md#debugging-with-gdb)
-   [clangd](https://clangd.llvm.org/installation): An LSP server implementation
    for C/C++.

    -   To ensure that `clangd` reports accurate diagnostics. It needs a
        generated file called `compile_commands.json`. This can be generated by
        invoking the command below:

        ```
        ./scripts/create_compdb.py
        ```

        -   **NOTE**: This assumes you have `python` 3 installed on your system.

-   [`uv`](https://docs.astral.sh/uv/): A fast Python package manager.
    -   Notably, `uv` supports automatic management of even complex Python
        dependencies for scripts: https://docs.astral.sh/uv/guides/scripts/
    -   Installation: https://docs.astral.sh/uv/getting-started/installation/

#### Jujutsu (`jj`)

[Jujutsu](https://github.com/jj-vcs/jj) is a Git-compatible version control
system that can be used instead of or alongside Git. See the
[documentation for using Jujutsu with GitHub](https://jj-vcs.github.io/jj/latest/github/)
for more information.

If you use `jj`, you may find the following configuration snippets (added to
`jj config path --user`) helpful for your workflow:

```sh
# Clean up untracked or abandoned commits.
jj config set --user aliases.abandon-untagged '["abandon", "~ancestors(working_copies() | bookmarks() | remote_bookmarks())"]'

# Use Git-style conflict markers, which VS Code can provide merge support for.
jj config set --user ui.conflict-marker-style 'git'

# Produce Git-compatible diff format.
jj config set --user ui.diff.format 'git'

# Automatically add a trailer to commits to indicate that they were AI-assisted.
jj config set --user templates.commit_trailers "$(echo -e "'''\n\"Assisted-by: My AI Tool\"'''")"

# Make `jj bookmark advance` / `jj b a` only move bookmarks that point to
# mutable commits, and move them to the most recent non-empty descendant.
jj config set --user revsets.bookmark-advance-from 'heads(::to & bookmarks()) & ~immutable_heads()'
jj config set --user revsets.bookmark-advance-to 'heads(::@ & ~(description("") & empty() & ~merges()))'
```

<!-- google-doc-style-ignore -->

As well as this per-repository configuration (added to `jj config path --repo`)
describing how your GitHub checkout is configured:

```sh
# Automatically track all remote bookmarks.
jj config set --repo remotes.origin.auto-track-bookmarks '*'

# `trunk()` is a jj builtin, but defaults to `main@upstream`.
jj config set --repo 'revset-aliases."trunk()"' 'trunk@upstream'

# Treat github.com/carbon-language/carbon-lang as immutable, but treat your fork
# as mutable.
jj config set --repo 'revset-aliases."immutable_heads()"' 'remote_bookmarks(*, upstream)'
```

<!-- google-doc-style-resume -->

The above assumes that you have configured the remote name `origin` to refer to
your fork and `upstream` to refer to `github.com/carbon-language/carbon-lang`,
and will need to be adjusted if you use different remote names.

#### AI assistants

When using AI assistants and reviewing terminal commands, some commands which
may be helpful and reasonably safe to allowlist (assuming prefix-based
allowlisting) are:

```
# Carbon development commands.
bazelisk build
bazelisk test
bazelisk run //toolchain/testing:file_test --
clang-format
pre-commit run
./toolchain/autoupdate_testdata.py

# Shell commands. Note that these allow reading arbitrary files on your local
# file system.
cat
grep
head
ls

# VCS commands.
git diff
git log
git show
git status
```

### Running tests with AddressSanitizer (ASan)

By default, the Bazel build mode for the toolchain does not enable
AddressSanitizer (ASan). If you wish to enable ASan for local testing, you must
pass the `--config=asan` flag explicitly:

```shell
bazelisk test --config=asan //...
```

Note that our Continuous Integration (CI) infrastructure runs a separate
configuration for ASan to ensure test coverage without slowing down the default
test cycle.

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
run `clang --version`, you should see at least version 19. If you see an older
version, please update, or use the special `clang-19` instructions above.

System installs of macOS typically won't work, for example being an old LLVM
version or missing llvm-ar; [setup commands](#setup-commands) includes LLVM from
Homebrew for this reason.

Run [`bazel clean`](#bazel-clean) when changing the installed LLVM version.

### Debugging

See the [toolchain documentation](/toolchain/docs/debugging.md) for guidance on
how to debug problems with the toolchain itself.

### Asking for help

If you're having trouble resolving issues, please ask on
[#build-help](https://discord.com/channels/655572317891461132/824137170032787467),
providing the output of the following diagnostic commands:

```shell
echo $CC
which clang
which clang-19
clang --version
grep llvm_bindir $(bazel info workspace)/bazel-execroot/external/+clang_toolchain_extension+bazel_cc_toolchain/clang_detected_variables.bzl

# If on macOS:
brew --prefix llvm
```

These commands will help diagnose potential build issues by showing which
tooling is in use.
