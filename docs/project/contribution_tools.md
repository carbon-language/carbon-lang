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

-   [Tool setup flow](#tool-setup-flow)
-   [Package managers](#package-managers)
    -   [Linux and MacOS](#linux-and-macos)
        -   [Homebrew](#homebrew)
        -   [`python3` and `pip3`](#python3-and-pip3)
-   [Main tools](#main-tools)
    -   [Bazel and Bazelisk](#bazel-and-bazelisk)
    -   [Clang and LLVM](#clang-and-llvm)
        -   [Manual installations (not recommended)](#manual-installations-not-recommended)
        -   [Troubleshooting build issues](#troubleshooting-build-issues)
        -   [Troubleshooting debug issues](#troubleshooting-debug-issues)
    -   [zlib (Linux-only)](#zlib-linux-only)
    -   [pre-commit](#pre-commit)
-   [Optional tools](#optional-tools)
    -   [Carbon-maintained](#carbon-maintained)
        -   [new_proposal.py](#new_proposalpy)
        -   [pr_comments.py](#pr_commentspy)
    -   [GitHub](#github)
        -   [gh CLI](#gh-cli)
        -   [GitHub Desktop](#github-desktop)
    -   [`rs-git-fsmonitor` and Watchman](#rs-git-fsmonitor-and-watchman)
    -   [Vim](#vim)
        -   [vim-prettier](#vim-prettier)
    -   [Visual Studio Code](#visual-studio-code)
        -   [DevContainer](#devcontainer)
    -   [pre-commit enabled tools](#pre-commit-enabled-tools)
        -   [black](#black)
        -   [codespell](#codespell)
        -   [Prettier](#prettier)

<!-- tocstop -->

## Tool setup flow

In order to set up a machine and git repository for developing on Carbon, a
typical tool setup flow is:

<!-- google-doc-style-ignore -->
<!-- Need to retain "repo" in "gh repo clone". -->

1.  Install [package managers](#package-managers).
2.  Install [main tools](#main-tools) and any desired
    [optional tools](#optional-tools).
3.  Set up the [git](https://git-scm.com/) repository:
    -   `gh repo fork --clone carbon-language/carbon-lang`: this will both
        create a GitHub fork and clone the repository locally
    -   `cd carbon-lang` to go into the cloned fork's directory.
    -   `pre-commit install` to set up [pre-commit](#pre-commit) in the clone.
4.  Validate your installation by invoking `bazel test //...:all' from the
    project root. All tests should pass.

<!-- google-doc-style-resume -->

## Package managers

Instructions for installing tools can be helpful for installing tooling. These
instructions will try to rely on a minimum of managers.

### Linux and MacOS

#### Homebrew

[Homebrew](https://brew.sh/) is a package manager, and can help install several
tools that we recommend.

Our recommended way of installing is to run
[the canonical install command](https://brew.sh/).

To get the latest version of `brew` packages, it will be necessary to
periodically run `brew upgrade`.

#### `python3` and `pip3`

Carbon requires Python 3.9 or newer. The included `pip3` should typically be
used for Python package installation rather than other package managers.

**NOTE**: Carbon will focus support on Homebrew installs of Python 3.9, but it
may not be necessary if you have Python 3.9 installed another way. If you're
trying to use a non-Homebrew Python but have issues involving Carbon and Python,
please try Homebrew's Python.

Our recommended way of installing is:

```bash
brew install python@3.9
pip3 install -U pip
```

**NOTE**: `pip3` runs may print deprecation warnings referencing
https://github.com/Homebrew/homebrew-core/issues/76621. These will need to be
addressed in the future, but as of August 2021 can be ignored.

To get the latest version of `pip3` packages, it will be necessary to
periodically run `pip3 list --outdated`, then `pip3 install -U <package>` to
upgrade desired packages. Keep in mind when upgrading that version dependencies
may mean packages _should_ be outdated, and not be upgraded.

## Main tools

These tools are key for contributions, primarily focused on validating
contributions.

### Bazel and Bazelisk

[Bazel](https://www.bazel.build/) is Carbon's standard build system.
[Bazelisk](https://docs.bazel.build/versions/master/install-bazelisk.html) is
recommended for installing Bazel.

Our recommended way of installing is:

```bash
brew install bazelisk
```

### Clang and LLVM

[Clang](https://clang.llvm.org/) and [LLVM](https://llvm.org/) are used to
compile and link Carbon as part of its build. Bazel will also download and build
against a specific upstream LLVM commit. While the Bazel uses upstream LLVM
sources, the project expects the LLVM 12 release (or newer) to be installed with
Clang and other tools in your `PATH` for use in building Carbon itself.

Our recommended way of installing is:

```bash
brew install llvm
```

On **MacOS only** (not Linux), `llvm` is keg-only; bear in mind this requires
updating `PATH` for it because it's not part of the standard Homebrew path. Read
the output of `brew install` for the necessary path changes, or add something to
your `PATH` like:

```bash
export PATH="$(brew --prefix llvm)/bin:${PATH}"
```

Carbon expects the `PATH` to include the installed tooling. If set, `CC` should
also point at `clang`. Our build environment will detect the `clang` binary
using `CC` then `PATH`, and will expect the rest of the LLVM toolchain to be
available in the same directory as `clang`. However, various scripts and tools
assume that the LLVM toolchain will be in `PATH`, particularly for tools like
`clang-format` and `clang-tidy`.

> TODO: We'd like to use `apt`, but standard LLVM Debian packages are not
> configured correctly for our needs. We are currently aware of two libc++
> issues, [43604](https://bugs.llvm.org/show_bug.cgi?id=43604) and
> [46321](https://bugs.llvm.org/show_bug.cgi?id=46321).

#### Manual installations (not recommended)

You can also build and install `LLVM` yourself if you prefer. The essential
CMake options to pass in order for this to work reliably include:

```
-DLLVM_ENABLE_PROJECTS=clang;clang-tools-extra;lld
-DLLVM_ENABLE_RUNTIMES=compiler-rt;libcxx;libcxxabi;libunwind
-DRUNTIMES_CMAKE_ARGS=-DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF;-DCMAKE_POSITION_INDEPENDENT_CODE=ON;-DLIBCXX_ENABLE_STATIC_ABI_LIBRARY=ON;-DLIBCXX_STATICALLY_LINK_ABI_IN_SHARED_LIBRARY=OFF;-DLIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY=ON;-DLIBCXX_USE_COMPILER_RT=ON;-DLIBCXXABI_USE_COMPILER_RT=ON;-DLIBCXXABI_USE_LLVM_UNWINDER=ON
```

However, we primarily test against the Homebrew installation, so if building
LLVM and Clang yourself you may hit some issues.

#### Troubleshooting build issues

Many build issues result from the particular options `clang` and `llvm` have
been built with, particularly when it comes to system-installed versions. This
is why we recommend using [Homebrew's LLVM](#clang-and-llvm).

After installing from Homebrew, you may need to open a new shell to get `$PATH`
changes. It may also be necessary to run `bazel clean` in order to clean up
cached state.

If issues continue, please ask on
[#build-help](https://discord.com/channels/655572317891461132/824137170032787467),
providing the output of the following diagnostic commands:

```shell
brew --prefix llvm
echo $CC
which clang
grep llvm_bindir $(bazel info workspace)/bazel-execroot/external/bazel_cc_toolchain/clang_detected_variables.bzl
```

These commands will help diagnose potential build issues because they'll expose
what's occurring with
[clang detection](/bazel/cc_toolchains/clang_configuration.bzl).

#### Troubleshooting debug issues

Use the `--compilation_mode=dbg` argument to `bazel build` in order to compile
with debugging enabled. For example:

```shell
bazel build --compilation_mode=dbg //explorer
```

Then debugging works with GDB:

```shell
gdb bazel-bin/explorer/explorer
```

Note that LLVM uses DWARF v5 debug symbols, which means that GDB version 10.1 or
newer is required. If you see an error like this:

```shell
Dwarf Error: DW_FORM_strx1 found in non-DWO CU
```

It means that the version of GDB used is too old, and does not support the DWARF
v5 format.

### zlib (Linux-only)

On **Linux**, you need to have the zlib headers installed. For Debian- and
Ubuntu-based distributions, you can install the development package:

```bash
sudo apt install zlib1g-dev
```

### pre-commit

We use [pre-commit](https://pre-commit.com) to run
[various checks](/.pre-commit-config.yaml). This will automatically run
important checks, including formatting.

Our recommended way of installing is:

```bash
pip3 install pre-commit

# From within each carbon-language git repository:
pre-commit install
```

> NOTE: There are other ways of installing listed at
> [pre-commit.com](https://pre-commit.com/#installation), but `pip` is
> recommended for reliability.

When you have changes to commit to git, a standard pre-commit workflow can look
like:

```bash
# Let pre-commit fix style issues.
pre-commit run
# Add modifications made by pre-commit.
git add .
# Commit the changes
git commit
```

When modifying or adding pre-commit hooks, please run
`pre-commit run --all-files` to see what changes.

## Optional tools

### Carbon-maintained

Carbon-maintained tools are provided by the `carbon-lang` repository, rather
than a separate install. They are noted here mainly to help findability.

#### new_proposal.py

[new_proposal.py](/proposals/scripts/new_proposal.py) is a helper for generating
the PR and proposal file for a new proposal. It's documented in
[the proposal template](/proposals/scripts/template.md).

**NOTE**: This requires installing [the gh CLI](#gh-cli).

#### pr_comments.py

[pr_comments.py](/github_tools/pr_comments.py) is a helper for scanning comments
in GitHub. It's particularly intended to help find threads which need to be
resolved.

Options can be seen with `-h`. A couple key options to be aware of are:

-   `--long`: Prints long output, with the full comment.
-   `--comments-after LOGIN`: Only print threads where the final comment is not
    from the given user. For example, use when looking for threads that you
    still need to respond to.
-   `--comments-from LOGIN`: Only print threads with comments from the given
    user. For example, use when looking for threads that you've commented on.

This script may be run directly if `gql` is installed:

```bash
pip install gql
./github_tools/pr_comments.py <PR#>
```

It may also be run using `bazel`, without installing `gql`:

```bash
bazel run //github_tools:pr_comments -- <PR#>
```

### GitHub

#### gh CLI

[The gh CLI](https://github.com/cli/cli) supports some GitHub queries, and is
used by some scripts.

Our recommended way of installing is:

```bash
brew install gh
```

#### GitHub Desktop

[GitHub Desktop](https://desktop.github.com/) provides a UI for managing git
repositories. See the page for installation instructions.

### `rs-git-fsmonitor` and Watchman

> **WARNING**: Bugs in `rs-git-fsmonitor` and/or Watchman can result in
> `pre-commit` deleting files. If you see files being deleted, disable
> `rs-git-fsmonitor` with `git config --unset core.fsmonitor`.

[rs-git-fsmonitor](https://github.com/jgavris/rs-git-fsmonitor) is a file system
monitor that uses [Watchman](https://github.com/facebook/watchman) to speed up
git on large repositories, such as `carbon-lang`.

Our recommended way of installing is:

-   Linux:

    > If you don't have Rust's [Cargo](https://doc.rust-lang.org/cargo/) package
    > manager, install it first with
    > [the official install command](https://rustup.rs/).

    ```bash
    brew install watchman
    cargo install --git https://github.com/jgavris/rs-git-fsmonitor.git

    # Configure the git repository to use fsmonitor.
    git config core.fsmonitor rs-git-fsmonitor
    ```

-   MacOS:

    ```bash
    brew tap jgavris/rs-git-fsmonitor \
      https://github.com/jgavris/rs-git-fsmonitor.git
    brew install rs-git-fsmonitor

    # Configure the git repository to use fsmonitor.
    git config core.fsmonitor rs-git-fsmonitor
    ```

### Vim

#### vim-prettier

[vim-prettier](https://github.com/prettier/vim-prettier) is a vim integration
for [Prettier](#prettier).

If you use vim-prettier, the `.prettierrc.yaml` should still apply as long as
`config_precedence` is set to the default `file-override`. However, we may need
to add additional settings where the `vim-prettier` default diverges from
`prettier`, as we notice them.

Our recommended way of installing is to use
[the canonical instructions](https://github.com/prettier/vim-prettier#install).

### Visual Studio Code

[Visual Studio Code](https://code.visualstudio.com/) is a code editor used by
several of us. We provide [recommended extensions](/.vscode/extensions.json) to
assist Carbon development. Some settings changes must be made separately:

-   Python › Formatting: Provider: `black`

Our recommended way of installing is to use
[the canonical download](https://code.visualstudio.com/Download).

> **WARNING:** Visual Studio Code modifies the `PATH` environment variable,
> particularly in the terminals it creates. The `PATH` difference can cause
> `bazel` to detect different startup options, discarding its build cache. As a
> consequence, it's recommended to use **either** normal terminals **or** Visual
> Studio Code to run `bazel`, not both in combination. Visual Studio Code can
> still be used for other purposes, such as editing files, without interfering
> with `bazel`.

#### DevContainer

To support developers join the project without deploying the build env, we
provide VSCode `DevContainer`.

-   Install VSCode and Docker;
-   Install the plugin `ms-vscode-remote.remote-containers` under VSCode;
-   Open `Carbon` project folder in
    [VSCode](https://docs.microsoft.com/en-us/azure-sphere/app-development/container-build-vscode#build-and-debug-the-project);
    <br> Visual Studio Code detects the new files and opens a message box
    saying:
    `Folder contains a Dev Container configuration file. Reopen to folder to develop in a container.`
-   Select the `Reopen in Container` button to reopen the folder in the
    container created by the `.devcontainer/Dockerfile` file;
-   And then, you are ready to start writing code.

### pre-commit enabled tools

If you're using pre-commit, it will run these tools. Installing and running them
manually is _entirely optional_, as they can be run without being installed
through `pre-commit run`, but install instructions are still noted here for
direct execution.

#### black

We use [Black](https://github.com/psf/black) to format Python code. Although
[Prettier](#prettier) is used for most languages, it doesn't support Python.

Our recommended way of installing is:

```bash
pip install black
```

#### codespell

We use [codespell](https://github.com/codespell-project/codespell) to spellcheck
common errors. This won't catch every error; we're trying to balance true and
false positives.

Our recommended way of installing is:

```bash
pip install codespell
```

#### Prettier

We use [Prettier](https://prettier.io/) for formatting. There is an
[rc file](/.prettierrc.yaml) for configuration.

Our recommended way of installing is to use
[the canonical instructions](https://prettier.io/docs/en/install.html).
