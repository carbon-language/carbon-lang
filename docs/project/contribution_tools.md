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
        -   [Python using `pyenv`](#python-using-pyenv)
    -   [Linux only](#linux-only)
        -   [`go get`](#go-get)
        -   [Cargo (optional)](#cargo-optional)
-   [Main tools](#main-tools)
    -   [Bazel and Bazelisk](#bazel-and-bazelisk)
    -   [buildifier](#buildifier)
    -   [Clang and LLVM](#clang-and-llvm)
    -   [Ninja](#ninja)
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
    -   In GitHub, create a fork for development at
        https://github.com/carbon-language/carbon-lang.
    -   `gh repo clone USER/carbon-lang`, or otherwise clone the fork.
    -   `cd carbon-lang` to go into the cloned fork's directory.
    -   `git submodule update --init --depth=1` to sync submodules if you'll be
        building c++ code or working on the compiler.
    -   `git config core.fsmonitor rs-git-fsmonitor` to set up
        [rs-git-fsmonitor](#rs-git-fsmonitor-and-watchman) in the clone.
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
tools that we recommend. See the

Our recommended way of installing is to run
[the canonical install command](https://brew.sh/).

To get the latest version of `brew` packages, it will be necessary to
periodically run `brew upgrade`.

#### Python using `pyenv`

Carbon requires Python 3.6 or newer. Everything below assumes that `python` or
`pip` reach the Python 3 tools, not legacy installations of Python 2.

We strongly recommend using [pyenv](https://github.com/pyenv/pyenv) to manage
[Python](python.org) and Python's `pip` package manager. `pip` should typically
be used for Python package installation rather than other package managers.

Our recommended way of installing is:

```bash
brew install pyenv
pyenv install 3.8.5
pyenv global 3.8.5

# Add 'eval "$(pyenv init -)"' to your shell rc file, for example zshrc.
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
# Load the shell rc file changes.
exec $SHELL
```

To get the latest version of `pip` packages, it will be necessary to
periodically run `pip list --outdated`, then `pip install -U <package>` to
upgrade desired packages. Keep in mind when upgrading that version dependencies
may mean packages _should_ be outdated, and not be upgraded.

### Linux only

Linux-specific package managers are typically used for packages which work
through [brew](#brew) on MacOS, but not on Linux.

Installation instructions assume Debian- or Ubuntu-based Linux distributions
with [apt](<https://en.wikipedia.org/wiki/APT_(software)>) available.

#### `go get`

[go get](https://golang.org/pkg/cmd/go/internal/get/) is Go's package manager.

Our recommended way of installing is:

```bash
apt install golang
```

To get the latest version of `go` packages, it will be necessary to periodically
re-run the original `go get ...` command used to install the package.

#### Cargo (optional)

Rust's [Cargo](https://doc.rust-lang.org/cargo/) package manager is used to
install a couple tools on Linux.

Our recommended way of installing is to run
[the canonical install command](https://rustup.rs/).

To get the latest version of `cargo` packages, it will be necessary to
periodically re-run the original `cargo install ...` command used.

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

### buildifier

[Buildifier](https://github.com/bazelbuild/buildtools/tree/master/buildifier) is
a tool for formatting Bazel BUILD files, and is distributing separately from
Bazel.

Our recommended way of installing is:

-   Linux:

    ```bash
    go get github.com/bazelbuild/buildtools/buildifier
    ```

-   MacOS:

    ```bash
    brew install buildifier
    ```

### Clang and LLVM

[Clang](https://clang.llvm.org/) and [LLVM](https://llvm.org/) are used to
compile and link Carbon, and are provided through git submodules. A complete
toolchain will be built and cached as part of standard `bazel` execution. This
can be very slow on less powerful computers or laptops (30 minutes to an hour).
However, it should only happen when either Bazel or LLVM's submodule is updated,
which we try to minimize. If you need to force a rebuild of the toolchain, you
can use `bazel sync --configure`.

### Ninja

[Ninja](https://ninja-build.org/) is used to build Clang and LLVM.

Our recommended way of installing is:

```bash
brew install ninja
```

### pre-commit

We use [pre-commit](https://pre-commit.com) to run
[various checks](/.pre-commit-config.yaml). This will automatically run
important checks, including formatting.

Our recommended way of installing is:

```bash
pip install pre-commit

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
[the proposal template](/proposals/template.md).

**NOTE**: This requires installing [the gh CLI](#gh).

#### pr_comments.py

[pr_comments.py](/github/pr_comments.py) is a helper for scanning comments in
GitHub. It's particularly intended to help find threads which need to be
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
brew install github/gh/gh
```

#### GitHub Desktop

[GitHub Desktop](https://desktop.github.com/) provides a UI for managing git
repositories. See the page for installation instructions.

### `rs-git-fsmonitor` and Watchman

[rs-git-fsmonitor](https://github.com/jgavris/rs-git-fsmonitor) is a file system
monitor that uses [Watchman](https://github.com/facebook/watchman) to speed up
git on large repositories, such as `carbon-lang` when submodules are synced.

Our recommended way of installing is:

-   Linux:

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

[Visual Studio Code](https://code.visualstudio.com/) is an IDE used by several
of us. We provide [recommended extensions](/.vscode/extensions.json) to assist
Carbon development. Some settings changes must be made separately:

-   Python › Formatting: Provider: `black`

Our recommended way of installing is to use
[the canonical download](https://code.visualstudio.com/Download).

> **WARNING:** Visual Studio Code's Remote Development mode modifies the `PATH`
> environment variable, particularly in the terminals it creates. The `PATH`
> difference can cause `bazel` to detect different startup options, discarding
> its build cache. In order to avoid this issue, you can add to your rc file:
>
> ```bash
> export PATH=$(echo "${PATH}" | sed -e 's/:[^:]*\/.vscode-server\/[^:]*:/:/g')
> ```

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
