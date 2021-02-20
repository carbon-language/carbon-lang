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

-   [Contribution Setup Flow](#contribution-setup-flow)
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
    -   [gql](#gql)
    -   [PyGitHub](#pygithub)
-   [Setup Validation](#setup-validation)
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
    -   [Atom](#atom)
    -   [pre-commit enabled tools](#pre-commit-enabled-tools)
        -   [black](#black)
        -   [codespell](#codespell)
        -   [markdown-toc](#markdown-toc)
        -   [Prettier](#prettier)

<!-- tocstop -->

## Contribution Setup Flow

In order to set up a machine and git repository for developing on Carbon, a
typical setup flow for contribution is:

1.  Set up the [git](https://git-scm.com/) repository:
    -   In GitHub, create a fork for development at
        https://github.com/carbon-language/carbon-lang.
    -   `gh repository clone USER/carbon-lang`, or otherwise clone the fork.
    -   `cd carbon-lang` to go into the cloned fork's directory.
    -   `git config core.fsmonitor rs-git-fsmonitor` to set up
        [rs-git-fsmonitor](#rs-git-fsmonitor-and-watchman) in the clone.
    -   `git submodule update --init` to sync submodules if you'll be building
        c++ code or working on the compiler. **Note:** this step is slow and you
        can proceed with steps 2-4 while it completes.
2.  Install [package managers](#package-managers).
3.  Install [main tools](#main-tools)
4.  `pre-commit install` in the cloned fork's directory to set up
    [pre-commit](#pre-commit).
5.  If you you updated submodules in step 1, and the command is complete, you
    can validate your required tools by invoking `bazel test //...:all' from the
    project root. All tests should pass. **Note:** the first time around, this
    step is very slow (10 minutes on the fastest workstations). You can proceed
    with step 6 while it completes.
6.  Install any desired [optional tools](#optional-tools).

## Package managers

Instructions for installing tools can be helpful for installing tooling. These
instructions will try to rely on a minimum of managers.

### Linux and MacOS

#### Homebrew

[Homebrew](https://brew.sh/) is a package manager, and can help install several
tools that we recommend. See the [installation instructions](https://brew.sh/).

To get the latest version of `brew` packages, it will be necessary to
periodically run `brew upgrade`.

#### Python using `pyenv`

We strongly recommend using [pyenv](https://github.com/pyenv/pyenv) to manage
[Python](python.org) and Python's `pip` package manager. `pip` should typically
be used for Python package installation rather than other package managers.

These can be installed together through `brew`:

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
install a couple tools on Linux. See the
[installation instructions](https://rustup.rs/).

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

To set up pre-commit, see the
[installation instructions](https://pre-commit.com/#installation), or:

```bash
pip install pre-commit

# From within each carbon-language git repository:
pre-commit install
```

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

### gql

```bash
pip install gql
```

### PyGitHub

```bash
pip install PyGitHub
```

## Setup Validation

If you've installed all the foregoing tools in this document. You can validate
that you've done everything right by invoking `bazelisk test //...:all' from the
project root. All tests should pass.

## Optional tools

### Carbon-maintained

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

### GitHub

#### gh CLI

[The gh CLI](https://github.com/cli/cli) supports some GitHub queries, and is
used by some scripts.

To install gh, run:

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

### Atom

[Atom](https://atom.io/) is an IDE that's mainly mentioned here for its Markdown
support. See the page for installation instructions.

Some packages that may be helpful are:

-   [markdown-preview-enhanced](https://atom.io/packages/markdown-preview-enhanced):
    An improvement over the default-installed `markdown-preview` package. If
    using this:
    -   Disable `markdown-preview`.
    -   In the package settings, turn off 'Break On Single New Line' -- this is
        consistent with [carbon-lang.dev](https://carbon-lang.dev) and
        [Prettier](#prettier).
-   [prettier-atom](https://atom.io/packages/prettier-atom): For
    [Prettier](#prettier) integration.
-   [python-black](https://atom.io/packages/python-black): For [black](#black)
    integration.

### pre-commit enabled tools

If you're using pre-commit, it will run these tools. Installing and running them
manually is optional, but may be helpful.

#### black

We use [Black](https://github.com/psf/black) to format Python code. Although
[Prettier](#prettier) is used for most languages, it doesn't support Python.

#### codespell

We use [codespell](https://github.com/codespell-project/codespell) to spellcheck
common errors. This won't catch every error; we're trying to balance true and
false positives.

#### markdown-toc

We use [markdown-toc](https://github.com/jonschlinkert/markdown-toc) to provide
GitHub-compatible tables of contents for some documents.

If run manually, specify `--bullets=-` to use Prettier-compatible bullets, or
always run Prettier after markdown-toc.

#### Prettier

We use [Prettier](https://prettier.io/) for formatting. There is an
[rc file](/.prettierrc.yaml) for configuration.
