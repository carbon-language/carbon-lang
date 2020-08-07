# Contribution tools

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

The Carbon language project has a number of tools used to assist in preparing
contributions.

## Table of contents

<!-- toc -->

-   [Main tools](#main-tools)
    -   [brew](#brew)
    -   [pyenv and Python](#pyenv-and-python)
    -   [pre-commit](#pre-commit)
-   [Optional tools](#optional-tools)
    -   [Carbon-maintained](#carbon-maintained)
        -   [new_proposal.py](#new_proposalpy)
        -   [pr_comments.py](#pr_commentspy)
    -   [GitHub](#github)
        -   [gh CLI](#gh-cli)
        -   [GitHub Desktop](#github-desktop)
    -   [Vim](#vim)
        -   [vim-prettier](#vim-prettier)
    -   [Atom](#atom)
    -   [pre-commit enabled tools](#pre-commit-enabled-tools)
        -   [black](#black)
        -   [codespell](#codespell)
        -   [markdown-toc](#markdown-toc)
        -   [Prettier](#prettier)

<!-- tocstop -->

## Main tools

These tools are key for contributions, primarily focused on validating
contributions.

### brew

[brew](https://brew.sh/) is a package manager, and can help install several
tools that we recommend. See the [installation instructions](https://brew.sh/).

### pyenv and Python

[pyenv](https://github.com/pyenv/pyenv) is the recommended way to install
[Python](python.org). Our recommended way of installing both is:

```bash
brew update
brew install pyenv
pyenv install python3.8.5
pyenv global python3.8.5

# Add 'eval "$(pyenv init -)"' to your shell rc file.
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
# Load the shell rc file changes.
exec $SHELL
```

### pre-commit

We use [pre-commit](https://pre-commit.com) to run
[various checks](/.pre-commit-config.yaml). This will automatically run
important checks, including formatting.

To set up pre-commit, see the
[installation instructions](https://pre-commit.com/#installation), or:

```bash
pip install pre-commit

# From within each carbon-language git repo:
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

## Optional tools

### Carbon-maintained

#### new_proposal.py

[new_proposal.py](/src/scripts/new_proposal.py) is a helper for generating the
PR and proposal file for a new proposal. It's documented in
[the proposal template](/proposals/template.md).

**NOTE**: This requires installing [the gh CLI](#gh).

#### pr_comments.py

[pr_comments.py](/src/scripts/pr_comments.py) is a helper for scanning comments
in GitHub. It's particularly intended to help find threads which need to be
resolved.

Flags can be seen with `-h`. A couple key flags to be aware of are:

-   `--long`: Prints long output, with the full comment.
-   `--comments-after LOGIN`: Only print threads where the final comment is not
    from the given user. For example, use when looking for threads that you
    still need to respond to.
-   `--comments-from LOGIN`: Only print threads with comments from the given
    user. For example, use when looking for threads that you've commented on.

**NOTE**: This requires the Python gql package:

```bash
pip install gql
```

### GitHub

#### gh CLI

[The gh CLI](https://github.com/cli/cli) supports some GitHub queries, and is
used by some scripts.

To install gh, run:

```bash
brew update
brew install github/gh/gh
```

#### GitHub Desktop

[GitHub Desktop](https://desktop.github.com/) provides a UI for managing git
repos. See the page for installation instructions.

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
