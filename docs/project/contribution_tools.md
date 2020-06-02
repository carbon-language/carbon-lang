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

- [pre-commit](#pre-commit)
- [black](#black)
- [codespell](#codespell)
- [markdown-toc](#markdown-toc)
- [Prettier](#prettier)
  - [vim-prettier](#vim-prettier)

<!-- tocstop -->

## pre-commit

We use [pre-commit](https://pre-commit.com) to run
[various checks](/.pre-commit-config.yaml). This will automatically run
important checks, including formatting.

To set up pre-commit:

- Follow the [installation instructions](https://pre-commit.com/#installation).
- Enable per-repo: `pre-commit install`
  - We already have `pre-commit` configured for Carbon repos -- do not go
    through the `Quick start` instructions.
- pre-commit may be run either automatically with `git commit` or manually with
  `pre-commit run`.
  - When files are modified, including by pre-commit failures, `git add` will
    need to be run to include the modifications in the commit, and the commit
    re-started.

When modifying or adding pre-commit hooks, please run
`pre-commit run --all-files` to see what changes.

## black

> **pre-commit enabled**: If you're using pre-commit, it will run this.
> Installing and running manually is optional, but may be helpful.

We use [Black](https://github.com/psf/black) to format Python code. Although
[Prettier](#prettier) is used for most languages, it doesn't support Python.

## codespell

> **pre-commit enabled**: If you're using pre-commit, it will run this.
> Installing and running manually is optional, but may be helpful.

We use [codespell](https://github.com/codespell-project/codespell) to spellcheck
common errors. This won't catch every error; we're trying to balance true and
false positives.

## markdown-toc

> **pre-commit enabled**: If you're using pre-commit, it will run this.
> Installing and running manually is optional, but may be helpful.

We use [markdown-toc](https://github.com/jonschlinkert/markdown-toc) to provide
GitHub-compatible tables of contents for some documents.

If run manually, specify `--bullets=-` to use Prettier-compatible bullets, or
always run Prettier after markdown-toc.

## Prettier

> **pre-commit enabled**: If you're using pre-commit, it will run this.
> Installing and running manually is optional, but may be helpful.

We use [Prettier](https://prettier.io/) for formatting. There is an
[rc file](/.prettierrc) for configuration.

### vim-prettier

If you use [vim-prettier](https://github.com/prettier/vim-prettier), it may help
to add to your `.vimrc`:

```
let g:prettier#config#print_width = '80'
let g:prettier#config#tab_width = '2'
let g:prettier#config#use_tabs = 'false'
let g:prettier#config#prose_wrap = 'always'
```
