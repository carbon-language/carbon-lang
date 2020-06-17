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
- [markdown-toc](#markdown-toc)
- [Prettier](#prettier)
  - [vim-prettier](#vim-prettier)

<!-- tocstop -->

## GitHub commandline interface

There is a friendly commandline interface for GitHub called
[`gh`](https://cli.github.com). We use this tool in some of our process
documentation to create pull requests, query their state, etc.

To set up `gh`:

- [Install the tool](https://cli.github.com/manual/installation)
- On first use (such as `gh pr view`), a browser will be launched to authorize
  the tool.
- If running on a machine without a proper web browser, there is a
  [workaround](https://github.com/cli/cli/issues/297#issuecomment-587708363).
  Run the tool locally where a browser is available, and then copy the
  `~/.config/gh/config.yml` file that results to other machines without a
  browser.
  - If even that is not possible, there are some other possible workarounds
    discussed in the main [issue](https://github.com/cli/cli/issues/297) for the
    `gh` tool.

## pre-commit

We use [pre-commit](https://pre-commit.com) to run
[various checks](/.pre-commit-config.yaml). This will automatically run
important checks, including formatting.

To set up pre-commit:

- Follow the [installation instructions](https://pre-commit.com/#installation).
- Enable per-repo: `pre-commit install`

When modifying or adding pre-commit hooks, please run
`pre-commit run --all-files` to see what changes.

## markdown-toc

**pre-commit enabled**: markdown-toc will be run by pre-commit, if installed.

We use [markdown-toc](https://github.com/jonschlinkert/markdown-toc) to provide
GitHub-compatible tables of contents for some documents.

If run manually, specify `--bullets=-` to use Prettier-compatible bullets, or
always run Prettier after markdown-toc.

## Prettier

**pre-commit enabled**: Prettier will be run by pre-commit, if installed.

We use [Prettier](https://prettier.io/) for formatting. There is an
[rc file](/.prettierrc) for configuration.

### vim-prettier

If you use [vim-prettier](https://github.com/prettier/vim-prettier), it may help
to add to your `.virmc`:

```
let g:prettier#config#print_width = '80'
let g:prettier#config#tab_width = '2'
let g:prettier#config#use_tabs = 'false'
let g:prettier#config#prose_wrap = 'always'
```
