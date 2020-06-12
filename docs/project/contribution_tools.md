# Contribution tools

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

The Carbon language project has a number of tools used to assist in preparing
contributions.

## pre-commit

We use [pre-commit](https://pre-commit.com) to run
[various checks](/.pre-commit-config.yaml). This will automatically run
important checks, including formatting.

To set up pre-commit:

- Follow the [installation instructions](https://pre-commit.com/#installation).
- Enable pre-repo: `pre-commit install`

## Prettier

We use [Prettier](https://prettier.io/) for formatting. There is an
[rc file](/.prettierrc) for configuration.

Prettier will be run by pre-commit, if installed.

### vim-prettier

If you use [vim-prettier](https://github.com/prettier/vim-prettier), it may help
to add to your `.virmc`:

```
let g:prettier#config#print_width = '80'
let g:prettier#config#tab_width = '2'
let g:prettier#config#use_tabs = 'false'
let g:prettier#config#prose_wrap = 'always'
```
