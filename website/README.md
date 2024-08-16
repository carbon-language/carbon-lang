# Website

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Website generation is done by
[gh_pages_deploy.yaml](/.github/workflows/gh_pages_deploy.yaml). It runs
`prebuild.py`, which prepares files for website generation, then builds the
website using Jekyll, configured through `_config.yml`.

## Development tips

[rbenv](https://github.com/rbenv/rbenv) can be used to set up Ruby and `bundle`.

To run a server, run `bundle exec jekyll serve`. See
[Jekyll docs](https://jekyllrb.com/docs/usage/) for more commands.

To update the `Gemfile.lock` after `Gemfile` changes, run `bundle update`.
