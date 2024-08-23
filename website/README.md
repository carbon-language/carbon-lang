# Documentation website

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Build process](#build-process)
-   [Development tips](#development-tips)
-   [Domain configuration](#domain-configuration)

<!-- tocstop -->

## Overview

Carbon's main website is the GitHub project page. Carbon remains too early and
experimental to have a full-fledged website.

This directories contains infrastructure for building the convenience
documentation website, which should be at <https://docs.carbon-lang.dev>.

## Build process

Website generation is done by
[gh_pages_deploy.yaml](/.github/workflows/gh_pages_deploy.yaml). It runs
`prebuild.py`, which prepares files for website generation, then builds the
website using Jekyll, configured through `_config.yml`.

## Development tips

[rbenv](https://github.com/rbenv/rbenv) can be used to set up Ruby and `bundle`.

To run a server, run `bundle exec jekyll serve`. See
[Jekyll docs](https://jekyllrb.com/docs/usage/) for more commands.

To update the `Gemfile.lock` after `Gemfile` changes, run `bundle update`.

## Domain configuration

The
[custom domain](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site)
is configured in three places:

-   GitHub organization
    [verified domains](https://github.com/organizations/carbon-language/settings/pages)
-   GitHub repository
    [custom domain](https://github.com/carbon-language/carbon-lang/settings/pages)
-   Google Cloud DNS
    -   This is visible with `dig docs.carbon-lang.dev`

Note all of these require admin permissions to modify. For sharing test pages, a
GitHub user and repository can be used, pushing to `<username>.github.io` (or
getting a custom DNS setup).
