# Files, libraries, and packages

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [TODO](#todo)
- [Alternatives](#alternatives)
  - [File extensions](#file-extensions)

<!-- tocstop -->

## TODO

This is a skeletal design, added to support [the overview](README.md). Please
feel welcome to rewrite and update as appropriate.

## Alternatives

### File extensions

The use of `6c` as a short file extension or top-level CLI (with subcommands
below it similar to `git` or `go`) has some drawbacks. There are several other
possible extensions / commands:

- `cb`: This collides with several acronyms and may not be especially memorable
  as referring to Carbon.
- `c6`: This seems a weird incorrect ordering of the atomic number and has a bad
  (if _extremely_ obscure) Internet slang association (NSFW, use caution if
  searching, as with too much Internet slang).
- `carbon`: This is an obvious and unsurprising choice, but also quite long.

This seems fairly easy for us to change as we go along, but we should at some
point do a formal proposal to gather other options and let the core team try to
find the set that they feel is close enough to be a bikeshed.
