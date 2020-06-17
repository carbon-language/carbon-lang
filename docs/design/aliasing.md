# Aliasing

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [TODO](#todo)
- [Overview](#overview)
  - [Alternatives](#alternatives)

<!-- tocstop -->

## TODO

This is a skeletal design, added to support [the overview](README.md). It should
not be treated as accepted by the core team; rather, it is a placeholder until
we have more time to examine this detail. Please feel welcome to rewrite and
update as appropriate.

## Overview

Naming is one of the things that most often requires careful management over
time -- things tend to get renamed and moved around.

Carbon provides a fully general name aliasing facility to declare a new name as
an alias for a value; everything is a value in Carbon. This is a fully general
facility because everything is a value in Carbon, including types.

For example:

```
alias ??? MyInt = Int;
```

This creates an alias called `MyInt` for whatever `Int` resolves to. Code
textually after this can refer to `MyInt`, and it will transparently refer to
`Int`.

### Alternatives

The syntax here is not at all in a good state yet. We've considered a few
alternatives, but they all end up being confusing in some way. We need to figure
out a good and clean syntax that can be used here.
