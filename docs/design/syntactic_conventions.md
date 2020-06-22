# Syntactic conventions

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
  - [Types before or after name](#types-before-or-after-name)

<!-- tocstop -->

## TODO

This is a skeletal design, added to support [the overview](README.md). It should
not be treated as accepted by the core team; rather, it is a placeholder until
we have more time to examine this detail. Please feel welcome to rewrite and
update as appropriate.

## Overview

Right now we expect variable syntax like: `Int: x`.

There are probably other syntactic conventions that can be added here, too.

## Alternatives

### Types before or after name

While we are currently keeping types first, matching C++, there is significant
uncertainty around the right approach here. While adding the colon improves the
grammar by unambiguously marking the transition from type to a declared
identifier, in essentially every other language with a colon in a similar
position, the identifier is first and the type follows. However, that ordering
would be very _inconsistent_ with C++.

One very important consideration here is the fundamental approach to type
inference. Languages which use the syntax `<identifier>: <type>` typically allow
completely omitting the colon and the type to signify inference. With C++,
inference is achieved with a placeholder keyword `auto`, and Carbon is currently
being consistent there as well with `auto: <identifier>`. For languages which
simply allow omission, this seems an intentional incentive to encourage
inference. On the other hand, there has been strong advocacy in the C++
community to not overly rely on inference and to write the explicit type
whenever convenient. Being consistent with the _ordering_ of identifier and type
may ultimately be less important than being consistent with the incentives and
approach to type inference. What should be the default that we teach? Teaching
to avoid inference unless it specifically helps readability by avoiding a
confusing or unhelpfully complex type name, and incentivizing that by requiring
`auto` or another placeholder, may cause as much or more inconsistency with
languages that use `<identifier>: <type>` as retaining the C++ ordering.

That said, all of this is largely unknown. It will require a significant
exploration of the trade-offs and consistency differences. It should also factor
in further development of pattern matching generally and whether that has an
influence on one or another approach. Last but not least, while this may seem
like something that people will get used to with time, it may be worthwhile to
do some user research to understand the likely reaction distribution, strength
of reaction, and any quantifiable impact these options have on measured
readability. We have only found one _very_ weak source of research that focused
on the _order_ question (rather than type inference vs. explicit types or other
questions in this space). That was a very limited PhD student's study of Java
programmers that seemed to indicate improved latency for recalling the type of a
given variable name with types on the left (as in C++). However, those results
are _far_ from conclusive.

**TODO**: Get a useful link to this PhD research (a few of us got a copy from
the professor directly).
