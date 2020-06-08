# Carbon principle: Metaprogramming

<!--
Part of the Carbon Language, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Principle

Metaprogramming is a technique that can unlock performance, which is
[the top priority for Carbon](https://github.com/jonmeow/carbon-lang/blob/proposal-goals/docs/project/goals.md#performance-critical-software).
C++ provides (at least) two metaprogramming facilities, the preprocessor and
template metaprogramming, and our users will expect to be able to fulfill end
goals in Carbon.

We would like to provide metaprogramming while minimizing downsides that
conflict with Carbon's goals such as code readability and understandability, see
["Carbon Goals"](https://github.com/jonmeow/carbon-lang/blob/proposal-goals/docs/project/goals.md).
The safer we can make metaprogramming in Carbon, with fewer caveats and
pitfalls, the more people will be able to realize the benefits (e.g. better
performance via custom tailored code at compile time).

We expect to do this via a few principles:

- We should aim to minimize the power of the metaprogramming constructs we add
  to just what is needed to solve the anticipated use cases. We prefer a greater
  number of specialized, localized tools over a few all-powerful tools that are
  hard to use.
- Meta-programming code doing ordinary things should look like normal code, and
  not use an entirely separate syntax.
- Constructs that are meta-specific should be distinguished using a common
  syntactical convention that identifies that additional capabilities are being
  exercised. Constructs that do not behave like ordinary code should not look
  ordinary.

### Caveats

- No expectation that you would be able to use metaprogramming to implement an
  embedded DSL that fundamentally differs from Carbon syntax.
- Don't want facilities that introduce non-determinism in builds, such as being
  able to access the clock or non-whitelisted files from code executed at
  compile time.
- Don't want to impede anticipated tooling, such as the ability to scan a file
  for imports quickly and easily, or using brace matching to provide an outline
  view of code in an editor or code viewer.

## Applications of these principles

Minimizing power:

- Metaprogramming constructs should produce fully formed AST subtrees
  representing complete grammatical concepts like an expression, statement, or
  declaration. Metaprogramming facilities should _not_ operate at the textual or
  token level. There is no need to represent incomplete, non-grammatical
  fragments like "7 + ".
- We will not provide a general facility for transforming code. A macro written
  by the user can take either ordinary values (that can be evaluated at compile
  time) or code as a black box (perhaps usable as a closure). If metacode wants
  to use information about a type, it needs to get that information using
  ordinary reflection instead of parsing the type's declaration.

Normal code:

- Should be able to write ordinary imperative code and have it executed at
  compile time, written using normal Carbon syntax. Contrast with both of C++'s
  metaprogramming facilities use substantially different syntax and programming
  models than ordinary C++ code. Also contrast with Lisp-style macros which
  employ lots of quoting and unquoting to do ordinary things like get the value
  of arguments.
- Pure functions (those that don't perform I/O or introduce non-determinism),
  including those defined by the user, should generally be callable both at
  compile time and run time.
- Generally speaking, we will provide APIs for reflection and introspection that
  are uniformly available at both compile time and run time. **Concern:** We
  need a runtime reflection system that does not incur runtime costs (e.g.
  binary size) if it is not used.

Consider an example of how meta code should be distinguished: repeating a block
of code (with variation) should be done with a meta construct ("meta-for"),
rather than trying to use an ordinary `for` loop.

- Repeating a block of code is a meta-operation: it is about algorithmically
  generating code to be consumed by a later stage in the compilation pipeline --
  and as such it should look different.
- The meta-for loop construct should behave according to the rules for
  meta-constructs rather than normal for loops. For example, the body of a
  meta-for loop should be treated as being inlined in the parent scope, rather
  than creating an ordinary scope.
- It is clear that a meta-for loop should be allowed to appear wherever you
  might want to repeat code.

## Proposals relevant to these principles

- ["Carbon metaprogramming" (TODO)](#broken-links-footnote)<!-- T:Carbon metaprogramming -->

## Broken links footnote

Some links in this document aren't yet available, and so have been directed here
until we can do the work to make them available.

We thank you for your patience.
