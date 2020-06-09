# Carbon principle: Refactoring

<!--
Part of the Carbon Language, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Principle

Carbon should facilitate code refactoring by making common transformations safe
and easy. This is in service to
["Both software and language evolution"](https://github.com/jonmeow/carbon-lang/blob/proposal-goals/docs/project/goals.md#both-software-and-language-evolution)
from
["Carbon Goals"](https://github.com/jonmeow/carbon-lang/blob/proposal-goals/docs/project/goals.md).

- Code should be movable, e.g., to another file, via a simple and easy procedure
  without danger of silently changing semantics.
- We should strive to have a uniform syntax that enables common refactorings
  without extensive rewriting.
- Carbon should make it easy to automate refactoring, with standardized tooling.
- Carbon should allow refactorings to be made incrementally if they could
  require global (cross-package) changes.

### Caveats

- Rewrites needed for refactoring do not necessarily need to be specified in
  Carbon source code. Some refactoring can be, e.g., an IDE feature, as long as
  we strive to make it straightforward to implement.

## Applications of this principle

Background: catalogs of common refactoring operations:

- [https://refactoring.com/catalog/](https://refactoring.com/catalog/)
- [https://sourcemaking.com/refactoring/refactorings](https://sourcemaking.com/refactoring/refactorings)
- [https://medium.com/@aserg.ufmg/what-are-the-most-common-refactorings-performed-by-github-developers-896b0db96d9d](https://medium.com/@aserg.ufmg/what-are-the-most-common-refactorings-performed-by-github-developers-896b0db96d9d)

Moving code:

- As much as possible, dependencies on the context for defining the semantics of
  code in a function should be explicit rather than implicit.
- Code should depend only minimally on its context. Ideally all you would only
  have to import any referenced names in order to move code within a library.
  Moving code to another library would involve also updating callers or leaving
  an alias (ideally marked "deprecated") behind.

Uniform syntax:

- Rewrites to increase the scope of a function (e.g. from lambda or private
  helper to public API) should not scale with the length of the function. (For
  example,
  [the Jai language has a goal of making this refactoring easy by keeping function syntax uniform](https://github.com/BSVino/JaiPrimer/blob/master/JaiPrimer.md#code-refactoring).)
  Similarly, it would be desirable to be able to change the visibility of a name
  without having to update all references to that name (contrast with
  [Go](https://tour.golang.org/basics/3) and
  [Python](https://docs.python.org/3/tutorial/classes.html#private-variables)).
- Carbon should allow a field to be replaced by a "property", implemented using
  a getter and/or setter function. This allows changing the data representation
  or validation of invariants of a type without having to update all of the
  accesses. **Concern:** This may be a rare enough need that it doesn't justify
  the ambiguity about whether something is a property or member access.
- Changing how generic a function is (e.g. non-generic -> generic or template ->
  generic) should not affect callers of that function.

Automation:

- Favor computer-friendly unambiguous syntax to facilitate tooling that
  automates refactoring. It should be straightforward to, for example, produce
  an index that allows you to look up all call sites for a function.
- When a refactoring requires multiple steps (since it requires global changes /
  changes across packages), the language should allow the user to express the
  desired change in the source in a way that _standard_ tooling will make the
  desired change in dependent packages. Standardization of tooling means that a
  package can be kept up to date with much lower cost.

Incremental changes:

- API changes must not break callers until they can be updated, as detailed in
  ["Carbon principle: API evolution"](https://github.com/josh11b/carbon-lang/blob/principle-api-evolution/docs/project/principles/principle-api-evolution.md).
- Carbon should provide an alias facility to allow a symbol to exist under two
  names, potentially in different packages, so either can be used during a
  transition. This one facility should work for all names independent of whether
  they are names of values, types, etc.

## Proposals relevant to this principle

- Properties:
  [Carbon struct types: Properties (TODO)](#broken-links-footnote)<!-- T:Carbon struct types --><!-- A:#heading=h.di9erwm5g1br -->,
  [Carbon interface evolution: Modifying a data field (TODO)](#broken-links-footnote)<!-- T:Carbon interface evolution --><!-- A:#heading=h.kfti6vaebvs8 -->

## Broken links footnote

Some links in this document aren't yet available, and so have been directed here
until we can do the work to make them available.

We thank you for your patience.
