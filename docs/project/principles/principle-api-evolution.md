# Carbon principle: API evolution

<!--
Part of the Carbon Language, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Principle

The language should support the evolution of interfaces between packages. This
is in service to
["Both software and language evolution"](https://github.com/jonmeow/carbon-lang/blob/proposal-goals/docs/project/goals.md#both-software-and-language-evolution)
from
["Carbon Goals"](https://github.com/jonmeow/carbon-lang/blob/proposal-goals/docs/project/goals.md).

- There should be defined ways for adding to, removing from, and updating APIs.
- The language should provide contracts that define what changes can be made
  safely with only local verification, and that set should be made as large as
  possible.
- Carbon should provide tooling to automate making large scale or global changes
  to support evolution as much as possible.

### Caveats

- Not via ABI compatibility:
  [Non-goal: Stable language and library ABI](https://github.com/jonmeow/carbon-lang/blob/proposal-goals/docs/project/goals.md#stable-language-and-library-abi).
  This is about source compatibility, not binary compatibility.
- We don't expect APIs to be completely backwards or forwards compatible
  forever:
  [Non-goal: Backwards or forwards compatibility](https://github.com/jonmeow/carbon-lang/blob/proposal-goals/docs/project/goals.md#backwards-or-forwards-compatibility);
  "our goals should be focused on _migration_ rather than _compatibility_."

## Applications of this principle

Adding a name to an API should be possible without breaking code elsewhere. In
general, this means look up rules should designate a single place to look up a
name. For example:

- This precludes features like extensions (e.g. in
  [Swift](https://docs.swift.org/swift-book/LanguageGuide/Extensions.html)), to
  avoid conflicts between names that you want to add to an API and names that
  are already added in some extension. **Question:** The concerns with
  extensions may be able to be addressed using a similar mechanism as for adding
  new names to a base class that may be used by some descendant, see below.
- Uniform Function Call Syntax has this problem: if someone adds a function
  `foo(x, y)` and callers are allowed to invoke it using `x.foo(y)`, then that
  can conflict with an addition of `foo` as a member to the type of `x`
  directly.
- Similarly we can't make an operation to concatenate interfaces by inlining all
  the methods of both. What happens when one of those interfaces adds a method
  that conflicts with the other?
- When a type implements an interface, that does not cause all of the
  interface's methods to be added to the type's namespace. Otherwise this could
  cause conflicts if a name is added to an interface that conflicts with the
  other methods of that type, or other interfaces that type implements. (Instead
  we make the methods of the type that match those in the interface an easy way
  of implementing the interface, and provide an explicit mechanism to use the
  interface methods instead of the type's.)
- The same principle precludes anonymous struct members, as in Go, where the
  container object's namespace adds all the member object's names (with names in
  the container winning in case of a conflict in the case of Go).

The more general principle is that the language should avoid adding something in
one place causing some code in some other place to suddenly become an error or
silently change semantics.

- This is a concern if you add an overload to a function that other code is
  taking the address of. We need some way to allow overloads to be added without
  breakage.
- Changing a function from non-generic to generic should not affect call sites.
- TODO: more examples of things we don't want

We need to be particularly careful about how to modify a base class without
breaking descendants. For example, a descendant may already have defined a
method with the same name as you want to add to the base class.

- There should be a path to add new functions to an API with default
  implementations without error. New functions should be marked as such, to
  create a transition period where name conflicts can be resolved.
- There should be a path to add new required functions to an API by first adding
  them with a default and then updating implementations of that API until you
  can remove the default implementation.
- There should be a path for removing functions by deprecating them while
  removing callers, and then disabling them until implementations are deleted.

Note that we could avoid these problems by avoiding inheritance e.g. by using
generics instead (as is done by Rust), but we expect to need to support
inheritance as part of our C++ transition story. We may want to discourage
inheritance in Carbon (for evolution reasons and others), in which case
mechanisms to support evolution may be overkill.

## Proposals relevant to this principle

- ["Carbon interface evolution" (TODO)](#broken-links-footnote)<!-- T:Carbon interface evolution -->

## Broken links footnote

Some links in this document aren't yet available, and so have been directed here
until we can do the work to make them available.

We thank you for your patience.
