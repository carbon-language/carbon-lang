# Carbon principle: Built-in types are not privileged

<!--
Part of the Carbon Language, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Principle

Carbon libraries should be able to make new types that have all the same
capabilities as builtin types, including the syntax for use. This allows:

- The ecosystem to evolve without requiring heavyweight modifications to the
  language itself by allowing new types to be developed in libraries. See
  [Carbon Goals: Both software and language evolution](https://github.com/jonmeow/carbon-lang/blob/proposal-goals/docs/project/goals.md#both-software-and-language-evolution).
- A form of orthogonality which aids simplicity. See
  [Carbon Goals: Code that is simple and easy to read, understand, and write](https://github.com/jonmeow/carbon-lang/blob/proposal-goals/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write).

**Background:**
[Swift implements all builtin types in Swift itself](https://github.com/apple/swift/tree/master/stdlib/public/core).
The implementations of types like `Int8` do involve LLVM intrinsics to perform
addition, etc., but because they are defined as ordinary Swift types, they
guarantee that they are not special.

### Caveats

- Possibly (still to be decided) this will not extend to operations like `&&`
  and `||` that can control whether their arguments get evaluated. (Note: Swift
  has a mechanism called `@autoclosure` for arguments that may not be evaluated,
  it is uncertain whether Carbon will have a similar mechanism.)
- One capability we likely won't support for non-builtins: having a name in the
  global namespace.
- There will be convenient syntax for writing literals for builtin types, which
  we don't feel obligated to support for non-builtins.

## Applications of these principles

We will support operator overloading, so a built-in type can define how `+`,
etc. work. There should not be any operation that works on built-in types that
can't be defined for library types.

We will support inheriting from built-in types.

## Proposals relevant to these principles

- [Carbon struct types (TODO)](#broken-links-footnote)<!-- T:Carbon struct types -->

## Broken links footnote

Some links in this document aren't yet available, and so have been directed here
until we can do the work to make them available.

We thank you for your patience.
