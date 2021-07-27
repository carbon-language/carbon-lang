# Carbon: Low context-sensitivity principle

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Principle](#principle)
-   [Applications of the principle](#applications-of-the-principle)

<!-- tocstop -->

## Principle

Carbon should favor designs and mechanisms that are not sensitive to context.
Instead, we should favor constructs that are not ambiguous so that they don't
need context for disambiguation. This is in service to the goal that
[Carbon code is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write).
In particular, this is about prioritizing reading and understanding over
writing. We should be willing to trade off conciseness, which still benefits
reading as well as writing, for a sufficiently impactful reduction in the amount
of context needed to read and understand code.

Context can be expensive in different ways, for example:

-   It can be _large_: it might require looking through a lot of lines of code
    to find all of the relevant contextual information.
-   It can be _distant_: the farther away from the current declaration or
    definition, the more expensive it is to find contextual information. This
    can scale from a separate definition in the same file, to a separate file,
    or even to a separate package.
-   It can be _unpredictable_: it might require careful searching of large
    bodies of code to locate the contextual information if its location cannot
    be predicted.
-   It can be _subtle_: the contextual clues might be easily missed or mistaken.

Code that isn't context sensitive is easier to copy or move between contexts,
like files or functions. It is code that needs fewer changes when it is
refactored, in support of
[software evolution](/docs/project/goals.md#software-and-language-evolution).

In general, we should start with more restrictive constructs that limit
ambiguity and see if we can make them work. If we find those restrictions are
burdensome, we will then have more information to inform the next step. Ideally
we would address those use cases with simple tools that solve multiple problems.
The goal is to make a bunch of orthogonal mechanisms, that each are easy to
understand and act in unsurprising ways.

If that next step is to loosen restrictions, that is generally easier to do
while maintaining compatibility with existing code than adding new restrictions.

**Question:** A way in which context could be less expensive is if the stakes
are low. That is if a mistake in understanding is unlikely to lead to a bug or
cause a misunderstanding in the semantics of code.

A specific example of this is when the compiler can detect mistakes.

**Question:** In the Rust world, they accept context that would be potentially
expensive in some cases where the compiler can verify that what is written is
correct. How much do we want to allow that? Context affecting meaning is much
more concerning than context affecting validity that the compiler can check.

**Background:** See
[this post on language ergonomics in the Rust blog](https://blog.rust-lang.org/2017/03/02/lang-ergonomics.html).

An example of this situation in Rust is that the same syntax is used for a move
and a copy of the value in a variable. Those cases are distinguished by whether
the type implements a specific trait, which may not be readily ascertained. The
compiler verifies that the code never uses a variable that is no longer valid
due to having been moved from, which is expected to catch the problems that
could arise from this difference. Otherwise the semantic difference between a
move and a copy is considered in Rust to be low-enough stakes for there to be no
need to signal that difference in the code.

## Applications of the principle

Adding an import or reodering imports should never change behavior of existing
code. This means you don't have to look through all imports to understand how
code behaves. This is also important for tooling, which should not have to worry
about unwanted side effects when adding an import.

We should limit how names can be reused with shadowing rules, so the meaning of
a name doesn't change in surprising ways between scopes. Further, if you find a
matching declaration you don't have to keep searching to see if there is another
that hides the one you found. This both expands the context you have to
consider, and is an opportunity to make a mistake identifying the correct
context, potentially leading to misunderstanding of the code.

This principle is an argument against
[flow-sensitive typing](https://en.wikipedia.org/wiki/Flow-sensitive_typing),
where the type of a name can change depending on control flow. For example,
[Midori used this for optional types](http://joeduffyblog.com/2016/02/07/the-error-model/#the-syntax).
If we were to support this in Carbon, you could unwrap an optional value by
testing it against `None`.

```
var x: Optional(Int) = ...;
if (x != None) {
  // x has type Int.
  PrintInt(x);
}
// x is back to type Optional(Int).
```

This can be taken farther, this example has `x` taking on three different types:

```
var x: Optional(Optional(Int)) = ...;
if (x != None) {
  // x has type Optional(Int).
  if (x != None) {
    // x has type Int.
    PrintInt(x);
  }
  // x has type Optional(Int).
}
// x has type Optional(Optional(Int)).
```

The concern here is that the context is very subtle. The type of `x` is affected
by otherwise ordinary-looking `if` statements and closing braces (`}`).

While we might not want to completely eliminate the possibility of
flow-sensitive typing in Carbon, it would have to overcome a large hurdle. We
would only want a flow-sensitive feature if it delivered sufficiently large
usability, consistency, or expressivity gains.

The
[one-definition rule (ODR)](https://en.wikipedia.org/wiki/One_Definition_Rule)
in C++ says that there should be only one definition for a name in a program,
but there is no way for the compiler to enforce it. As a result you can get
mysterious inconsistent behavior. In Carbon, namespaces should be local to
packages so that it is possible to check that names have a single definition by
checking each package independently. Similarly, Carbon generics should have
[coherence, like Rust](https://github.com/Ixrec/rust-orphan-rules#what-is-coherence),
where types have a single implementation of an interface. And this should be
enforced by the compiler, using rules like
[Rust's orphan rules](https://github.com/Ixrec/rust-orphan-rules#what-are-the-orphan-rules).

This principle argues against "using namespace" or "wildcard imports" mechanisms
that merge the names from one namespace into another. They introduce ambiguity
in where a name is coming from.

It argues against having a large block of code inside a namespace declaration,
where you have to search for the beginning of the block to see what namespace
you are in.

Since
[Carbon's number one goal is performance](/docs/project/goals.md#performance-critical-software),
it is important that the performance characteristics of code be predictable and
readily determined by readers. This argues that those characteristics should not
depend on expensive context. In contrast, in C++, the performance cost of a
`dynamic_cast` depends on the relationship between the source and target types.
A cast from a base to a derived class will be significantly cheaper than a cast
to a sibling. Similarly, virtual inheritance will add cost to ordinary looking
method calls and implicit conversions, even with single inheritance. When
combined with multiple inheritance, the vtable pointer for an object can be set
over and over during construction. Carbon should avoid features with hidden
costs, particularly when they scale based on subtle aspects of the context where
those features are used.
