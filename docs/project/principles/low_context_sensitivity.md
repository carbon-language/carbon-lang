# Principle: Low context-sensitivity

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Principle](#principle)
    -   [Mitigations of context-sensitive costs](#mitigations-of-context-sensitive-costs)
        -   [Visual aids](#visual-aids)
        -   [Contextual _validity_ rather than _meaning_](#contextual-validity-rather-than-meaning)
        -   [Reduced cost of mistakes](#reduced-cost-of-mistakes)
            -   [Compiler-checked context](#compiler-checked-context)
-   [Applications of the principle](#applications-of-the-principle)
    -   [Imports and namespaces](#imports-and-namespaces)
    -   [Name shadowing](#name-shadowing)
    -   [Flow-sensitive typing](#flow-sensitive-typing)
    -   [Coherence of names and generics](#coherence-of-names-and-generics)
    -   [Performance](#performance)

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
-   It can be _distant_: the further away from the current declaration or
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
The goal is to make a bunch of orthogonal mechanisms, each of which are easily
understood and act in unsurprising ways.

If that next step is to loosen restrictions, that is generally easier to do
while maintaining compatibility with existing code than adding new restrictions.

### Mitigations of context-sensitive costs

There are several ways that the potential costs of context-sensitive code can be
mitigated. These techniques can and should be leveraged to help minimize and
mitigate the contextual costs of Carbon features, and in some cases may provide
a path to a feature that would otherwise be prohibitively costly.

#### Visual aids

A direct way to reduce contextual costs is through lexical and syntactic
structures that form visual aids. These can both reinforce what the context is
and aid the reader in the expensive aspect of navigating the context. For
example, representing contexts with indentation, or IDE highlighting of matching
parentheses and braces. These visual hints make it easier for developers to
notice contextual elements.

#### Contextual _validity_ rather than _meaning_

When the context only affects the _validity_ of code, but not its meaning, the
costs are significantly reduced. In that case, understanding the meaning or
behavior of the code doesn't require context, and a developer can easily rely on
the compiler to check the validity. A simple example of this is contextually
valid syntax, which is relatively common and inexpensive. However, reusing the
same syntax with different contexts _with different meanings_ shifts the
contextual information from simple validity to impacting the meaning of code.

#### Reduced cost of mistakes

Another mitigation for the costs of context-sensitive code is when the cost of a
mistake due to the context is low. Some simple examples:

-   Context-sensitivity in comments is less expensive in general than in code.
-   In places where the general meaning is clear, developers can safely and
    reliably work with that general understanding, and the context only provides
    a minor refinement.

##### Compiler-checked context

Another way the costs of mistakes can be reduced is when the compiler can
reliably detect them. This is the fundamental idea behind statically
type-checked languages: the compiler enforcement reduces the contextual cost of
knowing what the types are. How early and effectively the compiler can detect
the mistakes also plays a role in reducing this cost, which is part of the value
proposition for
[definition-checked](/docs/design/generics/terminology.md#definition-checking)
generics.

An example of this situation in Rust is that the same syntax is used for a move
and a copy of the value in a variable. Those cases are distinguished by whether
the type implements a specific trait, which may not be readily ascertained. The
compiler verifies that the code never uses a variable that is no longer valid
due to having been moved from, which is expected to catch the problems that
could arise from this difference. Otherwise the semantic difference between a
move and a copy is considered in Rust to be low-enough stakes for there to be no
need to signal that difference in the code.

However, the reasoning that makes this example a good design on balance for Rust
doesn't necessarily apply to Carbon. The compiler is checking to prevent
_errors_, but it can't reliably check for unpredictable _performance_. Given
Carbon's priorities, that might make this level of contextual information still
too expensive.

More background on this area of Rust specifically is presented in
[their blog post on language ergonomics](https://blog.rust-lang.org/2017/03/02/lang-ergonomics.html).

## Applications of the principle

There are many parts of Carbon that could potentially be analyzed through this
lens, and we can't enumerate them all here. This section focuses on several
examples to help illustrate how the principle is likely to be relevant to
Carbon. They focus on either cases that showcase the principle in effect or
cases which make challenging tradeoffs of the costs in the principle.

### Imports and namespaces

There are several parts of the way
[imports](/docs/design/code_and_name_organization/#imports) and
[namespaces](/docs/design/code_and_name_organization/#namespaces) are designed
in Carbon that reflect applications of this principle:

-   Adding an import or reordering imports should never change behavior of
    existing code. This means the reader doesn't have to look through all the
    imports to understand how code behaves. This is also important for tooling,
    which should not have to worry about unwanted side effects when adding or
    sorting imports.

-   Carbon doesn't provide an analogy to C++'s
    [`using namespace`](https://en.cppreference.com/w/cpp/language/namespace#Using-directives)
    or a
    ["wildcard imports" mechanisms](/proposals/p0107.md#broader-imports-either-all-names-or-arbitrary-code)
    that merge the names from one namespace into another. Either would introduce
    ambiguity in where a name is coming from, making the code more
    context-sensitive.

-   Carbon doesn't support large blocks of code
    [inside a namespace declaration](/proposals/p0107.md#scoped-namespaces),
    where the reader would have to search for the beginning of the block to see
    what namespace applies.

### Name shadowing

We should limit how names can be reused with shadowing rules, so the meaning of
a name doesn't change in surprising ways between scopes. Further, if you find a
matching declaration you don't have to keep searching to see if there is another
that hides the one you found. This both expands the context you have to
consider, and is an opportunity to make a mistake identifying the correct
context, potentially leading to misunderstanding of the code.

### Flow-sensitive typing

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

### Coherence of names and generics

Carbon [packages](/docs/design/code_and_name_organization/#packages) are
designed to ensure all declared names belong to exactly one package and the
compiler can enforce Carbon's equivalent
[one-definition rule (ODR)](https://en.wikipedia.org/wiki/One_Definition_Rule).
This avoids an issue in C++ where the ODR is not reliably checked by the
compiler, which can leave the correctness of programs dependent on both distant
and subtle contextual information.

Similarly, Carbon generics should have
[coherence, like Rust](https://github.com/Ixrec/rust-orphan-rules#what-is-coherence),
where types have a single implementation of an interface. And this should be
enforced by the compiler, using rules like
[Rust's orphan rules](https://github.com/Ixrec/rust-orphan-rules#what-are-the-orphan-rules).

### Performance

Since
[Carbon's number one goal is performance](/docs/project/goals.md#performance-critical-software),
it is important that the performance characteristics of code be predictable and
readily determined by readers. This argues that those characteristics should not
depend on expensive context. For example, Carbon should not provide a
`dynamic_cast` facility with the same capabilities of C++'s where distant
aspects of the inheritance structure can cause surprising performance
differences. Similarly, Carbon should try to ensure normal looking method calls
and data member access don't have the surprising performance costs caused by
virtual inheritance in C++.

More generally, Carbon should avoid features with hidden costs, particularly
when they scale based on subtle aspects of the context where those features are
used.
