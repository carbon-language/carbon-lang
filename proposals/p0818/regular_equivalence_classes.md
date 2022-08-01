# Regular equivalence classes

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Problem

For generics, users will define _interfaces_ that describe types. These
interfaces may have associated types (see
[Swift](https://docs.swift.org/swift-book/LanguageGuide/Generics.html#ID189),
[Rust](https://doc.rust-lang.org/rust-by-example/generics/assoc_items/types.html)):

```
interface Container {
  let Elt:! Type;
  let Slice:! Container;
  let Subseq:! Container;
}
```

The next step is to express constraints between these associated types.

-   `Container.Slice.Elt == Container.Elt`
-   `Container.Slice.Slice == Container.Slice`
-   `Container.Subseq.Elt == Container.Elt`
-   `Container.Subseq.Slice == Container.Slice`
-   `Container.Subseq.Subseq == Container.Subseq`

Languages commonly express these constraints using `where` clauses, as in
[Swift](https://docs.swift.org/swift-book/LanguageGuide/Generics.html#ID557) and
[Rust](https://doc.rust-lang.org/rust-by-example/generics/where.html), that
directly represent these constraints as equations.

```
interface Container {
  let Elt:! Type;
  let Slice:! Container where .Elt == Elt
                          and .Slice == Slice;
  let Subseq:! Container where .Elt == Elt
                           and .Slice == Slice
                           and .Subseq == Subseq;
}
```

These relations give us a semi-group, where in general deciding whether two
sequences are equivalent is undecidable, see
[Swift type checking is undecidable - Discussion - Swift Forums](https://forums.swift.org/t/swift-type-checking-is-undecidable/39024).

If you allow the full undecidable set of `where` clauses, there are some
unpleasant options:

-   Possibly the compiler won't realize that two expressions are equal even
    though they should be.
-   Possibly the compiler will reject some interface declarations as "too
    complicated", due to "running for too many steps", based on some arbitrary
    threshold.

Furthermore, the algorithms are expensive and perform extensive searches.

### Goal

The goal is to identify some restrictions such that:

-   We have a terminating algorithm to decide if a set of constraints meets the
    restrictions.
-   For constraints that meet the restrictions we have a terminating algorithm
    for deciding which expressions are equal.
-   Expected use cases satisfy the restrictions.

The expected cases are things of these forms:

-   `X == Y`
-   `X == Y.Z`
-   `X == X.Y`
-   `X == Y.X`
-   and _some_ combinations and variations

For ease of exposition, I'm going to assume that all associated types are
represented by single letters, and omit the dots `.` between them.

## Idea

Observe that the equivalence class of words you can rewrite to in these expected
cases are described by a regular expression:

-   `X` can be rewritten to `(X|Y)` using the rewrite `X == Y`
-   `X` can be rewritten to `XY*` using the rewrite `X == XY`
-   `X` can be rewritten to `Y*X` using the rewrite `X == YX`

These regular expressions can be used instead of the rewrite rules: anything
matching the regular expression can be rewritten to anything else matched by the
same regular expression. This means we can take any sequence of letters, replace
anything that matches the regular expression by the regular expression itself,
and get a regular expression that matches the original sequence of letters. We
can even get a shortest/smallest representative of the class by dropping all the
things with a `*` and picking the shortest/smallest between alternatives
separated by a `|`.

### Question

Given a set of rewrite equations, is there a terminating algorithm that either:

-   gives a finite set of regular expressions (or finite automata) that
    describes their equivalence classes, or
-   rejects the rewrites.

We would be willing to reject cases where there are more equivalence classes
than rewrite rules. An example of this would be the rewrite rules `A == XY` and
`B == YZ` have equivalence classes `A|XY`, `B|YZ`, `XYZ|AZ|XB`.

## Problem

In particular, we've identified two operations for combining two finite automata
depending on how they overlap:

-   **Containment operation:** If we have automatas `X` and `Y`, where `Y`
    completely matches a subsequence matched by `X`, then we need to extend `X`
    to include all the rewrites offered by `Y`. For example, if `X` is `AB*` and
    `Y` is `(B|C)`, then we can extend `X` to `A(B|C)*`.
-   **Union operation:** If we have automatas `X` and `Y`, and there is a string
    `ABC` such that `X` matches the prefix `AB` and `Y` matches the suffix `BC`,
    then we need to make sure there is a rule for matching `ABC`.

Even though it is straightforward to translate a single rewrite rule into a
regular expression that matches at least some portion of the equivalence class,
to match the entire equivalence classes we need to apply these two operations in
ways that can run into difficulties.

One concern is that an individual regular expression might not cover the entire
equivalence for a rewrite rule. We may need to apply the union and containment
operations to combine the automata _with itself_. For example, we may convert
the rewrite rule `AB == BA` into the regular expression `AB|BA`, but this
doesn't completely describe the set of equivalence classes that this rule
creates, since it triggers the union operation twice to add `AAB|ABA|BAA` and
`ABB|BAB|BBA`. These would also trigger the union operation, and so on
indefinitely, so the rewrite rule `AB == BA` does not have a finite set of
regular equivalence classes. Similarly the rule `A == BCB` has an infinite
family of equivalence classes: `BCB|A`, `BCBCB|ACB|BCA`,
`BCBCBCB|ACBCB|ACA|BCACB|BCBCA`, ...

The other concern is that even in the case that each rule by itself can be
faithfully translated into a regular expression that captures the equivalence
classes of the rewrite, attempts to combine multiple rules using the two
operations might never reach a fixed point.

In both cases, we are looking for criteria that we can use to decide if the
application of the operations will terminate. If it would not terminate, then we
need to report an error to the user instead of applying those two operations
endlessly.

## Examples

### Regular single rules

-   Anything where there are no shared letters between the two sides, and no
    side has a prefix matching a suffix
    -   `A == B` => `A|B`
    -   `A == BC` => `A|BC`
    -   `A == BBC` => `A|BBC`
    -   `A == BCC` => `A|BCC`
-   `A == AB` => `AB*`
-   `A == BA` => `B*A`
-   `A == AA` => `AA*` or `A*A` or `A+`
-   `A == ABC` => `A(BC)*`
-   `A == ABB` => `A(BB)*`
-   `A == AAA` => `A(AA)*`
-   `A == BCA` => `(BC)*A`
-   `A == BBA` => `(BB)*A`
-   `A == ACA` => `A(CA)*` or `(AC)*A`

### Regular combinations

-   `A==AB` + `A==AC` => `AB*` + `AC*` => `A(B|C)*`
-   `A==AB` + `B==BC` => `AB*` + `BC*` => `A(B|C)*`, `BC*` with relation
    `A(B|C)* BC* == A(B|C)*`; Note: `A == AB == ABC == AC`
-   `A==AB` + `C==CB` => `AB*`, `CB*`, no relations
-   `A==AB` + `A==CA` => `AB*` + `C*A` => `C*AB*`
-   `A==AB` + `C==BC` => `AB*` + `B*C` => `AB*`, `B*C`, where if you have `AB*C`
    it doesn't matter how you split the `B`s between `AB*` and `B*C`.
-   `A==AB` + `B==AB` => `AB*` + `A*B` => `(A|B)+`
-   `A==AB` + `B==CB` => `AB*` + `C*B` => `A(C*B)*`, `C*B` with `A(C*B)* C*B` ->
    `A(C*B)*`
-   `A==AB` + `B==CB` + `C==CD` => `AB*` + `C*B` + `CD*` => `A((CD*)*B)*`,
    `(CD*)*B`, `CD*`
-   `A==ABC` + `D==CB` => `A(BC)*` + `(CB|D)` => `A(BD*C)*`, `A(BD*C)*BD*`,
    `(CB|D)`
-   `A==ABBBBB` + `A==ABBBBBBBB` (or `A==AB^5` + `A==AB^8`)=> `A(BBBBB)*` +
    `A(BBBBBBBB)*` => `AB*`, since `A=AB^8=AB^16=AB^11=AB^6=AB`

### Not regular

-   `AB == BA`, has equivalence classes: `AB|BA`, `AAB|ABA|BAA`, `ABB|BAB|BBA`,
    ...
-   Similarly: `AB==C` + `C==BA`
-   `ABC == CBA`, has equivalence classes: `ABC|CBA`, `ABABC|ABCBA|CBABA`,
    `ABCBC|CBABC|CBCBA`, ...
-   `A == BAB`, involves back references or counting the number of `B`s:
    `A|BAB|BBABB|BBBABBB|`...
-   `A == BAC`, involves matching the number of `B`s to `C`s:
    `A|BAC|BBACC|BBBACCC|`...
-   `A == BCB`, has equivalence classes: `BCB|A`, `BCBCB|ACB|BCA`,
    `BCBCBCB|ACBCB|ACA|BCACB|BCBCA`, ...
-   `A == BB`, has equivalence classes: `A|BB`, `BBB|AB|BA`, `(A|BB)(A|BB)|BAB`,
    ...
-   `A == BBB`, has equivalence classes: `A|BBB`, `BBBB|AB|BA`,
    `BBBBB|ABB|BAB|BBA`, ...
-   `AA == BB`, has equivalence classes: `AA|BB`, `AAA|ABB|BBA`, `BAA|BBB|AAB`,
    ...
-   `AB == AA`, has equivalence classes: `A(A|B)`, `A(A|B)(A|B)`,
    `A(A|B)(A|B)(A|B)`, ...
-   `AB == BB`, similarly
-   `AB == BC`, has equivalence classes: `(AB|BC)`, `(AAB|ABC|BCC)`, ...,
    `A`^i`BC`^(N-i)
-   `A == AAB` and `A == BAA`

## References

See these notes from discussions:

-   [Regular equivalence classes 2021-09-23](https://docs.google.com/document/d/1iyL7ZDfWT6ZmDSz6Fp8MrLcl5nOQc4ItQbeL3QlVXYU/edit#)
-   [Carbon minutes (rolling): Open discussions: 2021-09-27](https://docs.google.com/document/d/1UelNaT_j61G8rYp6qQZ-biRddTuGcxJtqXxrVbjB9rA/edit#heading=h.qh4b1gy7o5el)
-   [Carbon minutes (rolling): Open discussions: 2021-09-28](https://docs.google.com/document/d/1UelNaT_j61G8rYp6qQZ-biRddTuGcxJtqXxrVbjB9rA/edit#heading=h.9a1r39mla9ho)
-   [Regular equivalence classes 2 2021-10-01](https://docs.google.com/document/d/1xGDRSV6q534-CoDZnzuZJmzhty2yPki2ZfDDPLSHYek/edit#)
