# Updates to pattern matching for objects

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/5164)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Proposal](#proposal)
-   [Alternatives considered](#alternatives-considered)
    -   [Alternative approaches for declaring movable bindings](#alternative-approaches-for-declaring-movable-bindings)
        -   [Treat all bindings under `var` as variable bindings](#treat-all-bindings-under-var-as-variable-bindings)
        -   [Make `var` a binding pattern modifier](#make-var-a-binding-pattern-modifier)
    -   [Alternative approaches to the other problems](#alternative-approaches-to-the-other-problems)
        -   [Initialize storage once pattern matching succeeds](#initialize-storage-once-pattern-matching-succeeds)
        -   [Allow variable binding patterns to alias across `case`s](#allow-variable-binding-patterns-to-alias-across-cases)

<!-- tocstop -->

## Abstract

This proposal re-affirms (with additional rationale) that a `var` pattern
declares a durable complete object, and refines the terminology for binding
patterns in a `var` pattern to be more explicit about the intended semantics. It
also makes several other changes and clarifications to the semantics of pattern
matching on objects:

-   The storage for a variable pattern is initialized eagerly, rather than being
    deferred until the end of pattern matching.
-   Any initializing expressions in the scrutinee of a `match` statement are
    materialized before matching the `case`s.
-   An initializing expression can only initialize temporary storage or a single
    variable pattern, not a tuple/struct pattern or a subobject of a variable
    pattern. Removing this limitation is left as future work.

Finally, as a drive-by fix, it clarifies what parts of the `match` design are
still placeholders.

## Problem

Discussions arising from the implementation of `var` patterns have surfaced some
problems with how pattern matching deals with objects:

-   If binding patterns bind to subobjects of an enclosing `var` object,
    destructively moving them will lead to double-destruction.
-   Deferring initialization of variable bindings contradicts our specification
    of copy/move elision, and makes `if`-guards much less useful.
-   It was unclear what happens when a `match` statement's scrutinee is an
    initializing expression, since the result of an initializing expression
    can't be reused.
-   It was unclear whether and how copy/move elision applies when initializing a
    subobject of a variable binding or tuple/struct pattern.

The first point deserves some elaboration. We don't yet have a concrete proposal
for move semantics, but it's possible to discern the overall direction well
enough to see a problem with how it interacts with `var`. The following sketch
of move semantics should be considered a **placeholder**, not an approved
design.

`~x` is a _move_ from the reference expression `x`. It is an initializing
expression that initializes an object with the value that `x` held prior to the
move, while arbitrarily mutating `x` to make the move more efficient. By default
`~x` is _destructive_, meaning that it ends the lifetime of `x`; under some
conditions it may instead leave `x` in an
[unformed state](/docs/design/README.md#unformed-state), or make a copy of `x`,
but only if the type supports doing so.

Consider the following code, where `X` and `Y` are types that are movable but
not copyable, have nontrivial destructors, and do not have unformed states:

```carbon
fn A() -> (X, Y);
fn B(var x: X);

fn F() {
  var (x: X, y: Y) = A();
  B(~x);
}
```

Under the current design of pattern matching, the first line of `F` declares a
complete object of type `(X, Y)`, and binds `x` and `y` to its elements. At the
end of the body of `F`, that tuple object is still live, so its destructor will
run, which will recursively run the destructors for its elements. However, its
first element was already destroyed by `~x`, so this would result in
double-destruction of that sub-object.

In order to avoid that problem, the expression `~x` must be ill-formed. More
generally, if a `var` pattern contains a tuple or struct subpattern, the
bindings it declares cannot be moved from (unless, possibly, their types permit
them to be non-destructively moved, and/or safely destroyed twice). In order to
be well-formed, the first line of `F` must be rewritten as:

```carbon
  let (var x: X, var y: Y) = A();
```

This makes the code more verbose, and may be surprising to users.

## Proposal

The decision that `var` declares a single complete object is reaffirmed,
notwithstanding the problem described above. To be more explicit about the
intended meaning, we will adjust the terminology:

-   The term "variable binding pattern" is now limited to a binding pattern that
    binds to the entire object declared by an enclosing `var` pattern.
-   We introduce the term "reference binding pattern" to refer to any binding
    pattern that has an enclosing `var` pattern. Thus, every variable binding
    pattern is a reference binding pattern, but not vice-versa.

To address the other problems:

-   The storage for a variable binding pattern is initialized eagerly, rather
    than being deferred until the end of pattern matching.
-   Any initializing expressions in the scrutinee of a `match` statement are
    materialized before matching the `case`s.
-   An initializing expression can only initialize temporary storage or a single
    variable binding, not a tuple/struct pattern or a subobject of a variable
    binding. Removing this limitation is left as future work.

## Alternatives considered

### Alternative approaches for declaring movable bindings

See
[leads issue #5250](https://github.com/carbon-language/carbon-lang/issues/5250)
for further discussion of this problem.

#### Treat all bindings under `var` as variable bindings

We could treat all bindings under a `var` pattern as variable bindings, instead
of limiting that to the case of a single binding pattern immediately under the
`var` pattern. This would mean that all such bindings declare complete objects,
and hence are movable. By the same token, it would mean that a `var` pattern
does not declare a complete object.

However, we are likely to eventually need a way for a pattern to declare a
complete object and bind names to its parts, for example to support Rust-style
[@-bindings](https://doc.rust-lang.org/book/ch19-03-pattern-syntax.html#-bindings)
or to support in-place destructuring of user-defined types (where destroying the
object in order to make its subojects movable could have unwanted side effects).
This approach would make it difficult to do that with good ergonomics, because
both `var` and the hypothetical complete-object pattern syntax would change the
meanings of nested bindings, and the behavior of destructuring, in conflicting
ways.

This approach also expresses the programmer's intent less clearly, which could
harm readability, because a `var` with several nested bindings could be intended
to make those bindings movable, or just to make them mutable.

That also makes this option less future-proof. If we start with this approach
and later migrate to the status quo because we need the extra expressive power,
we would need to somehow infer the missing intent information. Conversely, if we
start with the status quo and later conclude we aren't going to make its extra
expressive power observable, existing valid code will remain valid, with the
same behavior, after switching to this approach.

The only major advantage of this approach over the status quo is that it's less
verbose when declaring multiple movable bindings, and may be less surprising to
users because it's less restrictive by default. However, those advantages aren't
significant enough to offset those costs.

#### Make `var` a binding pattern modifier

The current design makes binding patterns fairly context-sensitive, which we
generally [try to avoid](/docs/project/principles/low_context_sensitivity.md): a
binding pattern declares a variable binding pattern if it's the immediate child
of a `var` pattern, a non-variable reference binding pattern if it's an indirect
descendant of a `var` pattern, and a value binding pattern otherwise.

We could avoid context-sensitivity by making `var` a modifier on binding
patterns, like `template`. That would mean there are no reference binding
patterns, and the distinction between variable and value binding patterns is
always purely local.

However, allowing `var` to apply to many bindings at once improves the
ergonomics of destructuring into variables, which is likely to be a common use
case. Furthermore, the offsetting costs of allowing that are fairly minimal:

-   We are likely to eventually need reference bindings anyway, so introducing
    those semantics now isn't adding a cost, it's just incurring that cost
    sooner.
-   The cost of context-sensitivity in patterns is comparatively low, because
    patterns are generally quite small, and there's rarely much need to factor
    subpatterns out of their initial context.

### Alternative approaches to the other problems

#### Initialize storage once pattern matching succeeds

Prior to this proposal, the status quo was that the storage associated with a
pattern is not initialized until we know that the complete pattern matches. This
helps avoid situations where a `case` that does not match nevertheless has
visible side effects. However, this approach has several major drawbacks:

-   It contradicts or at least greatly complicates the guarantee that
    declarations like `var x: X = F();` do not require any temporary storage,
    because it would imply that `F()` is not even evaluated until we know the
    pattern matches. That's feasible for irrefutable patterns like this one, but
    not for refutable patterns. Even if we were willing to limit that guarantee
    to contexts that require irrefutable patterns, it would complicate the
    implementation, because the underlying logic would have major structural
    differences in the two cases.
-   It precludes using a variable binding before its enclosing complete pattern
    is known to match, because that variable would not be initialized. In
    particular, that means an
    [`if`-guard](/docs/design/pattern_matching.md#guards) cannot use variable
    bindings from the pattern it guards. Practically all the motivating use
    cases for `if`-guards involve using bindings from the guarded pattern, so
    this is tantamount to making `var` and `if` mutually exclusive.

#### Allow variable binding patterns to alias across `case`s

Consider the following code:

```carbon
fn F() -> X;
fn G() -> i32;

match ((F(), G())) {
  case (var x: X, 0) => { ... }
  case (var x: X, 1) => { ... }
  case (var x: X, 2) => { ... }
  case (var x: X, 3) => { ... }
  ...
}
```

Under this proposal, the result of `F()` is materialized in temporary storage,
and then copied into the storage for `x` as part of matching each `case` (it
can't be moved, because it must remain available for the next `case`). As a
result, this code may make as many copies of `X` as there are `case`s, and
doesn't even compile if `X` isn't copyable.

We could instead treat those `var x: X` declarations as aliases for the
materialized temporary. However, in order to generalize that approach we would
need to answer questions like:

-   Do the bindings alias between `case (var x: X, 0)` and `case var (x: X, 1)`?
-   Do the bindings alias if their names are different?
-   Do the bindings alias if the two `case`s are separated by another `case`
    that doesn't have a binding in that position?
-   Do the bindings alias if they have the same type as each other, but not the
    same type as the scrutinee?

Furthermore, the behavior of these rules would need to be intuitive and
unsurprising, because any change that prevents aliasing (in even one case) could
break the build or cause surprising performance changes.

Even in the best case, this approach is inherently limited. If the bindings have
different types, at most one of them can be an alias for the scrutinee; the
other must be initialized by a copying conversion.

This approach would open up the possibility of a non-taken `case` mutating the
state of the scrutinee, by invoking a mutating operation on a variable binding
in an expression pattern or an `if`-guard. We may be able to statically forbid
such mutations, but that will be easier to assess once we understand the role of
mutability in our broader safety story.

We conjecture that in most if not all cases where aliasing would be desirable,
it will be straightforward to rewrite the code to avoid the problem. For
example, the example above can be rewritten as:

```carbon
fn F() -> X;
fn G() -> i32;

var x: X = F();
match (G()) {
  case 0 => { ... }
  case 1 => { ... }
  case 2 => { ... }
  case 3 => { ... }
  ...
}
```

We can revisit this alternative if that conjecture turns out to be incorrect in
practice.
