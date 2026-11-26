# Redefining `alias`, `using`, and namespace aliases

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/8184)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
    -   [`alias`](#alias)
    -   [Namespace aliases](#namespace-aliases)
    -   [`using`](#using)
-   [Future work](#future-work)
    -   [Parameterized aliases](#parameterized-aliases)
    -   [Additional runtime expressions in `alias`](#additional-runtime-expressions-in-alias)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [`alias` of a reference expression](#alias-of-a-reference-expression)

<!-- tocstop -->

## Abstract

We redefine `alias` in terms of `let` with `auto` extended types rather than
operating purely in the domain of names: `alias A = B;` converts a compile-time
expression `B` to a value `V` and declares `let fwd A: auto = V;`, and
`alias ref A = R;` declares `let ref A: auto = R;` for reference expressions
with compile-time-known addresses. We add `using A.B.C;` as shorthand for
`alias C = A.B.C;`, and introduce `namespace C = A.B;` for namespace aliases
since namespaces are not first-class values.

## Problem

Prior to this proposal, `alias` is a construct we have been assuming would
operate purely in the domain of names, but has only been specified in a few
contexts. This approach has limitations that we need to address, as described in
[leads issue #5028: "More powerful aliasing feature"](https://github.com/carbon-language/carbon-lang/issues/5028).
The highest priority here is being able to give a name to a member of a
specific. For example, we've frequently assumed that declarations like
`alias String = BasicString(u8);`, `alias T = i32;` (equivalent to
`alias T = Core.Int(32);`), and `alias X = C(i32).X;` would somehow be allowed.

Leads issues
[#996: Generic let with auto?](https://github.com/carbon-language/carbon-lang/issues/996)
and
[#1371: Is let referentially transparent?](https://github.com/carbon-language/carbon-lang/issues/1371)
considered whether `let` could be combined with `auto` to make something
[referentially transparent](https://en.wikipedia.org/wiki/Referential_transparency),
as you would want from an `alias` construct, and concluded the answer was no
without a way to capture phase and category in addition to type.

## Background

This proposal was split out of
[proposal #5389](https://github.com/carbon-language/carbon-lang/pull/5389);
earlier revision and review history may be found in that pull request.

This proposal depends on pending
[proposal #8183: First-class extended types and `fwd`](https://github.com/carbon-language/carbon-lang/pull/8183),
specifically its `auto` extended type bindings (`let fwd x: auto = y;`).

It also builds on:

-   [Proposal #3720: Member binding operators](https://github.com/carbon-language/carbon-lang/pull/3720),
    which made more entities like class and interface members first-class
    values.
-   [Leads issue #5028: More powerful aliasing feature](https://github.com/carbon-language/carbon-lang/issues/5028).

> **Note:** This definition of `alias` is partially implemented in
> [PR #7190: Relax alias restrictions](https://github.com/carbon-language/carbon-lang/pull/7190).

## Proposal

### `alias`

We give `alias` a definition in terms of `let` bindings instead of operating on
names:

> `alias A = B;` converts `B` to a value `V` (which must be a compile-time
> expression), then means `let fwd A: auto = V;`

```carbon
alias a = 7;
alias String = BasicString(u8);
```

Note that this is more referentially transparent than `let a: auto =`... since
it preserves phase in addition to type. Unlike `let generic a: auto =`... it
doesn't erase the value of the right-hand side. Unlike
`let template a: auto =`... it doesn't change the phase to "template."

```carbon
fn F(generic T: type, template X: i32) {
  // `a` is symbolic, but not the same as `let generic`
  // since the fact that it is `buf(T)` is preserved.
  alias a = buf(T);
  // `b` has template phase.
  alias b = X;
}
```

In order to support renaming a variable, we allow an additional form:

> `alias ref A = R;` means `let ref A: auto = R;` (equivalently,
> `let fwd A: auto = R;`)

where `R` is a reference expression with address known at compile time (such as
the name of a variable). This syntax was decided in
[open discussion on 2026-11-18](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?tab=t.uvyw2c4zrz9e#heading=h.edvediz60r3d)
(see [the alternatives considered](#alias-of-a-reference-expression)). All other
expressions (including non-reference runtime expressions and expressions with
mixed compile-time and runtime phase) are forbidden. For example:

```carbon
var v: i32 = 7;
// ❌ Not a compile-time constant, need
// `ref` to alias a reference expression.
alias v_error = v;

// ✅ Allowed, reference expression with
// known address.
alias ref va = v;

var arr: array(i32, 64);
// ❌ Invalid with or without `ref`, since
// address unknown at compile time.
alias ref aa = arr[v];

// ❌ Invalid with or without `ref`, since
// non-reference runtime expression.
alias p_err1 = v + 2;
alias ref p_err2 = v + 2;
```

An `alias` declaration is allowed in these contexts:

-   at file scope or in the body of a function;
-   in the body of a `class`, but only to define a compile-time constant;
-   in the body of an `impl` in order to specify that an associated function is
    to be implemented by an existing function; and
-   in the body of an interface or named constraint, where the rewrite to `let`
    is additionally given the `final` modifier.

An `alias` may have an access modifier such as `private`, when allowed in that
scope.

This definition of `alias` was discussed on
[2025-06-03](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.hdj6hk4bwd2z).
This approach has the benefit of defining `alias` in terms of other Carbon
features, reducing the need to specify its semantics separately and removing a
situation where there were two ways to do something that were slightly
different.

### Namespace aliases

However, this definition of `alias` only works with first-class values. The good
news here is that almost everything we want to alias is first class,
particularly with
[proposal #3720](https://github.com/carbon-language/carbon-lang/pull/3720) which
made more entities like class and interface members first class. The main
exception is that we have no intention of making namespaces first class, because
the set of members is incoherent (different between libraries depending on what
they import) and in fact changes as new entities are declared in the namespace.
The solution is to, like C++, use another mechanism to alias namespaces. To this
end, we say that a `namespace` declaration may now optionally declare a
namespace alias by using a `=`:

```carbon
// Previous syntax: declare a namespace
// `A.B`:
namespace A.B;

// New syntax: declare `C` as an alias
// for namespace `A.B`:
namespace C = A.B;

// New syntax: declare `D` as a private
// namespace alias:
private namespace D = A.B;
```

Aliases of namespaces that belong to other packages are restricted to being
declared `private` or in an impl file.

### `using`

We define `using A.B.C;` to be a shorthand for `alias C = A.B.C;`. Note that the
argument is not required to have any periods, so `using D;` is the same as
`alias D = D;`, which creates an alias to a `D` found in a containing scope.
However, the expression must either be _something_`.`_word_ or a word by itself,
so that the resulting name is clear.

Example:

```carbon
class C {
  class D {}
}

class E(T: type) {
  // Equivalent to `alias D = C.D;`:
  using C.D;

  // Equivalent to `alias C = package.C;`:
  using C;

  // Afterward `E(X).T == X`.
  using T;
}
```

## Future work

### Parameterized aliases

Just like
[C++ supports parameterized type aliases](https://en.cppreference.com/w/cpp/language/type_alias.html),
it would be useful for Carbon to support parameterized aliases. These would act
a bit like functions, except that the arguments to the alias may be deduced. For
example, this generic `Double` alias:

```carbon
alias Double(T: type) = (T, T) as type;
```

would allow you to define a function that accepted any pair of two values of the
same type:

```carbon
// `T` is deduced from the argument type.
fn SumPair[T: Add](p: Double(T)) -> T {
  return p.0 + p.1;
}

let three: i32 = 3;
Assert(SumPair((three, three)) == 6);
```

A parameterized alias was mentioned in
[proposal #2875](p002875-functions-function-types-and-function-calls.md#function-pointers)
as a way to define a parameterized `FunctionPtr` type.

### Additional runtime expressions in `alias`

We could support more runtime expressions if there is a demonstrated need as
future work, perhaps by capturing the right-hand side as a lambda. For now we
don't want to support that since it introduces ambiguity concerns around when
the lambda is evaluated.

## Rationale

This proposal advances [Carbon's goals](/docs/project/goals.md) and
[principles](/docs/project/principles):

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    Defining `alias` in terms of `let fwd` enables aliasing specifics and other
    compile-time expressions (such as `alias String = BasicString(u8);`), while
    `using` eliminates repetitive boilerplate when re-exporting or importing a
    name into a scope. Requiring `alias ref` for mutable variables makes
    reference aliasing visible at the declaration site.
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution):
    Consistent aliasing for values, variables, members, and namespaces supports
    incremental migrations when renaming or relocating entities.
-   [Principle: One way to do things](/docs/project/principles/one_way.md):
    Defining `alias` in terms of `let` bindings avoids maintaining two slightly
    different mechanisms for introducing compile-time or reference names.

## Alternatives considered

### `alias` of a reference expression

We considered several alternatives for how [`alias`](#alias) should handle
reference expressions, reflecting a tension between thinking of `alias` as
"giving a name to an existing entity" (where aliasing a variable and aliasing a
function look the same, though pure name aliasing cannot handle expressions like
`buf(i32)`) versus "giving a name to a value" (where we want to distinguish when
the alias's value may change if the underlying storage is mutated):

-   **No `ref` marker (always referentially transparent)**: `alias A = R;`
    automatically preserves the category of the right-hand side (while still
    converting initializing expressions to values, and potentially capturing
    mixed-category tuples). We chose to require `ref` instead so readers can
    tell at the declaration site that the value accessed through `A` may change
    when the underlying storage is mutated through other names.
-   **Explicit `val` marker (`alias A = val E;`)**: bare `alias` follows the
    right-hand side's category, while `val` (or another keyword like `eval`)
    forces conversion to a value (which would be a special case of the previous
    option if Carbon had a general `val` operator). We decided against this
    since we didn't value highlighting the difference between naming and
    evaluating in the source. We have existing ways of converting expressions
    to values, such as `let template` or a `musteval` function, in the rare case
    that conversion is needed.
-   **`alias A = ref R;` instead of `alias ref A = R;`**: putting `ref` on the
    right-hand side would mirror `ref` on call arguments. We chose
    `alias ref A = R;` instead because `alias` acts as an untyped binding
    analogous to `let ref A: T = R;`.
-   **Rejecting all reference expressions without `ref`, even if
    compile-time-readable**: Under this stricter variant, `alias A = B;` would
    forbid any reference expression on the right-hand side rather than
    converting `B` to a value. Instead, we chose to convert `B` to a value
    because we wanted to support initializing expressions as long as the result
    of the conversion is known at compile time. Applying the same rule to
    reference expressions also supports those whose values are known at compile
    time; we want to treat them like compile-time constants and not require a
    `ref` marker since they won't be mutated. The actual way things like global
    constant arrays are declared is outside the scope of this proposal, but we
    imagined they might use `let` instead of `var` to reflect that they are
    immutable. For mutable variables (declared with `var`), converting to a
    value does not produce a compile-time constant, so omitting `ref` is still
    an error rather than silently changing semantics.

This was discussed on
[2026-04-02](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?tab=t.h6lecvcwu214#heading=h.vvcml6bfo69q)
and
[2026-11-18](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?tab=t.uvyw2c4zrz9e#heading=h.edvediz60r3d).
