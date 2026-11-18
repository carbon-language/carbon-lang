# Expression form basics

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/5545)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Mixed expression categories](#mixed-expression-categories)
    -   [Don't implicitly convert to less-primitive forms](#dont-implicitly-convert-to-less-primitive-forms)
    -   [Breadth-first evaluation order](#breadth-first-evaluation-order)
    -   [Depth-first evaluation with a different "horizontal" order](#depth-first-evaluation-with-a-different-horizontal-order)
    -   [Support binding `ref self` to ephemeral references](#support-binding-ref-self-to-ephemeral-references)

<!-- tocstop -->

## Abstract

This proposal introduces the concept of a _form_, which is a generalization of
"type" that encompasses all of the information about an expression that's
visible to the type system, including type and expression category. Forms can be
composed into _tuple forms_ and _struct forms_, which lets us track the
categories of individual tuple and struct literal elements.

## Problem

It's unclear what expression category tuple and struct literals should have. For
example, this code can only compile if the tuple literal is an initializing
expression:

```carbon
var t: (NonMovable, NonMovable) = (MakeNonMovable(), MakeNonMovable())
```

But this code can only compile if the tuple literal is a value expression:

```carbon
let x: NonCopyable = MakeNonCopyable();
let t: (NonCopyable, NonCopyable) = (x, MakeNonCopyable());
```

And there's plausible code that can't compile if the tuple literal has _any_
single expression category:

```carbon
let x: NonCopyable = MakeNonCopyable();
let (a: NonCopyable, var b: NonMovable) = (x, MakeNonMovable());
```

At present it's always possible to rewrite examples like that to avoid the
problem by disaggregating the tuple patterns into separate statements. However,
when the copy and move operations in question are expensive rather than outright
disabled, those examples will result in silent inefficiency rather than a noisy
build failure, which is less harmful but easier to overlook.

## Background

The Carbon toolchain already implements a solution to this problem: it treats
tuple and struct literals as having a "mixed" expression category, and when
individual elements of the literal are accessed (such as during pattern
matching), the element's original category is propagated.

Proposal [#5434](https://github.com/carbon-language/carbon-lang/pull/5434)
introduces plausible use cases that cannot compile if we assign any single
expression category to a tuple or struct literal, and there is no way to avoid
the problem by rewriting. For example:

```carbon
fn F() -> (ref NonCopyable, NonMovable);
let (a: NonCopyable, var b: NonMovable) = F();
```

## Proposal

This proposal solves that problem by introducing the concept of a _form_, which
is a generalization of "type" that encompasses all of the information about an
expression that's visible to the type system, including type and expression
category. In the common case, an expression has a _primitive form_ which
consists of a type, an expression category, and a few other properties. However,
a tuple literal has a _tuple form_, which is a tuple of the forms of its
elements. This allows us to directly represent the fact that different elements
have different categories, and propagate that difference into operations that
access those elements.

In order to help describe the semantics of forms, this proposal also formalizes
the concept of the _result_ of an expression evaluation. Results are a
generalization of values and references in the same way that forms are a
generalization of types. In this proposal they are primarily a descriptive
convenience, but they are also intended to function as the thing that a
form-generic binding binds to, when that is proposed.

The results of initializing expressions, called _initializing results_, have
somewhat subtle semantics. Results present an idealized model of expression
evaluation where information flows from each expression to the context where it
is used, but initializing expressions require information to flow in both
directions: the context supplies a storage location, and then the expression
supplies the contents of that storage location. We finesse this "impedance
mismatch" by saying that the initializing result represents an obligation on the
context to supply a storage location (somewhat like a callback or
`std::promise`), which it must fulfill by either materializing or transferring
the result. Furthermore, even though this formally happens after the expression
is evaluated, it is constrained in such a way that it can actually be computed
beforehand and passed to the expression's hidden output parameter.

This proposal also integrates type, category, and phase conversions into a
unified set of rules for form conversions. Notably, those rules call for
conversions to be evaluated depth-first (along with the expressions they convert
from and the pattern-matches they feed into), and for struct conversions to use
the field order of the source, not the target.

Finally, this proposal splits the reference expression categories into _entire_
and _non-entire_ references, where an entire reference is known to refer to a
complete object. This lets us decouple materialization (which now produces an
ephemeral entire reference) from `var` binding (which now expects an ephemeral
entire reference), which lets us resolve a TODO to allow an initializing
expression to be destructured into multiple `var` bindings.

In the process of doing this, it became clear that the special case that allowed
`ref self` patterns to match ephemeral references was not internally consistent,
so that special case has been removed. We will need some way of supporting the
use cases that were intended to be covered by that rule, but that is being left
as future work.

## Details

See the edits in the
[pull request](https://github.com/carbon-language/carbon-lang/pull/5545)
associated with this proposal, particularly in `values.md`.

## Rationale

This proposal supports
[performance-critical software](/docs/project/goals.md#performance-critical-software)
and making
[code easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
by ensuring that tuple and struct literals don't introduce unnecessary category
conversions (which may cause build failures and performance overhead).

## Alternatives considered

### Mixed expression categories

This proposal models an expression like `(x, MakeNonMovable())` from the earlier
example as having a tuple form consisting of primitive forms with `NonCopyable`
and `NonMovable` types (respectively) and "value" and "initializing" categories
(respectively). We could instead support composite expression categories, so
that it has type `(NonCopyable, NonMovable)` and expression category
`(value, initializing)`.

This would avoid the need to introduce the concept of "form", and preserve the
existing separation between types and categories. That separation would be
somewhat superficial (for example, an expression couldn't have a tuple
expression category if it doesn't have a tuple type), but no more so than the
separation between types and values.

However, we expect to need an explicit syntax to express these properties of
expressions, for example to define functions that return tuples whose elements
have different categories. A syntax consisting of separate type and category
tuples will be much less ergonomic, and much easier to misuse, than a syntax
that combines both in a single tuple (for example,
[#5434](https://github.com/carbon-language/carbon-lang/pull/5434) represents the
form of `(x, MakeNonMovable())` as `(val NonCopyable, NonMovable)`).

Furthermore, we anticipate needing to support code that is generic with respect
to forms, not just with respect to types. We plan to achieve that with
parameters of a special "form" type together with ways of deducing and using
them. It might be possible to instead support category parameters that are
deduced and used in conjunction with type parameters, but that would be
syntactically onerous, and oblige users to keep each category correctly paired
with the corresponding type, in order to bring them together at the point of
use. Those challenges will be further compounded when/if it becomes possible to
manipulate types and categories by way of metaprogramming.

Given that we need to present forms as an integrated whole at the syntactic and
metaprogramming levels, there is very little to be gained by decoupling them at
the level of language semantics.

### Don't implicitly convert to less-primitive forms

Consider the following possible ways of initializing an array, where there is an
implicit conversion from integer literals to `X`:

```carbon
impl Y as Core.ImplicitAs(X);
let a: array(X, 3) = (MakeY(), MakeY(), MakeY());

let x_tuple: (X, X, X) = (MakeY(), MakeY(), MakeY());
let b: array(X, 3) = x_tuple;
```

Under this proposal, both `a` and `b` are valid. However, some people's
intuition is that a tuple is different enough from an array that there should
not be implicit conversions between them in general. In this mental model, a
tuple-form expression wouldn't represent a tuple per se; rather, it would
abstractly represent a sequence of results, which can be used to initialize
either a tuple or an array (or a user-defined type). Similarly, a struct-form
expression wouldn't represent a struct per se, but rather a more abstract
sequence of _named_ results. Consequently, there would be no implicit conversion
from a tuple value to a tuple form, and so `array` could disallow `b` by
requiring its initializer to have a tuple form.

Perhaps more importantly, making this distinction could simplify C++ interop,
because we could map C++ braced initializer lists to tuple and struct forms,
while mapping C++ `std::pair`s and `std::tuple`s to primitive-form Carbon
tuples, and mapping C++ structs to primitive-form Carbon structs. We cannot do
that under the status quo, because Carbon's implicit conversions from primitive
to composite forms would have no C++ counterpart, and so overload resolution
behavior would change too radically when crossing the language boundary.

However, consider the following examples:

```carbon
let c: array(X, 3) = (MakeY(), MakeY(), MakeY()) as (X, X, X);

let y_tuple: (Y, Y, Y) = (MakeY(), MakeY(), MakeY());
let d: array(X, 3) = y_tuple as (X, X, X);
```

Presumably we would want the declaration of `c` to be valid, because it just
makes the implicit type conversion from `a` explicit. That result emerges pretty
naturally, because the `as` conversion operates element-wise on its tuple-form
input, so its output would likewise have a tuple form. On the other hand, the
declaration of `d` should not be valid, because it just makes the implicit type
conversion from `b` explicit. That result does not emerge naturally; instead, we
have to add a final form composition step at the end of the type conversion, to
ensure the result has a primitive form (but only when the input had a primitive
form).

But that means we need to choose the category of that primitive form, that is
whether the conversion from `(Y, Y, Y)` to `(X, X, X)` is a value expression or
an initializing expression (form composition can't produce reference
expressions). Upcoming proposals are expected to enable the user to define the
implicit conversion from `Y` to `X` to have any category, so suppose that the
conversion from `Y` to `X` is a reference expression. In that case, the
requirement to convert to a primitive form breaks some use cases that would
otherwise be valid:

```carbon
let (ref s1: X, ref s2: X, ref r3: X) = y_tuple as (X, X, X);
```

Even for usages that are not broken outright, this conversion may add
substantial inefficiency, depending on which category we convert to:

```carbon
let (p1: X, p2: X, p3: X) = y_tuple as (X, X, X);
```

This requires 3 value acquisitions in either case, but if we convert to an
initializing expression it also requires 3 copy-init and materialization steps.

```carbon
let (var q1: X, var q2: X, var q3: X) = y_tuple as (X, X, X);
```

In either case we're starting and ending with the object representation of `X`,
passing through the initializing representation (which is typically very closely
tied to the object representation), but if the type conversion produces a value
expression we also pass through the value representation (which can be much less
trivial to convert to and from).

Finally, examples like this one are less efficient if we have to convert to a
primitive form, regardless of which category we convert to:

```carbon
let (r1: X, var r2: X, var r3: X) = y_tuple as (X, X, X);
```

Some but not all of these problems can be mitigated by making the target
category an input to the type conversion, but that has a number of unwelcome
consequences:

-   It adds complexity to the conversion APIs that user-defined types will
    almost never need.
-   We would need to change the syntax for `x as T` so that it specifies the
    category as well as the type.
-   It would enable user-defined conversions to produce different values
    depending on the target category, which we definitely don't want.
-   It makes overload resolution more complicated and more surprising.

This alternative was considered and rejected in
[leads issue #6160](https://github.com/carbon-language/carbon-lang/issues/6160).

### Breadth-first evaluation order

Under this proposal, the observable effects of a pattern-matching operation take
place in depth-first order (with respect to tuple and struct elements), but
breadth-first order would have several advantages:

-   It would help enable us to define tuple type conversions as ordinary impls
    of the type conversion APIs, because modeling conversion as a function call
    is inherently breadth-first. Opting for depth-first evaluation seems to rule
    that out.
-   It would give us more options for supporting struct and class conversions
    where the field orders doesn't match. For example, the previously-preferred
    approach was to evaluate the source expression in its own field order, but
    then perform conversions in the target's field order. This seemed to provide
    a good balance of efficiency and ergonomics, but it's inherently
    breadth-first. With the depth-first approach, we have to use a single field
    order.
-   It would simplify the specification, because the logical semantics of
    conversion and pattern matching are much more naturally described
    breadth-first, so opting for depth-first creates an "impedance mismatch"
    between the logical and physical semantics.

However, breadth-first evaluation order has a crucial efficiency cost: in all
but the most trivial use cases, it forces (and in some sense even maximizes)
lifetime overlap between the results of the function calls that make up the
pattern-matching operation. That means it requires more temporary storage, and
more work to manage that storage (particularly at the register level).

For example, consider this
[generated code](https://cpp.compiler-explorer.com/z/Pxaar4edb) for C++
approximating the two options. The `LayerWise` function templates model the
breadth-first evaluation order, while the `ElementWise` model the proposed
depth-first order. Looking at the two element case for ARM:

```asm
void LayerWise<S1, S1>(S1, S1):
        stp     x29, x30, [sp, #-32]!
        stp     x20, x19, [sp, #16]
        mov     x29, sp
        mov     x19, x1
        bl      Convert1(S1)
        mov     x20, x0
        mov     x0, x19
        bl      Convert1(S1)
        mov     x19, x0
        mov     x0, x20
        bl      Convert2(S2)
        mov     x20, x0
        mov     x0, x19
        bl      Convert2(S2)
        mov     x1, x0
        mov     x0, x20
        ldp     x20, x19, [sp, #16]
        ldp     x29, x30, [sp], #32
        b       void Target<S1, S1>(S1, S1)

void ElementWise<S1, S1>(S1, S1):
        stp     x29, x30, [sp, #-32]!
        stp     x20, x19, [sp, #16]
        mov     x29, sp
        mov     x19, x1
        bl      Convert1(S1)
        bl      Convert2(S2)
        mov     x20, x0
        mov     x0, x19
        bl      Convert1(S1)
        bl      Convert2(S2)
        mov     x1, x0
        mov     x0, x20
        ldp     x20, x19, [sp, #16]
        ldp     x29, x30, [sp], #32
        b       void Target<S1, S1>(S1, S1)
```

The breadth-first approach forces significantly more data movement (the extra
moves between `x20` and `x19`) and temporary registers. This distinction
continues in versions with more elements, making the problem more and more
severe. This is an inherent cost, and not something we can realistically expect
an optimizing compiler to recover.

Furthermore, we don't see any general way for developers to work around this
problem, and achieve the performance they would get automatically with
depth-first evaluation. On the other hand, with depth-first evaluation within
pattern-matching operations, developers can still ensure a breadth-first
evaluation order relatively easily, by expressing each "layer" as a separate
pattern-matching operation (for example with a sequence of statements, or a
chain of function calls). In some cases, this may involve move operations that
could be elided in a language-native breadth-first evaluation, but that depends
on the to-be-determined design of move semantics.

In short, the ergonomic costs of depth-first evaluation appear to be manageable,
whereas the performance cost of breadth-first evaluation is unavoidable (and
carries more weight, because supporting
[performance-critical software](/docs/project/goals.md#performance-critical-software)
is our top goal for the language).

This alternative was considered and rejected in
[leads issue #6456](https://github.com/carbon-language/carbon-lang/issues/6456).

### Depth-first evaluation with a different "horizontal" order

To order operations that don't have a dependency relationship, this proposal
uses a hybrid scheme that follows both the lexical order of the primitive
patterns and the lexical order of the scrutinee function calls (diagnosing an
error if they conflict), and also follows the lexical order of the scrutinee's
declared type to the extent that it doesn't conflict with the primitive pattern
order.

One potential drawback of this rule is it means that it means that, unlike C++,
Carbon doesn't guarantee that the fields of an object are initialized in
declaration order (or in any fixed order), and so it can't guarantee that
they'll be destroyed in reverse order of initialization.

We could solve that problem by guaranteeing to evaluate in the scrutinee type's
field order. However, that would mean evaluating function calls out of lexical
order in cases like this:

```carbon
var ab: {.a: A, .b: B} = {.b = MakeB(), .a = MakeA()};
```

Evaluating `MakeA()` before `MakeB()` risks causing surprises, or even bugs, and
it would mean that the evaluation order within a single expression depends on
how it's used.

Instead, we adopt a rule that ensures side effects are evaluated in lexical
order within both the pattern and the scrutinee. We expect that to be sufficient
for struct types, where destruction order is unlikely to be an issue. Class
types may be more sensitive to these ordering issues, so we may also need some
way for the class to disallow initializers whose field order doesn't match the
class, but this is left as future work.

### Support binding `ref self` to ephemeral references

`ref` patterns can only match durable reference expressions, but prior to this
proposal, `ref self` patterns could match ephemeral references as a special-case
exception. This was intended to support certain C++ idioms that rely on
materializing a temporary and then mutating it in place, such as fluent
builders. For example:

```carbon
class FooBuilder {
   // These methods mutate `self` and then return a reference to it.
   fn SetBar[ref self: Self]() -> ref Self;
   fn SetBaz[ref self: Self]() -> ref Self;

   fn Build[ref self: Self]() -> Foo;
}
fn MakeFoo() -> FooBuilder;

let foo: Foo = MakeFoo().SetBar().SetBaz().Build();
```

Prior to this proposal, this code would be valid: `MakeFoo()` is an initializing
expression, but when it is matched with `ref self: Self` as part of the `SetBar`
call, it is implicitly converted to an ephemeral reference, and then the
special-case rule allows `ref self: Self` to bind to the materialized temporary.

However, those rules also imply that code like this would be valid:

```carbon
let builder: FooBuilder = MakeFoo();
builder.SetBar();
builder.SetBaz();
let foo: Foo = builder.Build();
```

Here the programmer has accidentally used `let` instead of `var`, so `builder`
is an immutable value. But a value expression can be implicitly converted to an
initializing expression by direct initialization, and as we already saw, it's
valid to call `MakeFoo()` and `MakeBar()` on an initializing expression. So this
code repeatedly materializes, mutates, and then discards a copy of `builder`,
and then ultimately initializes `foo` with the state returned by `MakeFoo()`,
which is surely not what the programmer intended. This sort of "lost mutation"
bug is exactly what the distinction between durable and ephemeral references was
intended to prevent, but the `ref self` special case combines with the
transitivity of category conversions to defeat that protection.

A proper resolution of this issue seems beyond the scope of this proposal, so
this proposal removes that special case without replacement, leaving the problem
of supporting idioms like fluent builders as future work.
