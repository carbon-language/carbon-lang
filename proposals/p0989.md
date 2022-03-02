# Member access expressions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/989)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [Separate syntax for static versus dynamic access](#separate-syntax-for-static-versus-dynamic-access)
    -   [Use a different lookup rule in templates](#use-a-different-lookup-rule-in-templates)
    -   [Meaning of `Type.Interface`](#meaning-of-typeinterface)

<!-- tocstop -->

## Problem

We need syntaxes for a number of closely-related operations:

-   Given an expression denoting a package, namespace, class, interface, or
    similar, and the name of one of its members, form an expression denoting the
    member. In C++ and Rust, this is spelled `Container::MemberName`. In many
    other languages, it is spelled `Container.MemberName`.

-   Given an expression denoting an object and a name of one of its fields, form
    an expression denoting the corresponding subobject. This is commonly written
    as `object.field`, with very little deviation across languages.

-   Given an expression denoting an object and a name of one of its methods,
    form an expression that calls the function on the object. This is commonly
    written as `object.function(args)`.

-   Given an expression denoting a type, and an expression denoting a member of
    an interface, form an expression denoting the corresponding member in the
    `impl` of that interface for that type.

Further, we need rules describing how the lookup for the member name is
performed, and how this lookup behaves in generics and in templates in cases
where the member name depends on the type or value of the first operand.

## Background

C++ and Rust distinguish between the first use case and the rest. Other
languages, such as Swift and C#, do not, and model all of these use cases as
some generalized form of member access, where the member might be a namespace
member, an interface member, an instance member, or similar.

See also:

-   [Exploration of member lookup in generic and non-generic contexts](https://docs.google.com/document/d/1-vw39x5YARpUZ0uD2xmKepLEKG7_u122CUJ67hNz3hk/edit)
-   [Question for leads: constrained template name lookup](https://github.com/carbon-language/carbon-lang/issues/949)

## Proposal

All these operations are performed using `.`:

```carbon
fn F() {
  // Can perform lookup inside the package or namespace.
  var x: Package.Namespace.Class;
  // Can perform lookup inside the type of the value.
  x.some_field = x.SomeFunction(1, 2, 3);
}
```

When the type of the left-hand operand is a generic type parameter, lookup is
performed in its type-of-type instead. Effectively, a generic type parameter
behaves as an archetype:

```carbon
interface Hashable {
  let HashValue:! Type;
  fn Hash[me: Self]() -> HashValue;
  fn HashInto[me: Self](s: HashState);
}
fn G[T:! Hashable](x: T) {
  // Can perform lookup inside the type-of-type if the type is
  // a generic type parameter.
  x.Hash();
}
```

When the type of the left-hand operand is a template parameter, the lookup is
performed both in the actual type corresponding to that template parameter and
in the archetype, as described above. If a result is found in only one lookup,
or the same result is found in both lookups, that result is used. Otherwise, the
member access is invalid.

```carbon
class Potato {
  fn Mash[me: Self]();
  fn Hash[me: Self]();
  alias HashValue = Hashable.HashValue;
}
external impl Potato as Hashable where .HashValue = u32 {
  // ...
}
fn H[template T:! Hashable](x: T, s: HashState) {
  // When called with T == Potato:
  // ❌ Ambiguous, could be `Potato.Hash` or `Hashable.Hash`.
  x.Hash();
  // ✅ OK, found only in `Potato`.
  x.Mash();
  // ✅ OK, found only in `Hashable`.
  x.HashInto(s);

  // ✅ OK, same `HashValue` found in both `Potato` and `Hashable`;
  // `Hashable.Hash` unambiguously names the interface member.
  var v: T.HashValue = x.(Hashable.Hash)();

  // ✅ OK, unambiguously names the type member.
  x.(Potato.Hash)();
}
```

## Details

See
[the changes to the design](https://github.com/carbon-language/carbon-lang/pull/989/files).

## Rationale based on Carbon's goals

-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
    -   Rejecting cases in a template where a generic interpretation and an
        interpretation with specific types would lead to different meanings
        supports incremental migration towards generics by way of a template,
        where the compiler will help you find places that would change meaning.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   Using a single, familiar `container.member` notation for all the member
        access use cases minimizes the complexity of this portion of the
        language syntax.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   The behavior of templates is aligned with that in C++, simplifying both
        comprehension for C++ developers and migration of C++ code.

## Alternatives considered

### Separate syntax for static versus dynamic access

We could follow C++ and Rust, and use `::` for static lookup, reserving `.` for
instance binding:

```
var x: Package::Namespace::Class;
Class::Function();
x.field = x.Function();
x.(Interface::Method)();
```

Advantages:

-   Visually separates operations that readers may think of as being distinct: a
    `::` path statically identifies an object whereas a `.` path dynamically
    identifies a subobject or forms a bound method.
-   Improves familiarity for those coming from C++.
-   Removes most of the need for parenthesized member access: `a.(b.c)` would
    generally become `a.b::c`, like in C++.

Disadvantages:

-   Adds a new token and a new operation.
-   Swift, C#, and Java do not distinguish these operations syntactically, and
    we have no evidence that this lack of syntactic distinction creates problems
    for them in practice.
-   Likely to result in complexity and inconsistency for operations falling
    between the two options. For example, in C++:
    ```
    struct A {
      static void F();
      enum { e };
    };
    enum class B { e };
    void G(A a, B b) {
      a.F(); // OK, but static dispatch, like A::F().
      a.e;   // OK, but static dispatch, like A::e.
      b.e;   // Error.
    }
    ```
-   Does not provide an obvious syntax for `impl` lookup.
    `Type::Interface::method` would be ambiguous and `Type.Interface::method`
    would be inconsistent with using `::` for static lookup, so we would likely
    end up with `Type::(Interface::method)` syntax or similar.
-   May create the suggestion that `.`s imply a performance-relevant operation
    and `::`s do not. This will typically not be the case, as `.`s will
    typically result in, at worst, a constant offset. However, `impl` lookup,
    which may be performed by either a `.` or a `::`, may require a memory
    access in cases where dynamic dispatch is in use.

### Use a different lookup rule in templates

See
[question for leads: constrained template name lookup](https://github.com/carbon-language/carbon-lang/issues/949)
for more in-depth discussion and leads decision.

Given a situation where the same name can be found in both a type and a
constraint when instantiating a template, and resolves to two different things,
we could use various different rules to pick the outcome:

```
class Potato {
  fn Bake[me: Self]();
  fn Hash[me: Self]();
}
interface Hashable {
  fn Hash[me: Self]() -> HashState;
  fn HashInto[me: Self](s: HashState);
}
external impl Potato as Hashable;

fn MakePotatoHash[template T:! Hashable](x: T, s: HashState) {
  x.Bake();
  x.Hash();
  x.HashInto(s);
}
```

We considered the following options:

| Option                | Type only: `x.Bake()` | Both: `x.Hash()` | Constraint only: `x.HashInto(s)` |
| --------------------- | --------------------- | ---------------- | -------------------------------- |
| Type                  | -> Type               | -> Type          | ❌ Rejected                      |
| Type over constraint  | -> Type               | -> Type          | -> Constraint                    |
| Type minus conflicts  | -> Type               | -> Type          | ❌ Rejected                      |
| Union minus conflicts | -> Type               | ❌ Rejected      | -> Constraint                    |
| Constraint over type  | -> Type               | -> Constraint    | -> Constraint                    |
| Constraint            | ❌ Rejected           | -> Constraint    | -> Constraint                    |

Of these rules:

-   "Type" and "type over constraint" mean the constraints in a constrained
    template do not guide the meaning of the program, which creates a surprising
    discontinuity when migrating from templates to generics.
-   "Type minus conflicts" does not present a valuable improvement over "union
    minus conflicts".
-   "Union minus conflicts" makes the type-only case behave like a non-template,
    and the constraint-only case behave like a generic. This means that explicit
    qualification is necessary for all qualified names in a template if it wants
    to defend against ambiguity from newly-added names, whereas all the earlier
    options require qualification only for names intended to be found in the
    constraint, and all the later options require qualification for names
    intended to be found in the type. However, most of the other rules require
    explicit qualification in the same cases to defend against names being
    _removed_.
-   "Constraint over type" means there is potential for a discontinuity in
    behavior depending on whether we're able to symbolically resolve the type or
    not: if semantic analysis can determine a type symbolically, you get the
    behavior from the constraint, and if not, you get the behavior from the
    type. This may lead to surprising and hard-to-understand program behavior.
-   "Constraint" means that a constrained template behaves essentially the same
    as a generic, which harms the ability to use constrained templates as an
    incremental, evolutionary stepping stone from non-constrained templates into
    generics.

No rule provides ideal behavior. The most significant disadvantage of the chosen
rule, "union minus conflicts", is that it requires explicit qualification with
either the type or the constraint in a fully-robust template. However, the other
leading contender, "constraint over type", also requires qualification of all
names to prevent silent changes in behavior if a constraint is changed, and
"union minus conflict" seems preferable to "constraint over type" in other ways.

### Meaning of `Type.Interface`

In this proposal, `impl` lookup is performed when a member of an interface
appears on the right of a `.`. We could also consider applying `impl` lookup
when the name of an interface appears on the right of a `.`. Under that
alternative, `Class.(Interface)` would be a name for the `impl`, that is, for
`impl Class as Interface`.

Because we have previously decided we don't want facet types, such a name would
be restricted to only appear in the same places where package and namespace
names can appear: on the left of a `.` or the right of an `alias`.

For example:

```
interface MyInterface {
  fn F();
  fn G[me: Self]();
}
class MyClass {
  alias InterfaceAlias = MyInterface;
  impl as MyInterface {
    fn F();
    fn G[me: Self]();
  }
}

fn G(x: MyClass) {
  // OK with this proposal and the alternative.
  MyClass.(MyInterface.F)();
  // Error with this proposal, OK with the alternative.
  MyClass.(MyInterface).F();

  // Names the interface with this proposal.
  // Names the `impl` with the alternative.
  alias AnotherInterfaceAlias = MyClass.InterfaceAlias;

  // Error with this proposal, OK with the alternative.
  MyClass.InterfaceAlias.F();
  // OK with this proposal, error with the alternative.
  MyClass.(MyClass.InterfaceAlias.F)();

  // Error under this proposal, OK with the alternative.
  x.MyInterface.F();
  // Error under both this proposal.
  // Also error under the alternative, unless we introduce
  // a notion of a "bound `impl`" so that `x.MyInterface`
  // remembers its receiver object.
  x.MyInterface.G();
  // OK under this proposal and the alternative.
  x.(MyInterface.G)();
}
```

Advantages:

-   Gives a way to name an `impl`.

Disadvantages:

-   It's not clear that we need a way to name an `impl`.
-   Presents a barrier to supporting member interfaces, because
    `MyClass.MemberInterface` would name the `impl MemberInterface as MyClass`,
    not the interface itself.
-   Reintroduces facet types, without the ability to use them as a type. Having
    a way of naming an `impl` may lead to confusion over whether they are
    first-class entities.
-   Would either surprisingly reject constructs like `x.MyInterface.G()` or
    require additional complexity in the form of a "bound `impl`" value. The
    value of such a type would presumably be equivalent to a facet type.

As a variant of this alternative, we could disallow `Type.Interface` for now, in
order to reserve syntactic space for a future decision. However, it's not clear
that the cost of evolution nor the likelihood of such a change is sufficiently
high to warrant including such a rule.
