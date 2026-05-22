# Matching redeclarations

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/3763)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
    -   [#1132 rule](#1132-rule)
    -   [C++ rule](#c-rule)
    -   [One-definition rule](#one-definition-rule)
    -   [Where declarations can appear](#where-declarations-can-appear)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Syntactic vs semantic matching](#syntactic-vs-semantic-matching)
        -   [Unqualified name lookup](#unqualified-name-lookup)
        -   [`impl`s that have not yet been declared](#impls-that-have-not-yet-been-declared)
    -   [Declaration modifiers](#declaration-modifiers)
    -   [`_` parameter names and `unused` modifier](#_-parameter-names-and-unused-modifier)
    -   [Scope differences](#scope-differences)
    -   [`impl` declarations](#impl-declarations)
        -   [Redeclarations](#redeclarations)
        -   [Differences between `impl` declarations](#differences-between-impl-declarations)
        -   [Out-of-line definitions of associated functions](#out-of-line-definitions-of-associated-functions)
        -   [`impl` members vs `interface` members](#impl-members-vs-interface-members)
    -   [`virtual` functions](#virtual-functions)
    -   [`extern` declarations](#extern-declarations)
    -   [`let` and `var` declarations](#let-and-var-declarations)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Use a partially or fully semantic rule](#use-a-partially-or-fully-semantic-rule)
    -   [Use package-wide name poisoning](#use-package-wide-name-poisoning)
    -   [Allow shadowing in implementation file after use in API file](#allow-shadowing-in-implementation-file-after-use-in-api-file)

<!-- tocstop -->

## Abstract

Require exact syntactic matching in redeclarations. Provide new terminology for
redeclaration matching and agreement. Specify non-redeclaration rules for the
other contexts where we require multiple declarations to match, such as `impl`s
of `interfaces`, `impl`s of `virtual fn`s.

## Problem

When we see two declarations that might declare the same entity, we need to
know:

-   Do they actually declare the same entity?
-   Are they similar enough to be _valid_ declarations of the same entity?

[Leads issue #1132](https://github.com/carbon-language/carbon-lang/issues/1132)
has rules for this, and those rules were partially incorporated into the design
by [#1084 Generics details 9: forward declarations](/proposals/p1084.md).
However:

-   #1084 only covers generics, not the whole scope of the language, and

-   #1132's hybrid approach of using a syntactic rule but allowing syntactic
    deviations for aliases has proven challenging and awkward to implement in
    the Carbon toolchain.

## Background

Related issues and proposals:

-   [Issue #472: Open question: Calling functions defined later in the same file](https://github.com/carbon-language/carbon-lang/issues/472)
-   [Issue #1132: How do we match forward declarations with their definitions?](https://github.com/carbon-language/carbon-lang/issues/1132)
-   [Proposal #875: Principle: information accumulation](https://github.com/carbon-language/carbon-lang/pull/875)
-   [Proposal #1084: Generics details 9: forward declarations](https://github.com/carbon-language/carbon-lang/pull/1084)
-   [Proposal #3762: Merging forward declarations](https://github.com/carbon-language/carbon-lang/pull/3762)

### #1132 rule

Under issue #1132, as fleshed out by proposal #1084:

Two declarations declare the same entity if they _match_, which in most cases
simply means that they have the same name and same scope. For `impl`
declarations, which don't have names, there is a more complicated rule.

Two declarations that declare the same entity _agree_ if they are similar enough
to be valid redeclarations of that entity. This requires them to have the same
syntactic form, with some exceptions:

1.  Parameter names can be replaced by `_` if the parameter is not used in the
    declaration.
2.  Grouping parentheses can be present in one declaration and absent in another
    if it doesn't change the parse result.
3.  An alias can be used in one declaration, where the alias target or a
    different alias to the same target is used in the other declaration.

[Concerns have been raised](https://discord.com/channels/655572317891461132/963846118964350976/1214372759224983562)
that the terminology in use here is problematic:

-   Declarations are said to "match" even when they are quite dissimilar -- for
    example, functions with different signatures are said to match.
-   The term "match" is easy to confuse with pattern-matching terminology.
-   The term "agree" sounds like it is expressing an opinion.
-   These terms don't fit well into diagnostics. For example, we considered
    diagnostic text similar to "Declaration X does not agree with matching
    declaration Y", which doesn't clearly describe the problem. While it's not
    necessary for us to use the formal, technical terminology in diagnostics, it
    would be useful to have terminology that also works in that context.

The chosen rule has also been found to be problematic to implement in practice,
because it combines the purely-syntactic concern of whether the same syntax is
used with the semantic, scope-based concern of the name lookup result, at least
for aliases.

### C++ rule

In C++, two function or function template declarations declare the same entity
if they have the same:

-   scope
-   name
-   template parameter lists
-   return type, if a function template
-   parameter types and trailing ellipsis

In each case, the component of the declaration can be written in different ways
between declaration and definition. In general, names of parameters can be
arbitrarily different. Types can be written in different ways, so long as they
resolve to the same thing. However, parts of the declaration that depend on
template parameters must be written with the same token sequence, which must be
interpreted in the same way, or the dragons of "ill-formed, no diagnostic
required" may burn your program to the ground.

There is also a grey area between cases where token-by-token matching is
required and more liberal matching is permitted, where it's not clear what rule
to apply. The oldest still-open core language issue against C++,
[CWG issue 2](https://www.open-std.org/jtc1/sc22/wg21/docs/cwg_active.html#2),
concerns such a case, but there are many more examples.

In C++, because the scope of a declaration is only entered part-way through
declaring it -- specifically, after the return type -- such problems are hard to
avoid. There are constructs that must be written differently in an in-class
declaration versus in an out-of-line definition.

```c++
struct A {
  struct B {};
  B f();
};
// Can't write this as `B f()`.
A::B A::f() {}
```

In Carbon, there is no such problem, so a simpler rule is available to us.
Indeed, it's desirable to support a simple syntactic rule for transforming an
in-class declaration into an out-of-class definition, both for tooling and for
programmers.

### One-definition rule

Proposal #3762 establishes the following rules:

If an entity is declared in the `api` file of a library, it can be redeclared in
`impl` files subject to other rules on redundant redeclarations, and a
definition is required in exactly one `impl` file.

If the entity is _not_ declared in the `api` file, and is instead introduced as
non-`extern` in an `impl` file, then it must be defined in that `impl` file.
Note that this means that such an entity can only be declared in a single `impl`
file; if an entity is intended to be shared by multiple `impl` files, it must be
declared in the `api` file, typically as a `private` member of the library.

### Where declarations can appear

Because declarations always declare their entity within the lexically-enclosing
package, and the package forms part of the scope, declarations for an entity can
only appear in a single package.

As specified in #3762, entities can be declared in more than one library: Each
entity has an owning library, and all declarations of that entity outside the
owning library are declared `extern`. All declarations of an entity without the
`extern` keyword are required to appear in the same library.

## Proposal

Replace the terminology ("match" and "agree") and corresponding rules here as
follows:

-   Two named declarations _declare the same entity_ if they have the same scope
    and the same name.
-   One declaration _redeclares_ another if they declare the same entity and the
    second declaration appears after the first, including the case where the
    first declaration is imported. In this case, the second declaration is said
    to be a _redeclaration_ of the first.
-   Two declarations _differ_ if the sequence of tokens in the declaration
    following the introducer keyword and the optional scope, up to the semicolon
    or open brace, is different, except for `unused` modifiers on parameters.
-   The program is invalid if it contains two declarations of the same entity
    that differ.

```carbon
class A {
  fn F(n: i32);
  fn G(n: i32);
}

// ❌ Error, parameter name differs.
fn A.F(m: i32) {}

// ❌ Error, parameter type differs syntactically.
fn A.G(n: (i32)) {}
```

## Details

### Syntactic vs semantic matching

Redeclarations are primarily checked syntactically. The intention is that
whenever the syntax matches, the semantics must also match. This is largely
true, and is supported by the
[information accumulation principle](https://github.com/carbon-language/carbon-lang/pull/875)
that it is an error if information learned later would have affected the meaning
of earlier code.

Two specific cases are worth calling out here:

-   [Unqualified name lookup](#unqualified-name-lookup)
-   [`impl`s that have not yet been declared](#impls-that-have-not-yet-been-declared)

Because the semantics of declarations should always match if the syntax matches,
we expect implementations of Carbon to be able to call out semantic differences
between redeclarations -- for example, "type of parameter `x` in redeclaration
is different from type in previous declaration" -- not only point to where the
syntax diverged.

All non-`extern` declarations of an entity are in the same library, and all
non-`extern` declarations see the first non-`extern` declaration of the entity,
which is either in the same file or, for a declaration in an implementation
file, in the API file, so we can always perform a precise syntactic
redeclaration check if we persist syntactic information from the API file to
implementation files.

#### Unqualified name lookup

```carbon
namespace N;
class C;
fn N.F(x: C);
class N.C;
fn N.F(x: C) {}
```

Here, it appears that the unqualified reference to `C` resolves to two different
classes in the two declarations. However, following the information accumulation
principle, we consider the declaration of `N.C` invalid because it would change
the meaning of the unqualified name `C` that has already been used.

Specifically, we add the following rule:

-   In a declarative scope, it is an error if a name is first looked up and not
    found, and later introduced.

Names for which unqualified lookup is performed and fails are said to be
_poisoned_ in that scope.

_Declarative scopes_ here are scopes like namespaces, classes, and interfaces.
Another way of looking at this rule is that reordering the declarations in a
declarative scope should never result in a program that is still valid but has a
different meaning.

It is also possible to declare an entity in a non-declarative scope, such as in
a function body. We call non-declarative scopes _sequential scopes_, and use a
different rule:

-   It is an error to redeclare an entity in a sequential scope.

This rule is provisional: no alternative has been suggested, and we don't know
whether there is strong motivation for supporting redeclaration in sequential
scopes. In C++, such redeclarations are permitted, but we are not aware of any
instance of this functionality being used. There are likely other good options
to use here, but we consider them outside the scope of this proposal.

In order to ensure that two declarations with the same syntax have the same
semantics when one declaration is in an API file and a redeclaration is in an
implementation file of the same library, we start each implementation file with
the name lookup state from the end of its API file. In particular:

-   All names declared in the API file are visible in the implementation file,
    even if they're `private`.
-   All imports in the API file are visible in the implementation file, even if
    they're not `export`ed.
-   Poisoned names are persisted from the API file to the implementation file:
    if a name is looked up by unqualified name lookup in an API file, but not
    found in that scope, the name cannot be declared in that scope in an
    implementation file.

Note that the above only apply within the same library. It is permitted for a
name to be poisoned in one library but declared in another.

Some [additional considerations](#extern-declarations) apply to `extern`
declarations, where it is possible for name lookups in two syntactically
identical declarations to have different results.

#### `impl`s that have not yet been declared

It is possible for a declaration of an entity to have one meaning because `impl`
lookup finds a broad `impl`, and for a redeclaration to have a second meaning
because `impl` lookup finds a narrower `impl`. However, in such a case, the
declaration of the narrower `impl` declaration is invalid, because it would
change the meaning of an earlier `impl` lookup. From #875:

> When an `impl` needs to be resolved, only those `impl` declarations that have
> that appear earlier are considered. However, if a later `impl` declaration
> would change the result of any earlier `impl` lookup, the program is invalid.

### Declaration modifiers

Declaration modifier keywords appear prior to the introducer keyword and scope
in a declaration, so are not involved in checking whether two declarations
differ. There may be other rules that determine whether modifiers may or must be
repeated on a redeclaration, but they are out of scope for this proposal.

```carbon
class A {
  virtual fn F[self: Self]();
}

// ✅︎ OK, `virtual` not required to match.
fn A.F[self: Self]() {}
```

### `_` parameter names and `unused` modifier

Issue #1132 permitted redeclarations of a function to differ by having one
declaration of a function name a parameter and the other leave it unnamed with
`_`. That flexibility is removed by this proposal.

This change improves the consistency of redeclarations, and also closes a hole
in the syntactic matching rule:

```carbon
fn F(T:! type, x: T);

alias T = i32;

// ❌ Not equivalent to previous declaration, but same syntax
// other than replacing parameter name with `_`.
fn F(_:! type, x: T);
```

However, it is important that a function definition can choose to not use its
parameter, in a way that is not visible in the API. Therefore we exclude
`unused` modifiers from the syntactic difference check.

```carbon
// Now equivalent to the first declaration above.
// However, this is invalid because `T` is declared unused
// but is used in the type of `x`.
fn F(unused T:! type, x: T):
```

For simplicity, and to avoid any need to check `unused` annotations match
between declaration and definition, we disallow `unused` from appearing on a
parameter in a non-defining declaration.

More generally, annotations on a declaration that are only relevant to the
definition should be excluded from the check. At the moment, the `unused`
annotation is the only such case, but we anticipate more cases emerging in the
future. Attributes on function parameters are another example that may need
special-casing.

### Scope differences

In addition to comparing whether the name and parameter portion of a declaration
differ from that of another declaration, we also need rules governing the scope
portion. The rule we use for a qualified declaration is:

-   Take the portion of the declaration from the introducer up to the end of the
    scope.
-   Replace the introducer keyword with the introducer keyword of the scope.
-   Replace the trailing `.` with a `;`.
-   The result must be a valid declaration of the scope, ignoring restrictions
    on how often the scope can be redeclared.

Put another way: each portion of the qualified name must not differ from the
declaration of the corresponding entity.

For example:

```carbon
namespace N;

class N.C(T:! type) {
  class D(U:! type) {
    fn F(a: T, b: U);
  }
}

fn N.C(T:! type).D(U:! type).F(a: T, b: U) {}
```

In this function definition:

-   `F(a: T, b: U)` does not differ from the declaration of `F`.
-   `class N.C(T:! type).D(U:! type);` would be a valid redeclaration of `D`,
    because:
    -   `D(U:! type)` does not differ from the declaration of `D`.
    -   `class N.C(T:! type);` would be a valid redeclaration of `C`, because:
        -   `C(T:! type)` does not differ from the declaration of `C`.
        -   `namespace N;` would be a valid redeclaration of `N`.

So this is a valid definition of `F`.

Note that this means that, for example, all members of a class must use the same
name for each generic parameter of that class. It cannot be `T` in one
out-of-line member definition and `ElementType` in another, or the scope in the
out-of-line definition would not match.

> **Future work:** This rule does not permit aliases to be used for top-level
> names in a declaration. This may be reasonable in most cases, because all
> declarations other than perhaps an `extern` declaration will be in the same
> library. However, names within namespaces may be declared anywhere within a
> package, and a mechanism to permit renaming of a namespace without making an
> atomic change to the entire package would be useful.

### `impl` declarations

#### Redeclarations

The rules from #1084 for `impl` declarations are largely unchanged in this
proposal, except that the allowance for `alias`es to be expanded is removed.

We cannot use the declaration name to determine whether two `impl` declarations
declare the same entity. Instead, two `impl` declarations declare the same
entity if the portion of the declaration from the introducer keyword until the
`;` or `{` does not differ, except:

-   If the constraint type in the `impl` is of the syntactic form
    `expression where constraints`, then the `constraints` portion is not
    considered.
-   `impl as` is rewritten to `impl Self as` before looking for a previous
    declaration and checking for differences from any previous declaration that
    is found.

```
interface I {
  let T:! type;
}

class A {
  impl as I where .T = ();
  // ✅︎ Redeclaration of previous declaration.
  impl Self as I where _ {}
}
```

> **Note:** These rules assume that `impl` declarations have an associated
> scope, and can only be redeclared in that scope, like all other declarations.
> This was
> [the consensus in discussion on 2024-03-11](https://docs.google.com/document/d/1s3mMCupmuSpWOFJGnvjoElcBIe2aoaysTIdyczvKX84/edit?resourcekey=0-G095Wc3sR6pW1hLJbGgE0g&tab=t.0#heading=h.p69b78lovqb7)
> but has not yet been incorporated into a design proposal. This will be the
> subject of a separate proposal.
>
> In the same discussion, the consensus was to permit such an `impl` to be
> redeclared out-of-line, with parentheses added around the name of the `impl`.
> That would lead to the following behavior under this proposal:
>
> ```carbon
> // ✅︎ Redeclaration of the `impl` from the previous example.
> impl A.(Self as I) where _;
>
> // ✅︎ Rewritten to `impl A.(Self as I) where _;` before redeclaration check.
> // Same as previous example.
> impl A.(as I) where _;
> ```
>
> However, this syntax change for `impl`s is not part of this proposal.

#### Differences between `impl` declarations

As specified in #1084, an `impl` declaration can be of the form

> `impl [...] as interface where _;`

with an `_` replacing the constraints in the `where` expression. For such a
declaration, a prior declaration must be found or the `impl` is invalid. The `_`
is replaced by the constraints in the prior declaration.

After this transformation, the normal rule is applied: the `impl` declaration
cannot differ from the previous declaration. For an implementation that performs
a semantic replacement of `_` rather than a syntactic one, it can terminate the
comparison when it reaches the `_`.

#### Out-of-line definitions of associated functions

A small change is made to the rule for [scopes](#scope-differences) for `impl`
members: parentheses are added around the corresponding portion of the scope.
For example:

```carbon
impl Type as Interface {
  fn F();
}
// Not `fn Type as Interface.F() {}`.
fn (Type as Interface).F() {}
```

Similarly for parameterized `impl`s:

```carbon
impl forall [T:! type] T as Interface(T) {
  fn F();
}
fn (forall [T:! type] T as Interface(T)).F() {}
```

And for class-scope `impl` members:

```carbon
class Class {
  impl as Interface {
    fn F();
    fn G();
  }
}

// ✅︎ OK
fn Class.(Self as Interface).F() {}

// ✅︎ Rewritten to `Self as Interface`.
fn Class.(as Interface).G() {}
```

> **Note:** As part of associating `impl` declarations with a scope,
> [the consensus in our discussion](https://docs.google.com/document/d/1s3mMCupmuSpWOFJGnvjoElcBIe2aoaysTIdyczvKX84/edit?resourcekey=0-G095Wc3sR6pW1hLJbGgE0g&tab=t.0#heading=h.p69b78lovqb7)
> was to also change the `impl forall` syntax to:
>
> ```
> impl [T:! type](T as Interface(T));
> ```
>
> If that change is made, the scope syntax will similarly change to reflect the
> new syntax:
>
> ```
> fn [T:! type](T as Interface(T)).F() {}
> ```
>
> However, that change is out of scope for this proposal.

#### `impl` members vs `interface` members

We need rules governing how associated functions in an `impl` are permitted to
differ from the corresponding declarations in the `interface`:

```carbon
interface I {
  fn F(s: Self);
}
impl i32 as I {
  // Is this valid?
  fn F(s: i32);
}
```

It is tempting to base the rules here on the rules we use to check
redeclarations. However, this turns out to be a poor choice:

-   The redeclaration rule would syntactically couple the declaration in the
    `interface` to the definition in the `impl`. But these declarations could be
    in different libraries, or even different packages, so such coupling would
    make refactorings that change the way that code is expressed but not its
    meaning either difficult or impossible.
-   The associated function in an `impl` is expected to have different syntax
    than that in the interface in some cases. The two declarations are in
    different scopes, so will refer to the same types in different ways. And the
    declaration in the `impl` is declared with knowledge of the `Self` type and
    associated constants for the interface, which it may be reasonable to use
    directly in the declaration of the function, as in the preceding example.

Therefore, different rules are used. In the specific case where an associated
function declaration in the `impl` can be used directly to satisfy a requirement
introduced by a function declaration in the `interface`, it is used directly.
For example, this is necessary to avoid infinite recursion when implementing the
`Call` interface.

The function in the `impl` is used directly when:

-   Each parameter in the `impl` function has the same type as the parameter in
    the `interface`. This includes the `self` parameter, which must be present
    in both functions if it is present in either.
-   Each parameter in the `impl` has the same category -- either `var` or not --
    as the parameter in the `interface`.
-   The return type in the `interface` and `impl` are the same type.

> **Note:** More constraints are expected to appear here over time. The key
> property we aim to identify is whether the two functions have the same calling
> convention.

Otherwise, a synthetic function called a _thunk_ is generated:

-   The declaration of the thunk is formed by substituting the `Self` type of
    the `impl` into the declaration in the `interface`. This implicitly also
    provides values for any associated constants used in the declaration.
-   The body of the thunk calls the function in the `impl`, passing in the
    arguments to the thunk, and, if a return type is specified in the interface,
    returning the value returned by the call.

If the function in the interface does not have a return type, the program is
invalid if the function in the `impl` specifies a return type.

> **Note:** Another rule might work better here. For example, we could allow
> `impl`s to add a return type in anticipation of the `interface` later adding
> one. However, such a feature is considered out of scope for this proposal.

Implicit conversions are performed as necessary to initialize the parameters of
the function in the `impl` from the parameters of the thunk, and to initialize
the return value of the function from the result of the call.

It is an error if a thunk is needed to wrap a function declaration with a `var`
parameter, because otherwise a copy would always be performed when initializing
the parameter.

> **Note:** Here as well another rule may work better and we should revisit in
> the future. Specifically, it might be better to allow these cases and simply
> move from the outer `var` to the inner `var` even though this still leaves two
> allocations in principle.

> **Note:** These rules do not cover the case where the function declaration in
> the `interface` or `impl` has implicit generic bindings. That case is
> considered to be out of scope for this proposal.

### `virtual` functions

For a `virtual` function, the same approach is taken as for `impl` functions: a
thunk is generated that differs from the declaration in the base class by
replacing the type of `self` with the derived class, unless the function chosen
for a class can be used directly. When a virtual function is used directly in a
base class and not overridden in the derived class, it is also used directly in
the derived class, even though its declared `self` parameter does not have a
matching type.

```carbon
base class B {
  // No thunk used.
  virtual fn F[addr self: B*]();
  virtual fn G[addr self: B*]();
}
base class C {
  extend B;
  // No thunk used: `self` has expected type `C*`.
  impl fn F[addr self: C*]();
  // Uses a thunk due to unexpected `self` type.
  impl fn G[addr self: B*]();
}
class D {
  // No thunk for `F`, because no thunk was used in `C`.
  // Uses thunk for `F`, because thunk was used in `C`.
  extend C;
}
```

Note that this supports covariant return types automatically, as well as any
other case where the return value from the derived class function can be
implicitly converted to the base class function's return type. However, an
`impl fn` doesn't introduce a new name lookup result, so the return type of a
call expression is always that of the `virtual fn`, which means this feature is
not useful.

> **Future work:** It might be useful to allow a declaration to both implement
> an existing virtual function and introduce a new one. This would allow
> introducing functions with covariant return types that work as expected. This
> could be achieved with syntax such as:
>
> ```carbon
> base class A {
>   virtual fn Clone[self: Self]() -> A*;
> }
> base class B {
>   virtual impl fn Clone[self: Self]() -> B*;
> }
> ```
>
> Here, a call to `b->Clone()` would find `B.Clone` rather than `A.Clone`, and
> so would have return type `B*`. The downside is that the vtable for `B` would
> have two `Clone` slots, for `A.Clone` and `B.Clone`, whereas a covariant
> return in C++ would only need a single vtable slot to express the same thing.

It is an error for a class with a custom value representation to declare or
implement a virtual function that passes `self` by value.

### `extern` declarations

For `extern` declarations, some additional considerations apply to the rule
disallowing redeclarations from syntactically differing:

-   This rule cannot in general be checked during compilation. For example,
    there may be no file that imports both an `extern` declaration and the
    library that owns the entity.
-   Despite being syntactically identical, `extern` declarations can have
    different meanings from the entity they redeclare due to unqualified lookup
    results differing.
-   A constraint that the declarations be syntactically identical is a greater
    burden, because it introduces syntactic coupling across libraries that may
    make refactoring harder.

We address these points as follows:

-   Checking whether `extern` declarations properly redeclare their targets
    during compilation is best-effort.
    -   `extern` declarations are expected to be fully checked at link time,
        including both syntactic and semantic checks, by emitting extra
        information into the object files.
    -   Some level of semantic checks may be important for an implementation in
        order to ensure its data structures are consistent. For example,
        constant evaluation may cross between file boundaries, and it may be
        important that types used in compile-time functions have the same
        meaning in the caller and callee in order to support such calls.
    -   Any further compile-time checking is optional and best-effort. Carbon
        implementations are encouraged to perform a reasonable set of checks
        during compilation in order to improve the quality of diagnostics and
        produce diagnostics earlier, but can defer harder cases until link time.
        For example, an implementation could choose to not track the syntactic
        form of a declaration, and perform only semantic checks during
        compilation, deferring syntactic checking until link time.
-   To support `extern` declarations, an additional constraint is imposed: the
    lookup results for names used in each declaration of an entity are required
    to be the same. This rule actually applies to all declarations of all
    entities, but it only has an effect -- and needs to be checked -- for
    `extern` declarations.
    -   It is an error if an `extern` declaration mentions any `private` entity
        that is not also `extern`. Such a declaration could never match an
        entity owned by another library.
    -   This check will in general need to be performed at link time, in
        addition to the link time syntactic checks.
-   For now, we will try this restrictive rule, even though it creates
    refactoring burden.
    -   This was discussed and tentatively agreed in the
        [open discussion on 2024-03-05](https://docs.google.com/document/d/1s3mMCupmuSpWOFJGnvjoElcBIe2aoaysTIdyczvKX84/edit?resourcekey=0-G095Wc3sR6pW1hLJbGgE0g&tab=t.0#heading=h.q9kh9bx2nmli)
    -   Creating an `extern` declaration necessarily creates additional coupling
        between libraries. Even a very permissive matching rule would not fully
        address the potential problems here.

### `let` and `var` declarations

Per
[issue #2590: syntax for declaring global variables in a namespace](https://github.com/carbon-language/carbon-lang/issues/2590),
there are two different forms for `let` and `var` declarations in declarative
scopes:

```carbon
// An optional scope, and a single name binding.
let Scope.A: Type = Value;

// An arbitrary pattern that is not a name binding.
let (A: Type1, B: Type2) = Value;
```

Note that this presentation is slightly different from #2590, which divided the
cases into qualified name bindings and arbitrary patterns, but the set of cases
is the same.

The former of these two cases is treated the same as any other declaration with
an introducer, an optional scope, a name, and a body, except that the end of the
declaration is at the `=` or `;` rather than at the `}` or `;`. If `let` and
`var` declarations can be redeclared -- which is a decision that is out of scope
for this proposal -- then these declarations follow the normal rules for
redeclarations, and the type of the variable is in the declaration portion, so
is not permitted to differ between declarations.

The latter case, with an arbitrary pattern that is not a single binding, does
not permit redeclarations.

## Rationale

Goals:

-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem)
    -   Matching of declarations by simple tools is possible without
        sophisticated semantic analysis. A simple token comparison is sufficient
        to match an out-of-line definition to a declaration.
    -   Moving a definition out of line is similarly a straightforward syntactic
        transformation.
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
    -   Function declarations that differ only in some subtle way are
        disallowed, leaving the maximum room available for future function
        overloading features.
    -   Functions in `impl`s and `interface`s are allowed to differ so long as
        conversions are available, allowing `interface`s to be changed in
        compatible ways.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   Requires writing code consistently between declarations.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   Provides a framework within which C++ virtual functions with covariant
        return types can be supported.

Principles:

-   [Information accumulation](https://github.com/carbon-language/carbon-lang/pull/875)
    -   Unqualified lookups follow the information accumulation rule: reordering
        declarations can't result in a different but valid program.

## Alternatives considered

### Use a partially or fully semantic rule

We could allow declarations to differ more arbitrarily. We would need a rule to
support comparisons of portions of declarations that are dependent on template
parameters, and such comparisons likely need to be partially or fully syntactic.

Advantages:

-   Allows the interface to be presented in a way that is suitable for a reader
    of the interface, and the implementation to be presented in a different way
    that is suitable for a reader of the implementation.
-   Additional flexibility for refactorings that are partially or incrementally
    applied.

Disadvantages:

-   A syntactic rule would still be needed sometimes, but would be applied
    inconsistently.
-   Drawing a line between the syntactic and semantic checks would be
    complicated and difficult, as evidenced by such a line not having been
    successfully drawn in C++.
-   Allowing divergence between the syntax used in redeclarations makes code
    harder to read and understand, and can permit unintended divergence, or even
    bugs such as function parameters of the same type being swapped.

### Use package-wide name poisoning

Instead of name poison only propagating from the API file to implementation
files of the same library, we could poison names across the whole package if an
unqualified lookup in a scope fails. If fully enforced, this would improve the
ability to move code between libraries, as it wouldn't be possible for
unqualified lookup results to change depending on where the code is. Full
enforcement of this rule would require a link-time check, but it could be
partially enforced at compile time, in cases where the poisoning lookup and the
declaration are both visible in some compilation step.

Whether this option might be appealing may depend on our position on name
shadowing. In addition to the option commonly used in other languages, of
allowing names in inner scopes to hide names in outer scopes, we have been
considering a more restrictive option: always look in all enclosing scopes, and
diagnose an ambiguity if an unqualified name is found in more than one enclosing
scope.

We do not yet have a proposal with a name shadowing rule. If we pick the more
restrictive option, the impact of package-wide name poisoning is also quite
restrictive: for example, a name used as a local variable in a function in one
library cannot be used as a public name in any enclosing namespace in any
library in that package. If we pick the less-restrictive rule, then package-wide
poisoning may become more palatable and should be reconsidered.

With either shadowing rule, some special accommodation for private names might
be reasonable. A use of a name in one library that results in the name being
poisoned should probably not conflict with a private declaration in another
library, because the two uses of the name cannot interact.

### Allow shadowing in implementation file after use in API file

In this proposal, we say it is an error for a name to be shadowed in an
implementation file after it is used in an API file:

```carbon
library "foo" api;
namespace N;
class A {}
fn N.F(x: A);
```

```carbon
library "foo" impl;
// ❌ Shadows `class A`. Cannot declare `N.A` after we already
// looked for it in the declaration of `N.F`.
class N.A {}
fn N.F(x: A) {}
```

We could allow such code. However, this would mean that the redeclaration check
for `N.F` would not be purely syntactic.

We considered other ways of addressing this. In particular, we considered
disallowing the declaration of `N.F` in the implementation file because it uses
a local name to declare a non-local function. However, this doesn't resolve the
problem: the class `N.A` could be imported from a different library instead of
being declared locally. Variations of this were considered, but none of them
seemed to work adequately.
