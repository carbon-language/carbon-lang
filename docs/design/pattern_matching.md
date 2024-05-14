# Pattern matching

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Pattern Syntax and Semantics](#pattern-syntax-and-semantics)
    -   [Expression patterns](#expression-patterns)
        -   [Alternatives considered](#alternatives-considered)
    -   [Binding patterns](#binding-patterns)
        -   [Name binding patterns](#name-binding-patterns)
        -   [Unused bindings](#unused-bindings)
            -   [Alternatives considered](#alternatives-considered-1)
        -   [Compile-time bindings](#compile-time-bindings)
        -   [`auto` and type deduction](#auto-and-type-deduction)
        -   [Alternatives considered](#alternatives-considered-2)
    -   [`var`](#var)
    -   [Tuple patterns](#tuple-patterns)
    -   [Struct patterns](#struct-patterns)
        -   [Alternatives considered](#alternatives-considered-3)
    -   [Alternative patterns](#alternative-patterns)
    -   [Templates](#templates)
    -   [Refutability, overlap, usefulness, and exhaustiveness](#refutability-overlap-usefulness-and-exhaustiveness)
        -   [Alternatives considered](#alternatives-considered-4)
-   [Pattern usage](#pattern-usage)
    -   [Pattern match control flow](#pattern-match-control-flow)
        -   [Guards](#guards)
    -   [Pattern matching in local variables](#pattern-matching-in-local-variables)
-   [Open questions](#open-questions)
    -   [Slice or array nested value pattern matching](#slice-or-array-nested-value-pattern-matching)
    -   [Pattern matching as function overload resolution](#pattern-matching-as-function-overload-resolution)
-   [Alternatives considered](#alternatives-considered-5)
-   [References](#references)

<!-- tocstop -->

## Overview

A _pattern_ is an expression-like syntax that describes the structure of some
value. The pattern may contain unknowns, so it can potentially match multiple
values, and those unknowns may have names, in which case they are called
_binding patterns_. When a pattern is executed by giving it a value called the
_scrutinee_, it determines whether the scrutinee matches the pattern, and if so,
determines the values of the bindings.

## Pattern Syntax and Semantics

Expressions are patterns, as described below. A pattern that is not an
expression, because it contains pattern-specific syntax such as a binding
pattern, is a _proper pattern_. Many expression forms, such as arbitrary
function calls, are not permitted as proper patterns, so cannot contain binding
patterns.

-   _pattern_ ::= _proper-pattern_

```carbon
fn F(n: i32) -> i32 { return n; }

match (F(42)) {
  // ❌ Error: binding can't appear in a function call.
  case (F(n: i32)) => {}
}
```

### Expression patterns

An expression is a pattern.

-   _pattern_ ::= _expression_

The pattern is compared with the expression using the `==` operator: _pattern_
`==` _scrutinee_.

```carbon
fn F(n: i32) {
  match (n) {
    // ✅ Results in an `n == 5` comparison.
    // OK despite `n` and `5` having different types.
    case 5 => {}
  }
}
```

Any `==` operations performed by a pattern match occur in lexical order, but for
repeated matches against the same _pattern_, later comparisons may be skipped by
reusing the result from an earlier comparison:

```carbon
class ChattyIntMatcher {
  impl as EqWith(i32) {
    fn Eq[me: ChattyIntMatcher](other: i32) {
      Print("Matching {0}", other);
      return other == 1;
    }
  }
}

fn F() {
  // Prints `Matching 1` then `Matching 2`,
  // may or may not then print `Matching 1` again.
  match ((1, 2)) {
    case ({} as ChattyIntMatcher, 0) => {}
    case (1, {} as ChattyIntMatcher) => {}
    case ({} as ChattyIntMatcher, 2) => {}
  }
}
```

#### Alternatives considered

-   [Introducer syntax for expression patterns](/proposals/p2188.md#introducer-syntax-for-expression-patterns)

### Binding patterns

#### Name binding patterns

A name binding pattern is a pattern.

-   _binding-pattern_ ::= _identifier_ `:` _expression_
-   _proper-pattern_ ::= _binding-pattern_

The _identifier_ specifies the name of the _binding_. The type of the binding is
specified by the _expression_. The scrutinee is implicitly converted to that
type if necessary. The binding is then _bound_ to the converted value.

```carbon
fn F() -> i32 {
  match (5) {
    // ✅ `5` is implicitly converted to `i32`.
    case n: i32 => {
      // The binding `n` has the value `5 as i32`,
      // which is the value returned.
      return n;
    }
  }
}
```

When a new object needs to be created for the binding, the lifetime of the bound
value matches the scope of the binding.

```carbon
class NoisyDestructor {
  fn Make() -> Self { return {}; }
  impl i32 as ImplicitAs(NoisyDestructor) {
    fn Convert[me: i32]() -> Self { return Make(); }
  }
  destructor {
    Print("Destroyed!");
  }
}

fn G() {
  // Does not print "Destroyed!".
  let n: NoisyDestructor = NoisyDestructor.Make();
  Print("Body of G");
  // Prints "Destroyed!" here.
}

fn H(n: i32) {
  // Does not print "Destroyed!".
  let (v: NoisyDestructor, w: i32) = (n, n);
  Print("Body of H");
  // Prints "Destroyed!" here.
}
```

#### Unused bindings

A syntax like a binding but with `_` in place of an identifier, or `unused`
before the name, can be used to ignore part of a value. Names that are qualified
with the `unused` keyword are visible for name lookup but uses are invalid,
including when they cause ambiguous name lookup errors. If attempted to be used,
a compiler error will be shown to the user, instructing them to either remove
the `unused` qualifier or remove the use.

-   _binding-pattern_ ::= `_` `:` _expression_
-   _binding-pattern_ ::= `unused` _identifier_ `:` _expression_

```carbon
fn F(n: i32) {
  match (n) {
    // ✅ Matches and discards the value of `n`.
    case _: i32 => {}
    // ❌ Error: unreachable.
    default => {}
  }
}
```

As specified in [#1084](/proposals/p1084.md), function redeclarations may
replace binding names with `_`s but may not use different names.

```carbon
fn G(n: i32);
fn H(n: i32);
fn J(n: i32);

// ✅ Does not use `n`.
fn G(_: i32) {}
// ❌ Error: name of parameter does not match declaration.
fn H(m: i32) {}
// ✅ Does not use `n`.
fn J(unused n: i32);
```

##### Alternatives considered

-   [Commented names](/proposals/p2022.md#commented-names)
-   [Only short form support with `_`](/proposals/p2022.md#only-short-form-support-with-_)
-   [Named identifiers prefixed with `_`](/proposals/p2022.md#named-identifiers-prefixed-with-_)
-   [Anonymous, named identifiers](/proposals/p2022.md#anonymous-named-identifiers)
-   [Attributes](/proposals/p2022.md#attributes)

#### Compile-time bindings

A `:!` can be used in place of `:` for a binding that is usable at compile time.

-   _compile-time-pattern_ ::= `template`? _identifier_ `:!` _expression_
-   _compile-time-pattern_ ::= `template`? `_` `:!` _expression_
-   _compile-time-pattern_ ::= `unused` `template`? _identifier_ `:!`
    _expression_
-   _proper-pattern_ ::= _compile-time-pattern_

```carbon
// ✅ `F` takes a symbolic facet parameter `T` and a parameter `x` of type `T`.
fn F(T:! type, x: T) {
  var v: T = x;
}
```

The `template` keyword indicates the binding pattern is introducing a template
binding, so name lookups into the binding will not be fully resolved until its
value is known.

#### `auto` and type deduction

The `auto` keyword is a placeholder for a unique deduced type.

-   _expression_ ::= `auto`

```carbon
fn F(n: i32) {
  var v: auto = SomeComplicatedExpression(n);
  // Equivalent to:
  var w: T = SomeComplicatedExpression(n);
  // ... where `T` is the type of the initializer.
}
```

The `auto` keyword is only permitted in specific contexts. Currently these are:

-   As the return type of a function.
-   As the type of a binding.

It is anticipated that `auto` may be permitted in more contexts in the future,
for example as a placeholder argument in a parameterized type that appears in a
context where `auto` is allowed, such as `Vector(auto)` or `auto*`.

When the type of a binding requires type deduction, the type is deduced against
the type of the scrutinee and deduced values are substituted back into the type
before pattern matching is performed.

```carbon
fn G[T:! Type](p: T*);
class X { impl as ImplicitAs(i32*); }
// ✅ Deduces `T = i32` then implicitly and
// trivially converts `p` to `i32*`.
fn H1(p: i32*) { G(p); }
// ❌ Error, can't deduce `T*` from `X`.
fn H2(p: X) { G(p); }
```

The above is only an illustration; the behavior of type deduction is not yet
specified.

#### Alternatives considered

-   [Shorthand for `auto`](/proposals/p2188.md#shorthand-for-auto)

### `var`

A `var` prefix indicates that a pattern provides mutable storage for the
scrutinee.

-   _proper-pattern_ ::= `var` _proper-pattern_

A `var` pattern matches when its nested pattern matches. The type of the storage
is the resolved type of the nested _pattern_. Any binding patterns within the
nested pattern refer to portions of the corresponding storage rather than to the
scrutinee.

```carbon
fn F(p: i32*);
fn G() {
  match ((1, 2)) {
    // `n` is a mutable `i32`.
    case (var n: i32, 1) => { F(&n); }
    // `n` and `m` are the elements of a mutable `(i32, i32)`.
    case var (n: i32, m: i32) => { F(if n then &n else &m); }
  }
}
```

Pattern matching precedes the initialization of the storage for any `var`
patterns. An introduced variable is only initialized if the complete pattern
matches.

```carbon
class X {
  destructor { Print("Destroyed!"); }
}
fn F(x: X) {
  match ((x, 1 as i32)) {
    case (var y: X, 0) => {}
    case (var z: X, 1) => {}
    // Prints "Destroyed!" only once, when `z` is destroyed.
  }
}
```

A `var` pattern cannot be nested within another `var` pattern. The declaration
syntax `var` _pattern_ `=` _expresson_ `;` is equivalent to `let` `var`
_pattern_ `=` _expression_ `;`.

### Tuple patterns

A tuple of patterns can be used as a pattern.

-   _tuple-pattern_ ::= `(` [_expression_ `,`]\* _proper-pattern_ [`,`
    _pattern_]\* `,`? `)`
-   _proper-pattern_ ::= _tuple-pattern_

A _tuple-pattern_ containing no commas is treated as grouping parens: the
contained _proper-pattern_ is matched directly against the scrutinee. Otherwise,
the behavior is as follows.

A tuple pattern is matched left-to-right. The scrutinee is required to be of
tuple type.

Note that a tuple pattern must contain at least one _proper-pattern_. Otherwise,
it is a tuple-valued expression. However, a tuple pattern and a corresponding
tuple-valued expression are matched in the same way because `==` for a tuple
compares fields left-to-right.

### Struct patterns

A struct can be matched with a struct pattern.

-   _proper-pattern_ ::= `{` [_field-init_ `,`]\* _proper-field-pattern_ [`,`
    _field-pattern_]\* `}`
-   _proper-pattern_ ::= `{` [_field-pattern_ `,`]+ `_` `}`
-   _field-init_ ::= _designator_ `=` _expression_
-   _proper-field-pattern_ ::= _designator_ `=` _proper-pattern_
-   _proper-field-pattern_ ::= _binding-pattern_
-   _field-pattern_ ::= _field-init_
-   _field-pattern_ ::= _proper-field-pattern_

A struct pattern resembles a struct literal, with at least one field initialized
with a proper pattern:

```carbon
match ({.a = 1, .b = 2}) {
  // Struct literal as an expression pattern.
  case {.b = 2, .a = 1} => {}
  // Struct pattern.
  case {.b = n: i32, .a = m: i32} => {}
}
```

The scrutinee is required to be of struct type, and to have the same set of
field names as the pattern. The pattern is matched left-to-right, meaning that
matching is performed in the field order specified in the pattern, not in the
field order of the scrutinee. This is consistent with the behavior of matching
against a struct-valued expression, where the expression pattern becomes the
left operand of the `==` and so determines the order in which `==` comparisons
for fields are performed.

In the case where a field will be bound to an identifier with the same name, a
shorthand syntax is available: `a: T` is synonymous with `.a = a: T`.

```carbon
match ({.a = 1, .b = 2}) {
  case {a: i32, b: i32} => { return a + b; }
}
```

If some fields should be ignored when matching, a trailing `, _` can be added to
specify this:

```carbon
match ({.a = 1, .b = 2}) {
  case {.a = 1, _} => { return 1; }
  case {b: i32, _} => { return b; }
}
```

This is valid even if all fields are actually named in the pattern.

#### Alternatives considered

-   [Struct pattern syntax](/proposals/p2188.md#struct-pattern-syntax)

### Alternative patterns

An alternative pattern is used to match one alternative of a choice type.

-   _proper-pattern_ ::= _callee-expression_ _tuple-pattern_
-   _proper-pattern_ ::= _designator_ _tuple-pattern_?

Here, _callee-expression_ is syntactically an expression that is valid as the
callee in a function call expression, and an alternative pattern is
syntactically a function call expression whose argument list contains at least
one _proper-pattern_.

If a _callee-expression_ is provided, it is required to name a choice type
alternative that has a parameter list, and the scrutinee is implicitly converted
to that choice type. Otherwise, the scrutinee is required to be of some choice
type, and the designator is looked up in that type and is required to name an
alternative with a parameter list if and only if a _tuple-pattern_ is specified.

The pattern matches if the active alternative in the scrutinee is the specified
alternative, and the arguments of the alternative match the given tuple pattern
(if any).

```carbon
choice Optional(T:! Type) {
  None,
  Some(T)
}

match (Optional(i32).None) {
  // ✅ `.None` resolved to `Optional(i32).None`.
  case .None => {}
  // ✅ `.Some` resolved to `Optional(i32).Some`.
  case .Some(n: i32) => { Print("{0}", n); }
  // ❌ Error, no such alternative exists.
  case .Other => {}
}

class X {
  impl as ImplicitAs(Optional(i32));
}

match ({} as X) {
  // ✅ OK, but expression pattern.
  case Optional(i32).None => {}
  // ✅ OK, implicitly converts to `Optional(i32)`.
  case Optional(i32).Some(n: i32) => { Print("{0}", n); }
}
```

Note that a pattern of the form `Optional(T).None` is an expression pattern and
is compared using `==`.

### Templates

Any checking of the type of the scrutinee against the type of the pattern that
cannot be performed because the type of the scrutinee involves a template
parameter is deferred until the template parameter's value is known. During
instantiation, patterns that are not meaningful due to a type error are instead
treated as not matching. This includes cases where an `==` fails because of a
missing `EqWith` implementation.

```carbon
fn TypeName[template T:! Type](x: T) -> String {
  match (x) {
    // ✅ OK, the type of `x` is a template parameter.
    case _: i32 => { return "int"; }
    case _: bool => { return "bool"; }
    case _: auto* => { return "pointer"; }
    default => { return "unknown"; }
  }
}
```

Cases where the match is invalid for reasons not involving the template
parameter are rejected when type-checking the template:

```carbon
fn MeaninglessMatch[template T:! Type](x: T*) {
  match (*x) {
    // ✅ OK, `T` could be a tuple.
    case (_: auto, _: auto) => {}
    default => {}
  }
  match (x->y) {
    // ✅ OK, `T.y` could be a tuple.
    case (_: auto, _: auto) => {}
    default => {}
  }
  match (x) {
    // ❌ Error, tuple pattern cannot match value of non-tuple type `T*`.
    case (_: auto, _: auto) => {}
    default => {}
  }
}
```

### Refutability, overlap, usefulness, and exhaustiveness

Some definitions:

-   A pattern _P_ is _useful_ in the context of a set of patterns _C_ if there
    exists a value that _P_ can match that no pattern in _C_ matches.
-   A set of patterns _C_ is _exhaustive_ if it matches all possible values.
    Equivalently, _C_ is exhaustive if the pattern `_: auto` is not useful in
    the context of _C_.
-   A pattern _P_ is _refutable_ if there are values that it does not match,
    that is, if the pattern `_: auto` is useful in the context of {_P_}.
    Equivalently, the pattern _P_ is _refutable_ if the set of patterns {_P_} is
    not exhaustive.
-   A set of patterns _C_ is _overlapping_ if there exists any value that is
    matched by more than one pattern in _C_.

For the purpose of these terms, expression patterns that match a constant tuple,
struct, or choice value are treated as if they were tuple, struct, or
alternative patterns, respectively, and `bool` is treated like a choice type.
Any expression patterns that remain after applying this rule are considered to
match a single value from an infinite set of values so that a set of expression
patterns is never exhaustive:

```carbon
fn IsEven(n: u8) -> bool {
  // Not considered exhaustive.
  match (n) {
    case 0 => { return true; }
    case 1 => { return false; }
    ...
    case 255 => { return false; }
  }
  // Code here is considered to be reachable.
}
```

```carbon
fn IsTrue(b: bool) -> bool {
  // Considered exhaustive.
  match (b) {
    case false => { return false; }
    case true => { return true; }
  }
  // Code here is considered to be unreachable.
}
```

When determining whether a pattern is useful, no attempt is made to determine
the value of any guards, and instead a worst-case assumption is made: a guard on
that pattern is assumed to evaluate to true and a guard on any pattern in the
context set is assumed to evaluate to false.

We will diagnose the following situations:

-   A pattern is not useful in the context of prior patterns. In a `match`
    statement, this happens if a pattern or `default` cannot match because all
    cases it could cover are handled by prior cases or a prior `default`. For
    example:

    ```carbon
    choice Optional(T:! Type) {
      None,
      Some(T)
    }
    fn F(a: Optional(i32), b: Optional(i32)) {
      match ((a, b)) {
        case (.Some(a: i32), _: auto) => {}
        // ✅ OK, but only matches values of the form `(None, Some)`,
        // because `(Some, Some)` is matched by the previous pattern.
        case (_: auto, .Some(b: i32)) => {}
        // ✅ OK, matches all remaining values.
        case (.None, .None) => {}
        // ❌ Error, this pattern never matches.
        case (_: auto, _: auto) => {}
      }
    }
    ```

-   A pattern match is not exhaustive and the program doesn't explicitly say
    what to do when no pattern matches. For example:

    -   If the patterns in a `match` are not exhaustive and no `default` is
        provided.

        ```carbon
        fn F(n: i32) -> i32 {
          // ❌ Error, this `match` is not exhaustive.
          match (n) {
            case 0 => { return 2; }
            case 1 => { return 3; }
            case 2 => { return 5; }
            case 3 => { return 7; }
            case 4 => { return 11; }
          }
        }
        ```

    -   If a refutable pattern appears in a context where only one pattern can
        be specified, such as a `let` or `var` declaration, and there is no
        fallback behavior. This currently includes all pattern matching contexts
        other than `match` statements, but the `var`/`let`-`else` feature in
        [#1871](https://github.com/carbon-language/carbon-lang/pull/1871) would
        introduce a second context permitting refutable matches, and overloaded
        functions might introduce a third context.

        ```carbon
        fn F(n: i32) {
          // ❌ Error, refutable expression pattern `5` used in context
          // requiring an irrefutable pattern.
          var 5 = n;
        }
        // ❌ Error, refutable expression pattern `5` used in context
        // requiring an irrefutable pattern.
        fn G(n: i32, 5);
        ```

-   When a set of patterns have no ordering or tie-breaker, it is an error for
    them to overlap unless there is a unique best match for any value that
    matches more than one pattern. However, this situation does not apply to any
    current language rule:

    -   For `match` statements, patterns are matched top-down, so overlap is
        permitted.
    -   We do not yet have an approved design for overloaded functions, but it
        is anticipated that declaration order will be used in that case too.
    -   For a set of `impl`s that match a given `impl` lookup, argument
        deduction is used rather than pattern matching, but `impl`s with the
        same type structure are an error unless a `match_first` declaration is
        used to order the `impl`s.

#### Alternatives considered

-   [Treat expression patterns as exhaustive if they cover all possible values](/proposals/p2188.md#treat-expression-patterns-as-exhaustive-if-they-cover-all-possible-values)
-   [Allow non-exhaustive `match` statements](/proposals/p2188.md#allow-non-exhaustive-match-statements)

## Pattern usage

This section is a skeletal design, added to support [the overview](README.md).
It should not be treated as accepted by the core team; rather, it is a
placeholder until we have more time to examine this detail. Please feel welcome
to rewrite and update as appropriate.

### Pattern match control flow

The most powerful form and easiest to explain form of pattern matching is a
dedicated control flow construct that subsumes the `switch` of C and C++ into
something much more powerful, `match`. This is not a novel construct, and is
widely used in existing languages (Swift and Rust among others) and is currently
under active investigation for C++. Carbon's `match` can be used as follows:

```carbon
fn Bar() -> (i32, (f32, f32));
fn Foo() -> f32 {
  match (Bar()) {
    case (42, (x: f32, y: f32)) => {
      return x - y;
    }
    case (p: i32, (x: f32, _: f32)) if (p < 13) => {
      return p * x;
    }
    case (p: i32, _: auto) if (p > 3) => {
      return p * Pi;
    }
    default => {
      return Pi;
    }
  }
}
```

There is a lot going on here. First, let's break down the core structure of a
`match` statement. It accepts a value that will be inspected, here the result of
the call to `Bar()`. It then will find the _first_ `case` that matches this
value, and execute that block. If none match, then it executes the default
block.

Each `case` contains a pattern. The first part is a value pattern
(`(p: i32, _: auto)` for example) optionally followed by an `if` and boolean
predicate. The value pattern has to match, and then the predicate has to
evaluate to `true` for the overall pattern to match. Value patterns can be
composed of the following:

-   An expression (`42` for example), whose value must be equal to match.
-   An identifier to bind the value to, followed by a colon (`:`) and a type
    (`i32` for example). An underscore (`_`) may be used instead of the
    identifier to discard the value once matched.
-   A tuple destructuring pattern containing a tuple of value patterns
    (`(x: f32, y: f32)`) which match against tuples and tuple-like values by
    recursively matching on their elements.
-   An unwrapping pattern containing a nested value pattern which matches
    against a variant or variant-like value by unwrapping it.

In order to match a value, whatever is specified in the pattern must match.
Using `auto` for a type will always match, making `_: auto` the wildcard
pattern.

#### Guards

We allow `case`s within a `match` statement to have _guards_. These are not part
of pattern syntax, but instead are specific to `case` syntax:

-   _case_ ::= `case` _pattern_ [`if` _expression_]? `=>` _block_

A guard indicates that a `case` only matches if some predicate holds. The
bindings in the pattern are in scope in the guard:

```carbon
match (x) {
  case (m: i32, n: i32) if m + n < 5 => { return m - n; }
}
```

For consistency, this facility is also available for `default` clauses, so that
`default` remains equivalent to `case _: auto`.

### Pattern matching in local variables

Value patterns may be used when declaring local variables to conveniently
destructure them and do other type manipulations. However, the patterns must
match at compile time, so they can't use an `if` clause.

```carbon
fn Bar() -> (i32, (f32, f32));
fn Foo() -> i32 {
  var (p: i32, _: auto) = Bar();
  return p;
}
```

This extracts the first value from the result of calling `Bar()` and binds it to
a local variable named `p` which is then returned.

## Open questions

### Slice or array nested value pattern matching

An open question is how to effectively fit a "slice" or "array" pattern into
nested value pattern matching, or whether we shouldn't do so.

### Pattern matching as function overload resolution

Need to flesh out specific details of how overload selection leverages the
pattern matching machinery, what (if any) restrictions are imposed, etc.

## Alternatives considered

-   [Type pattern matching](/proposals/p2188.md#type-pattern-matching)
-   [Allow guards on arbitrary patterns](/proposals/p2188.md#allow-guards-on-arbitrary-patterns)

## References

-   Proposal
    [#2022: Unused Pattern Bindings (Unused Function Parameters)](https://github.com/carbon-language/carbon-lang/pull/2022)
-   Proposal
    [#2188: Pattern matching syntax and semantics](https://github.com/carbon-language/carbon-lang/pull/2188)
