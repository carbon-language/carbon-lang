# Generics: Details

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Interfaces](#interfaces)
-   [Implementing interfaces](#implementing-interfaces)
    -   [Inline `impl`](#inline-impl)
    -   [`extend impl`](#extend-impl)
    -   [Out-of-line `impl`](#out-of-line-impl)
        -   [Defining an `impl` in another library than the type](#defining-an-impl-in-another-library-than-the-type)
    -   [Forward `impl` declaration](#forward-impl-declaration)
    -   [Implementing multiple interfaces](#implementing-multiple-interfaces)
    -   [Avoiding name collisions](#avoiding-name-collisions)
    -   [Qualified member names and compound member access](#qualified-member-names-and-compound-member-access)
    -   [Access](#access)
-   [Checked-generic functions](#checked-generic-functions)
    -   [Symbolic facet bindings](#symbolic-facet-bindings)
    -   [Return type](#return-type)
-   [Interfaces recap](#interfaces-recap)
-   [Facet types](#facet-types)
-   [Named constraints](#named-constraints)
    -   [Subtyping between facet types](#subtyping-between-facet-types)
-   [Combining interfaces by anding facet types](#combining-interfaces-by-anding-facet-types)
-   [Interface requiring other interfaces](#interface-requiring-other-interfaces)
    -   [Interface extension](#interface-extension)
        -   [`extend` and `impl` with named constraints](#extend-and-impl-with-named-constraints)
        -   [Diamond dependency issue](#diamond-dependency-issue)
    -   [Use case: detecting unreachable matches](#use-case-detecting-unreachable-matches)
-   [Adapting types](#adapting-types)
    -   [Adapter compatibility](#adapter-compatibility)
    -   [Extending adapter](#extending-adapter)
    -   [Use case: Using independent libraries together](#use-case-using-independent-libraries-together)
    -   [Use case: Defining an impl for use by other types](#use-case-defining-an-impl-for-use-by-other-types)
    -   [Use case: Private impl](#use-case-private-impl)
    -   [Use case: Accessing interface names](#use-case-accessing-interface-names)
    -   [Future work: Adapter with stricter invariants](#future-work-adapter-with-stricter-invariants)
-   [Associated constants](#associated-constants)
    -   [Associated class functions](#associated-class-functions)
-   [Associated facets](#associated-facets)
-   [Parameterized interfaces](#parameterized-interfaces)
    -   [Parameterized named constraints](#parameterized-named-constraints)
-   [Where constraints](#where-constraints)
    -   [Kinds of `where` constraints](#kinds-of-where-constraints)
        -   [Recursive constraints](#recursive-constraints)
        -   [Rewrite constraints](#rewrite-constraints)
        -   [Same-type constraints](#same-type-constraints)
            -   [Implementation of same-type `ImplicitAs`](#implementation-of-same-type-implicitas)
            -   [Manual type equality](#manual-type-equality)
            -   [Observe declarations](#observe-declarations)
        -   [Implements constraints](#implements-constraints)
            -   [Implied constraints](#implied-constraints)
        -   [Combining constraints](#combining-constraints)
    -   [Satisfying both facet types](#satisfying-both-facet-types)
    -   [Constraints must use a designator](#constraints-must-use-a-designator)
    -   [Referencing names in the interface being defined](#referencing-names-in-the-interface-being-defined)
    -   [Constraint examples and use cases](#constraint-examples-and-use-cases)
        -   [Parameterized type implements interface](#parameterized-type-implements-interface)
        -   [Another type implements parameterized interface](#another-type-implements-parameterized-interface)
        -   [Must be legal type argument constraints](#must-be-legal-type-argument-constraints)
    -   [Named constraint constants](#named-constraint-constants)
-   [Other constraints as facet types](#other-constraints-as-facet-types)
    -   [Is a derived class](#is-a-derived-class)
    -   [Type compatible with another type](#type-compatible-with-another-type)
        -   [Same implementation restriction](#same-implementation-restriction)
        -   [Example: Multiple implementations of the same interface](#example-multiple-implementations-of-the-same-interface)
        -   [Example: Creating an impl out of other implementations](#example-creating-an-impl-out-of-other-implementations)
    -   [Sized types and facet types](#sized-types-and-facet-types)
    -   [Destructor constraints](#destructor-constraints)
-   [Compile-time `let`](#compile-time-let)
-   [Parameterized impl declarations](#parameterized-impl-declarations)
    -   [Impl for a parameterized type](#impl-for-a-parameterized-type)
    -   [Conditional conformance](#conditional-conformance)
    -   [Blanket impl declarations](#blanket-impl-declarations)
        -   [Difference between a blanket impl and a named constraint](#difference-between-a-blanket-impl-and-a-named-constraint)
    -   [Wildcard impl declarations](#wildcard-impl-declarations)
    -   [Combinations](#combinations)
    -   [Lookup resolution and specialization](#lookup-resolution-and-specialization)
        -   [Type structure of an impl declaration](#type-structure-of-an-impl-declaration)
        -   [Orphan rule](#orphan-rule)
        -   [Overlap rule](#overlap-rule)
        -   [Prioritization rule](#prioritization-rule)
        -   [Acyclic rule](#acyclic-rule)
        -   [Termination rule](#termination-rule)
            -   [Non-facet arguments](#non-facet-arguments)
    -   [`final` impl declarations](#final-impl-declarations)
        -   [Libraries that can contain a `final` impl](#libraries-that-can-contain-a-final-impl)
    -   [Comparison to Rust](#comparison-to-rust)
-   [Forward declarations and cyclic references](#forward-declarations-and-cyclic-references)
    -   [Declaring interfaces and named constraints](#declaring-interfaces-and-named-constraints)
    -   [Declaring implementations](#declaring-implementations)
    -   [Matching and agreeing](#matching-and-agreeing)
    -   [Declaration examples](#declaration-examples)
    -   [Example of declaring interfaces with cyclic references](#example-of-declaring-interfaces-with-cyclic-references)
    -   [Interfaces with parameters constrained by the same interface](#interfaces-with-parameters-constrained-by-the-same-interface)
-   [Interface members with definitions](#interface-members-with-definitions)
    -   [Interface defaults](#interface-defaults)
    -   [`final` members](#final-members)
-   [Interface requiring other interfaces revisited](#interface-requiring-other-interfaces-revisited)
    -   [Requirements with `where` constraints](#requirements-with-where-constraints)
-   [Observing a type implements an interface](#observing-a-type-implements-an-interface)
    -   [Observing interface requirements](#observing-interface-requirements)
    -   [Observing blanket impl declarations](#observing-blanket-impl-declarations)
    -   [Observing equal to a type implementing an interface](#observing-equal-to-a-type-implementing-an-interface)
-   [Operator overloading](#operator-overloading)
    -   [Binary operators](#binary-operators)
    -   [`like` operator for implicit conversions](#like-operator-for-implicit-conversions)
-   [Parameterized types](#parameterized-types)
    -   [Generic methods](#generic-methods)
    -   [Conditional methods](#conditional-methods)
    -   [Specialization](#specialization)
-   [Future work](#future-work)
    -   [Dynamic types](#dynamic-types)
        -   [Runtime type parameters](#runtime-type-parameters)
        -   [Runtime type fields](#runtime-type-fields)
    -   [Abstract return types](#abstract-return-types)
    -   [Evolution](#evolution)
    -   [Testing](#testing)
    -   [Impl with state](#impl-with-state)
    -   [Generic associated facets and higher-ranked facets](#generic-associated-facets-and-higher-ranked-facets)
        -   [Generic associated facets](#generic-associated-facets)
        -   [Higher-ranked types](#higher-ranked-types)
    -   [Field requirements](#field-requirements)
    -   [Bridge for C++ customization points](#bridge-for-c-customization-points)
    -   [Variadic arguments](#variadic-arguments)
    -   [Value constraints for template parameters](#value-constraints-for-template-parameters)
-   [References](#references)

<!-- tocstop -->

## Overview

This document goes into the details of the design of Carbon's
[generics](terminology.md#generic-means-compile-time-parameterized), by which we
mean generalizing some language construct with compile-time parameters. These
parameters can be types, [facets](terminology.md#facet), or other values.

Imagine we want to write a function with a type (or facet) parameter. Maybe our
function is `PrintToStdout` and let's say we want to operate on values that have
a type for which we have an implementation of the `ConvertibleToString`
interface. The `ConvertibleToString` interface has a `ToString` method returning
a string. To do this, we give the `PrintToStdout` function two parameters: one
is the value to print, let's call that `val`, the other is the type of that
value, let's call that `T`. The type of `val` is `T`, what is the type of `T`?
Well, since we want to let `T` be any type implementing the
`ConvertibleToString` interface, we express that in the "interfaces are facet
types" model by saying the type of `T` is `ConvertibleToString`.

Since we can figure out `T` from the type of `val`, we don't need the caller to
pass in `T` explicitly, so it can be a
[deduced parameter](terminology.md#deduced-parameter) (also see
[deduced parameters](overview.md#deduced-parameters) in the Generics overview
doc). Basically, the user passes in a value for `val`, and the type of `val`
determines `T`. `T` still gets passed into the function though, and it plays an
important role -- it defines the key used to look up interface implementations.

That interface implementation has the definitions of the functions declared in
the interface. For example, the types `i32` and `String` would have different
implementations of the `ToString` method of the `ConvertibleToString` interface.

In addition to function members, interfaces can include other members that
associate a [compile-time value](/docs/design/README.md#expression-phases) for
any implementing type, called _associated constants_. For example, this can
allow a container interface to include the type of iterators that are returned
from and passed to various container methods.

The function expresses that the type argument is passed in statically, basically
generating a separate function body for every different type passed in, by using
the "compile-time parameter" syntax `:!`. By default, this defines a
[checked-generics parameter](#checked-generic-functions) below. In this case,
the interface contains enough information to
[type and definition check](terminology.md#complete-definition-checking) the
function body -- you can only call functions defined in the interface in the
function body.

Alternatively, the `template` keyword can be included in the signature to make
the type a template parameter. In this case, you could just use `type` instead
of an interface and it will work as long as the function is only called with
types that allow the definition of the function to compile.

The interface bound has other benefits:

-   allows the compiler to deliver clearer error messages,
-   documents expectations, and
-   expresses that a type has certain semantics beyond what is captured in its
    member function names and signatures.

The last piece of the puzzle is calling the function. For a value of type `Song`
to be printed using the `PrintToStdout` function, `Song` needs to implement the
`ConvertibleToString` interface. Interface implementations will usually be
defined either with the type or with the interface. They may also be defined
somewhere else as long as Carbon can be guaranteed to see the definition when
needed. For more on this, see
[the implementing interfaces section](#implementing-interfaces) below.

When the implementation of `ConvertibleToString` for `Song` is declared with
`extend`, every member of `ConvertibleToString` is also a member of `Song`. This
includes members of `ConvertibleToString` that are not explicitly named in the
`impl` definition but have defaults. Whether the type
[extends the implementation](terminology.md#extending-an-impl) or not, you may
access the `ToString` function for a `Song` value `s` by a writing function call
[using a qualified member access expression](terminology.md#qualified-member-access-expression),
like `s.(ConvertibleToString.ToString)()`.

If `Song` doesn't implement an interface or we would like to use a different
implementation of that interface, we can define another type that also has the
same data representation as `Song` that has whatever different interface
implementations we want. However, Carbon won't implicitly convert to that other
type, the user will have to explicitly cast to that type in order to select
those alternate implementations. For more on this, see
[the adapting type section](#adapting-types) below.

We originally considered following Swift and using a witness table
implementation strategy for checked generics, but ultimately decided to only use
that for the dynamic-dispatch case. This is because of the limitations of that
strategy prevent some features that we considered important, as described in
[the witness-table appendix](appendix-witness.md).

## Interfaces

An [interface](terminology.md#interface), defines an API that a given type can
implement. For example, an interface capturing a linear-algebra vector API might
have two methods:

```carbon
interface Vector {
  // Here the `Self` keyword means
  // "the type implementing this interface".
  fn Add[self: Self](b: Self) -> Self;
  fn Scale[self: Self](v: f64) -> Self;
}
```

The syntax here is to match
[how the same members would be defined in a type](/docs/design/classes.md#methods).
Each declaration in the interface defines an
[associated entity](terminology.md#associated-entity). In this example, `Vector`
has two associated methods, `Add` and `Scale`. A type
[implements an interface](#implementing-interfaces) by providing definitions for
all the associated entities declared in the interface,

An interface defines a [facet type](terminology.md#facet-type), that is a type
whose values are [facets](terminology.md#facet). Every type implementing the
interface has a corresponding facet value. So if the type `Point` implements
interface `Vector`, the facet value `Point as Vector` has type `Vector`.

## Implementing interfaces

Carbon interfaces are ["nominal"](terminology.md#nominal-interfaces), which
means that types explicitly describe how they implement interfaces. An
["impl"](terminology.md#impl-implementation-of-an-interface) defines how one
interface is implemented for a type, called the _implementing type_. Every
associated entity is given a definition. Different types satisfying `Vector` can
have different definitions for `Add` and `Scale`, so we say their definitions
are _associated_ with what type is implementing `Vector`. The `impl` defines
what is associated with the implementing type for that interface.

### Inline `impl`

An impl may be defined inline inside the type definition:

```carbon
class Point_Inline {
  var x: f64;
  var y: f64;
  impl as Vector {
    // In this scope, the `Self` keyword is an
    // alias for `Point_Inline`.
    fn Add[self: Self](b: Self) -> Self {
      return {.x = self.x + b.x, .y = self.y + b.y};
    }
    fn Scale[self: Self](v: f64) -> Self {
      return {.x = self.x * v, .y = self.y * v};
    }
  }
}
```

### `extend impl`

Interfaces that are implemented inline with the `extend` keyword contribute to
the type's API:

```carbon
class Point_Extend {
  var x: f64;
  var y: f64;
  extend impl as Vector {
    fn Add[self: Self](b: Self) -> Self {
      return {.x = self.x + b.x, .y = self.y + b.y};
    }
    fn Scale[self: Self](v: f64) -> Self {
      return {.x = self.x * v, .y = self.y * v};
    }
  }
}

var p1: Point_Extend = {.x = 1.0, .y = 2.0};
var p2: Point_Extend = {.x = 2.0, .y = 4.0};
Assert(p1.Scale(2.0) == p2);
Assert(p1.Add(p1) == p2);
```

Without `extend`, those methods may only be accessed with
[qualified member names and compound member access](#qualified-member-names-and-compound-member-access):

```carbon
// Point_Inline did not use `extend` when
// implementing `Vector`:
var a: Point_Inline = {.x = 1.0, .y = 2.0};
// `a` does *not* have `Add` and `Scale` methods:
// ❌ Error: a.Add(a.Scale(2.0));
```

This is consistent with the general Carbon rule that if the names of another
entity affect a class' API, then that is mentioned with an `extend` declaration
in the `class` definition.

**Comparison with other languages:** Rust only defines implementations lexically
outside of the `class` definition. Carbon's approach results in the property
that every type's API is described by declarations inside its `class` definition
and doesn't change afterwards.

**References:** Carbon's interface implementation syntax was first defined in
[proposal #553](https://github.com/carbon-language/carbon-lang/pull/553). In
particular, see
[the alternatives considered](/proposals/p0553.md#interface-implementation-syntax).
This syntax was changed to use `extend` in
[proposal #2760: Consistent `class` and `interface` syntax](https://github.com/carbon-language/carbon-lang/pull/2760).

### Out-of-line `impl`

An impl may also be defined after the type definition, by naming the type
between `impl` and `as`:

```carbon
class Point_OutOfLine {
  var x: f64;
  var y: f64;
}

impl Point_OutOfLine as Vector {
  // In this scope, the `Self` keyword is an
  // alias for `Point_OutOfLine`.
  fn Add[self: Self](b: Self) -> Self {
    return {.x = self.x + b.x, .y = self.y + b.y};
  }
  fn Scale[self: Self](v: f64) -> Self {
    return {.x = self.x * v, .y = self.y * v};
  }
}
```

Since `extend impl` may only be used inside the class definition, out-of-line
definitions do not contribute to the class's API unless there is a corresponding
[forward declaration in the class definition using `extend`](#forward-impl-declaration).

Conversely, being declared or defined lexically inside the class means that
implementation is available to other members defined in the class. For example,
it would allow implementing another interface or method that requires this
interface to be implemented.

**Open question:** Do implementations need to be defined lexically inside the
class to get access to private members, or is it sufficient to be defined in the
same library as the class?

**Comparison with other languages:** Both Rust and Swift support out-of-line
implementation.
[Swift's syntax](https://docs.swift.org/swift-book/LanguageGuide/Protocols.html#ID277)
does this as an "extension" of the original type. In Rust, all implementations
are out-of-line as in
[this example](https://doc.rust-lang.org/rust-by-example/trait.html). Unlike
Swift and Rust, we don't allow a type's API to be modified outside its
definition. So in Carbon a type's API is consistent no matter what is imported,
unlike Swift and Rust.

#### Defining an `impl` in another library than the type

An out-of-line `impl` declaration is allowed to be defined in a different
library from `Point_OutOfLine`, restricted by
[the coherence/orphan rules](#orphan-rule) that ensure that the implementation
of an interface can't change based on imports. In particular, the `impl`
declaration is allowed in the library defining the interface (`Vector` in this
case) in addition to the library that defines the type (`Point_OutOfLine` here).
This (at least partially) addresses
[the expression problem](https://eli.thegreenplace.net/2016/the-expression-problem-and-its-solutions).

You can't use `extend` outside the class definition, so an `impl` declaration in
a different library will never affect the class's API. This means that the API
of a class such as `Point_OutOfLine` doesn't change based on what is imported.
It would be particularly bad if two different libraries implemented interfaces
with conflicting names that both affected the API of a single type. As a
consequence of this restriction, you can find all the names of direct members
(those available by [simple member access](terminology.md#simple-member-access))
of a type in the definition of that type and entities referenced in by an
`extend` declaration in that definition. The only thing that may be in another
library is an `impl` of an interface.

**Rejected alternative:** We could allow types to have different APIs in
different files based on explicit configuration in that file. For example, we
could support a declaration that a given interface or a given method of an
interface is "in scope" for a particular type in this file. With that
declaration, the method could be called using
[simple member access](terminology.md#simple-member-access). This avoids most
concerns arising from name collisions between interfaces. It has a few downsides
though:

-   It increases variability between files, since the same type will have
    different APIs depending on these declarations. This makes it harder to
    copy-paste code between files.
-   It makes reading code harder, since you have to search the file for these
    declarations that affect name lookup.

### Forward `impl` declaration

An `impl` declaration may be forward declared and then defined later. If this is
done using [`extend` to add to the type's API](#extend-impl), only the
declaration in the class definition will use the `extend` keyword, as in this
example:

```carbon
class Point_ExtendForward {
  var x: f64;
  var y: f64;
  // Forward declaration in class definition using `extend`.
  // Signals that you should look in the definition of
  // `Vector` since those methods are included in this type.
  extend impl as Vector;
}

// Definition outside class definition does not.
impl Point_ExtendForward as Vector {
  fn Add[self: Self](b: Self) -> Self {
    return {.x = self.x + b.x, .y = self.y + b.y};
  }
  fn Scale[self: Self](v: f64) -> Self {
    return {.x = self.x * v, .y = self.y * v};
  }
}
```

More about forward declaring implementations in
[its dedicated section](#declaring-implementations).

### Implementing multiple interfaces

To implement more than one interface when defining a type, simply include an
`impl` block or forward declaration per interface.

```carbon
class Point_2Extend {
  var x: f64;
  var y: f64;
  extend impl as Vector {
    fn Add[self: Self](b: Self) -> Self { ... }
    fn Scale[self: Self](v: f64) -> Self { ... }
  }
  extend impl as Drawable {
    fn Draw[self: Self]() { ... }
  }
}
```

Since both were declared using `extend`, all the functions `Add`, `Scale`, and
`Draw` end up a part of the API for `Point_2Extend`.

**Note:** A type may implement any number of different interfaces, but may
provide at most one implementation of any single interface. This makes the act
of selecting an implementation of an interface for a type unambiguous throughout
the whole program.

**Open question:** Should we have some syntax for the case where you want both
names to be given the same implementation? It seems like that might be a common
case, but we won't really know if this is an important case until we get more
experience.

```carbon
class Player {
  var name: String;
  extend impl as Icon {
    fn Name[self: Self]() -> String { return self.name; }
    // ...
  }
  extend impl as GameUnit {
    // Possible syntax options for defining
    // `GameUnit.Name` as the same as `Icon.Name`:
    alias Name = Icon.Name;
    fn Name[self: Self]() -> String = Icon.Name;
    // ...
  }
}
```

### Avoiding name collisions

To avoid name collisions, you can't extend implementations of two interfaces
that have a name in common:

```carbon
class GameBoard {
  extend impl as Drawable {
    fn Draw[self: Self]() { ... }
  }
  extend impl as EndOfGame {
    // ❌ Error: `GameBoard` has two methods named `Draw`.
    fn Draw[self: Self]() { ... }
    fn Winner[self: Self](player: i32) { ... }
  }
}
```

To implement two interfaces that have a name in common, omit `extend` for one or
both.

You might also omit `extend` when implementing an interface for a type to avoid
cluttering the API of that type or to avoid a name collision with another member
of that type. A syntax for reusing method implementations allows us to include
names from an implementation selectively:

```carbon
class Point_ReuseMethodInImpl {
  var x: f64;
  var y: f64;
  // `Add()` is a method of `Point_ReuseMethodInImpl`.
  fn Add[self: Self](b: Self) -> Self {
    return {.x = self.x + b.x, .y = self.y + b.y};
  }
  // No `extend`, so other members of `Vector` are not
  // part of `Point_ReuseMethodInImpl`'s API.
  impl as Vector {
    // Syntax TBD:
    alias Add = Point_ReuseMethodInImpl.Add;
    fn Scale[self: Self](v: f64) -> Self {
      return {.x = self.x * v, .y = self.y * v};
    }
  }
}

// OR:

class Point_IncludeMethodFromImpl {
  var x: f64;
  var y: f64;
  // No `extend`, so members of `Vector` are not
  // part of `Point_IncludeMethodFromImpl`'s API.
  impl as Vector {
    fn Add[self: Self](b: Self) -> Self {
      return {.x = self.x + b.x, .y = self.y + b.y};
    }
    fn Scale[self: Self](v: f64) -> Self {
      return {.x = self.x * v, .y = self.y * v};
    }
  }
  // Include `Add` explicitly as a member.
  alias Add = Vector.Add;
}

// OR:

// This is the same as `Point_ReuseMethodInImpl`,
// except the `impl` is out-of-line.
class Point_ReuseByOutOfLine {
  var x: f64;
  var y: f64;
  fn Add[self: Self](b: Self) -> Self {
    return {.x = self.x + b.x, .y = self.y + b.y};
  }
}

impl Point_ReuseByOutOfLine as Vector {
  // Syntax TBD:
  alias Add = Point_ReuseByOutOfLine.Add;
  fn Scale[self: Self](v: f64) -> Self {
    return {.x = self.x * v, .y = self.y * v};
  }
}
```

### Qualified member names and compound member access

```carbon
class Point_NoExtend {
  var x: f64;
  var y: f64;
}

impl Point_NoExtend as Vector { ... }
```

Given a value of type `Point_NoExtend` and an interface `Vector` implemented for
that type, you can access the methods from that interface using a
[qualified member access expression](terminology.md#qualified-member-access-expression)
whether or not the implementation is done with an
[`extend impl` declaration](#extend-impl). The qualified member access
expression writes the member's _qualified name_ in the parentheses of the
[compound member access syntax](/docs/design/expressions/member_access.md):

```carbon
var p1: Point_NoExtend = {.x = 1.0, .y = 2.0};
var p2: Point_NoExtend = {.x = 2.0, .y = 4.0};
Assert(p1.(Vector.Scale)(2.0) == p2);
Assert(p1.(Vector.Add)(p1) == p2);
```

Note that the name in the parens is looked up in the containing scope, not in
the names of members of `Point_NoExtend`. So if there was another interface
`Drawable` with method `Draw` defined in the `Plot` package also implemented for
`Point_NoExtend`, as in:

```carbon
package Plot;
import Points;

interface Drawable {
  fn Draw[self: Self]();
}

impl Points.Point_NoExtend as Drawable { ... }
```

You could access `Draw` with a qualified name:

```carbon
import Plot;
import Points;

var p: Points.Point_NoExtend = {.x = 1.0, .y = 2.0};
p.(Plot.Drawable.Draw)();
```

**Comparison with other languages:** This is intended to be analogous to, in
C++, adding `ClassName::` in front of a member name to disambiguate, such as
[names defined in both a parent and child class](https://stackoverflow.com/questions/357307/how-to-call-a-parent-class-function-from-derived-class-function).

### Access

An `impl` must be visible to all code that can see both the type and the
interface being implemented:

-   If either the type or interface is private to a single file, then since the
    only way to define the `impl` is to use that private name, the `impl` must
    be defined private to that file as well.
-   Otherwise, if the type or interface is private but declared in an API file,
    then the `impl` must be declared in the same file so the existence of that
    `impl` is visible to all files in that library.
-   Otherwise, the `impl` must be declared in the public API file of the
    library, so it is visible in all places that might use it.

No access control modifiers are allowed on `impl` declarations, an `impl` is
always visible to the intersection of the visibility of all names used in the
declaration of the `impl`.

## Checked-generic functions

Here is a function that can accept values of any type that has implemented the
`Vector` interface:

```carbon
fn AddAndScaleGeneric[T:! Vector](a: T, b: T, s: f64) -> T {
  return a.Add(b).Scale(s);
}
var v: Point_Extend = AddAndScaleGeneric(a, w, 2.5);
```

Here `T` is a facet whose type is `Vector`. The `:!` syntax means that `T` is a
_[compile-time binding](terminology.md#bindings)_. Here specifically it declares
a _symbolic binding_ since it did not use the `template` keyword to mark it as a
_template binding_.

> **References:** The `:!` syntax was accepted in
> [proposal #676](https://github.com/carbon-language/carbon-lang/pull/676).

Since this symbolic binding pattern is in a function declaration, it marks a
_[checked](terminology.md#checked-versus-template-parameters)
[generic parameter](terminology.md#generic-means-compile-time-parameterized)_.
That means its value must be known to the caller at compile-time, but we will
only use the information present in the signature of the function to type check
the body of `AddAndScaleGeneric`'s definition.

Note that types may also be given compile-time parameters, see the
["parameterized types" section](#parameterized-types).

### Symbolic facet bindings

In our example, `T` is a facet which may be used in type position in the rest of
the function. Furthermore, since it omits the keyword `template` prefix, this is
a symbolic binding. so we need to be able to typecheck the body of the function
without knowing the specific value `T` from the caller.

This typechecking is done by looking at the constraint on `T`. In the example,
the constraint on `T` says that every value of `T` implements the `Vector`
interface and so has a `Vector.Add` and a `Vector.Scale` method.

Names are looked up in the body of `AddAndScaleGeneric` for values of type `T`
in `Vector`. This means that `AddAndScaleGeneric` is interpreted as equivalent
to adding a `Vector`
[qualification](#qualified-member-names-and-compound-member-access) to replace
all simple member accesses of `T`:

```carbon
fn AddAndScaleGeneric[T:! Vector](a: T, b: T, s: Double) -> T {
  return a.(Vector.Add)(b).(Vector.Scale)(s);
}
```

With these qualifications, the function can be type-checked for any `T`
implementing `Vector`. This type checking is equivalent to type checking the
function with `T` set to an [archetype](terminology.md#archetype) of `Vector`.
An archetype is a placeholder type considered to satisfy its constraint, which
is `Vector` in this case, and no more. It acts as the most general type
satisfying the interface. The effect of this is that an archetype of `Vector`
acts like a [supertype](https://en.wikipedia.org/wiki/Subtyping) of any `T`
implementing `Vector`.

For name lookup purposes, an archetype is considered to
[extend the implementation of its constraint](terminology.md#extending-an-impl).
The only oddity is that the archetype may have different names for members than
specific types `T` that don't extend the implementation of interfaces from the
constraint. This difference in names can also occur for supertypes in C++, for
example members in a derived class can hide members in the base class with the
same name, though it is not that common for it to come up in practice.

The behavior of calling `AddAndScaleGeneric` with a value of a specific type
like `Point_Extend` is to set `T` to `Point_Extend` after all the names have
been qualified.

```carbon
// AddAndScaleGeneric with T = Point_Extend
fn AddAndScaleForPoint_Extend(
    a: Point_Extend, b: Point_Extend, s: Double)
    -> Point_Extend {
  return a.(Vector.Add)(b).(Vector.Scale)(s);
}
```

This qualification gives a consistent interpretation to the body of the function
even when the type supplied by the caller does not
[extend the implementation of the interface](terminology.md#extending-an-impl),
like `Point_NoExtend`:

```carbon
// AddAndScaleGeneric with T = Point_NoExtend
fn AddAndScaleForPoint_NoExtend(
    a: Point_NoExtend, b: Point_NoExtend, s: Double)
    -> Point_NoExtend {
  // ✅ This works even though `a.Add(b).Scale(s)` wouldn't.
  return a.(Vector.Add)(b).(Vector.Scale)(s);
}
```

### Return type

From the caller's perspective, the return type is the result of substituting the
caller's values for the generic parameters into the return type expression. So
`AddAndScaleGeneric` called with `Point_Extend` values returns a `Point_Extend`
and called with `Point_NoExtend` values returns a `Point_NoExtend`. So looking
up a member on the resulting value will look in `Point_Extend` or
`Point_NoExtend` rather than `Vector`.

This is part of realizing
[the goal that generic functions can be used in place of regular functions without changing the return type that callers see](goals.md#path-from-regular-functions).
In this example, `AddAndScaleGeneric` can be substituted for
`AddAndScaleForPoint_Extend` and `AddAndScaleForPoint_NoExtend` without
affecting the return types. This may require a conversion of the return value to
the type that the caller expects, from the erased type used inside a
checked-generic function.

A checked-generic caller of a checked-generic function performs the same
substitution process to determine the return type, but the result may be a
symbolic value. In this example of calling a checked generic from another
checked generic,

```carbon
fn DoubleThreeTimes[U:! Vector](a: U) -> U {
  return AddAndScaleGeneric(a, a, 2.0).Scale(2.0);
}
```

the return type of `AddAndScaleGeneric` is found by substituting in the `U` from
`DoubleThreeTimes` for the `T` from `AddAndScaleGeneric` in the return type
expression of `AddAndScaleGeneric`. `U` is an archetype of `Vector`, and so acts
as if it extends `Vector` and therefore has a `Scale` method.

If `U` had a more specific type, the return value would have the additional
capabilities of `U`. For example, given a parameterized type `GeneralPoint`
implementing `Vector`, and a function that takes a `GeneralPoint` and calls
`AddAndScaleGeneric` with it:

```carbon
class GeneralPoint(C:! Numeric) {
  impl as Vector { ... }
  fn Get[self: Self](i: i32) -> C;
}

fn CallWithGeneralPoint[C:! Numeric](p: GeneralPoint(C)) -> C {
  // `AddAndScaleGeneric` returns `T` and in these calls `T` is
  // deduced to be `GeneralPoint(C)`.

  // ❌ Illegal: AddAndScaleGeneric(p, p, 2.0).Scale(2.0);
  //    `GeneralPoint(C)` implements but does not extend `Vector`,
  //    and so does not have a `Scale` method.

  // ✅ Allowed: `GeneralPoint(C)` has a `Get` method
  AddAndScaleGeneric(p, p, 2.0).Get(0);

  // ✅ Allowed: `GeneralPoint(C)` implements `Vector`, and so has
  //    a `Vector.Scale` method. `Vector.Scale` returns `Self`
  //    which is `GeneralPoint(C)` again, and so has a `Get`
  //    method.
  return AddAndScaleGeneric(p, p, 2.0).(Vector.Scale)(2.0).Get(0);
}
```

The result of the call to `AddAndScaleGeneric` from `CallWithGeneralPoint` has
type `GeneralPoint(C)` and so has a `Get` method and a `Vector.Scale` method.
But, in contrast to how `DoubleThreeTimes` works, since `Vector` is implemented
without `extend` the return value in this case does not directly have a `Scale`
method.

## Interfaces recap

Interfaces have a name and a definition.

The definition of an interface consists of a set of declarations. Each
declaration defines a requirement for any `impl` that is in turn a capability
that consumers of that `impl` can rely on. Typically those declarations also
have names, useful for both saying how the `impl` satisfies the requirement and
accessing the capability.

Interfaces are ["nominal"](terminology.md#nominal-interfaces), which means their
name is significant. So two interfaces with the same body definition but
different names are different, just like two classes with the same definition
but different names are considered different types. For example, lets say we
define another interface, say `LegoFish`, with the same `Add` and `Scale` method
signatures. Implementing `Vector` would not imply an implementation of
`LegoFish`, because the `impl` definition explicitly refers to the name
`Vector`.

An interface's name may be used in a few different contexts:

-   to define [an `impl` for a type](#implementing-interfaces),
-   as a namespace name in
    [a qualified name](#qualified-member-names-and-compound-member-access), and
-   as a [facet type](terminology.md#facet-type) for
    [a facet binding](#symbolic-facet-bindings).

While interfaces are examples of facet types, facet types are a more general
concept, for which interfaces are a building block.

## Facet types

A [facet type](terminology.md#facet-type) consists of a set of requirements and
a set of names. Requirements are typically a set of interfaces that a type must
satisfy, though other kinds of requirements are added below. The names are
aliases for qualified names in those interfaces.

An interface is one particularly simple example of a facet type. For example,
`Vector` as a facet type has a set of requirements consisting of the single
interface `Vector`. Its set of names consists of `Add` and `Scale` which are
aliases for the corresponding qualified names inside `Vector` as a namespace.

The requirements determine which types may be implicitly converted to a given
facet type. The result of this conversion is a [facet](terminology.md#facet).
For example, `Point_Inline` from [the "Inline `impl`" section](#inline-impl)
implements `Vector`, so `Point_Inline` may be implicitly converted to `Vector`
as considered as a type. The result is `Point_Inline as Vector`, which has the
members of `Vector` instead of the members of `Point_Inline`. If the facet
`Point_Inline as Vector` is used in a type position, it is implicitly converted
back to type `type`, see
["values usable as types" in the design overview](/docs/design/README.md#values-usable-as-types).
This recovers the original type for the facet, so
`(Point_Inline as Vector) as type` is `Point_Inline` again.

However, when a facet type like `Vector` is used as the binding type of a
symbolic binding, as in `T:! Vector`, the
[symbolic facet binding](#symbolic-facet-bindings) `T` is disassociated with
whatever facet value `T` is eventually bound to. Instead, `T` is treated as an
[archetype](terminology.md#archetype), with the members and
[member access](/docs/design/expressions/member_access.md) determined by the
names of the facet type.

This general structure of facet types holds not just for interfaces, but others
described in the rest of this document.

## Named constraints

If the interfaces discussed above are the building blocks for facet types,
[named constraints](terminology.md#named-constraints) describe how they may be
composed together. Unlike interfaces which are nominal, the name of a named
constraint is not a part of its value. Two different named constraints with the
same definition are equivalent even if they have different names. This is
because types don't have to explicitly specify which named constraints they
implement, types automatically implement any named constraints they can satisfy.

A named constraint definition can contain interface requirements using
`require Self impls` declarations and names using `alias` declarations. Note
that this allows us to declare the aspects of a facet type directly.

```carbon
constraint VectorLegoFish {
  // Interface implementation requirements
  require Self impls Vector;
  require Self impls LegoFish;
  // Names
  alias Scale = Vector.Scale;
  alias VAdd = Vector.Add;
  alias LFAdd = LegoFish.Add;
}
```

A `require Self impls` requirement may alternatively be on a named constraint,
instead of an interface, to add all the requirements of another named constraint
without adding any of the names:

```carbon
constraint DrawVectorLegoFish {
  // The same as requiring both `Vector` and `LegoFish`.
  require Self impls VectorLegoFish;
  // A regular interface requirement. No syntactic difference.
  require Self impls Drawable;
}
```

In general, Carbon makes no syntactic distinction between the uses of named
constraints and interfaces, so one may be replaced with the other without
affecting users. To accomplish this, Carbon allows a named constraint to be used
whenever an interface may be. This includes all of these
[uses of interfaces](#interfaces-recap):

-   A type may `impl` a named constraint to say that it implements all of the
    requirements of the named constraint, as
    [described below](#extend-and-impl-with-named-constraints).
-   A named constraint may be used as a namespace name in
    [a qualified name](#qualified-member-names-and-compound-member-access). For
    example, `VectorLegoFish.VAdd` refers to the same name as `Vector.Add`.
-   A named constraint may be used as a [facet type](terminology.md#facet-type)
    for [a facet binding](#symbolic-facet-bindings).

We don't expect developers to directly define many named constraints, but other
constructs we do expect them to use will be defined in terms of them. For
example, if `type` were not a keyword, we could define the Carbon builtin `type`
as:

```carbon
constraint type { }
```

That is, `type` is the facet type with no requirements (so matches every type),
and defines no names.

```carbon
fn Identity[T:! type](x: T) -> T {
  // Can accept values of any type. But, since we know nothing about the
  // type, we don't know about any operations on `x` inside this function.
  return x;
}

var i: i32 = Identity(3);
var s: String = Identity("string");
```

In general, the declarations in `constraint` definition match a subset of the
declarations in an `interface`. These named constraints can be used with checked
generics, as opposed to templates, and only include required interfaces and
aliases to named members of those interfaces.

To declare a named constraint that includes other declarations for use with
template parameters, use the `template` keyword before `constraint`. Method,
associated constant, and associated function requirements may only be declared
inside a `template constraint`. Note that a checked-generic constraint ignores
the names of members defined for a type, but a template constraint can depend on
them.

There is an analogy between declarations used in a `template constraint` and in
an `interface` definition. If an `interface` `I` has (non-`alias`,
non-`require`) declarations `X`, `Y`, and `Z`, like so:

```carbon
interface I {
  X;
  Y;
  Z;
}
```

Then a type implementing `I` would have `impl as I` with definitions for `X`,
`Y`, and `Z`, as in:

```carbon
class ImplementsI {
  // ...
  impl as I {
    X { ... }
    Y { ... }
    Z { ... }
  }
}
```

But a `template constraint`, `S`:

```carbon
template constraint S {
  X;
  Y;
  Z;
}
```

would match any type with definitions for `X`, `Y`, and `Z` directly:

```carbon
class ImplementsS {
  // ...
  X { ... }
  Y { ... }
  Z { ... }
}
```

### Subtyping between facet types

There is a subtyping relationship between facet types that allows calls of one
generic function from another as long as it has a subset of the requirements.

Given a symbolic facet binding `T` with facet type `I1`, it satisfies a facet
type `I2` as long as the requirements of `I1` are a superset of the requirements
of `I2`. This means a value `x: T` may be passed to functions requiring types to
satisfy `I2`, as in this example:

```carbon
interface Printable { fn Print[self: Self](); }
interface Renderable { fn Draw[self: Self](); }

constraint PrintAndRender {
  require Self impls Printable;
  require Self impls Renderable;
}
constraint JustPrint {
  require Self impls Printable;
}

fn PrintIt[T2:! JustPrint](x2: T2) {
  x2.(Printable.Print)();
}
fn PrintDrawPrint[T1:! PrintAndRender](x1: T1) {
  // x1 implements `Printable` and `Renderable`.
  x1.(Printable.Print)();
  x1.(Renderable.Draw)();
  // Can call `PrintIt` since `T1` satisfies `JustPrint` since
  // it implements `Printable` (in addition to `Renderable`).
  PrintIt(x1);
}
```

## Combining interfaces by anding facet types

In order to support functions that require more than one interface to be
implemented, we provide a combination operator on facet types, written `&`. This
operator gives the facet type with the union of all the requirements and the
union of the names.

```carbon
interface Printable {
  fn Print[self: Self]();
}
interface Renderable {
  fn Center[self: Self]() -> (i32, i32);
  fn Draw[self: Self]();
}

// `Printable & Renderable` is syntactic sugar for this facet type:
constraint {
  require Self impls Printable;
  require Self impls Renderable;
  alias Print = Printable.Print;
  alias Center = Renderable.Center;
  alias Draw = Renderable.Draw;
}

fn PrintThenDraw[T:! Printable & Renderable](x: T) {
  // Can use methods of `Printable` or `Renderable` on `x` here.
  x.Print();  // Same as `x.(Printable.Print)();`.
  x.Draw();  // Same as `x.(Renderable.Draw)();`.
}

class Sprite {
  // ...
  extend impl as Printable {
    fn Print[self: Self]() { ... }
  }
  extend impl as Renderable {
    fn Center[self: Self]() -> (i32, i32) { ... }
    fn Draw[self: Self]() { ... }
  }
}

var s: Sprite = ...;
PrintThenDraw(s);
```

It is an error to use any names that conflict between the two interfaces.

```carbon
interface Renderable {
  fn Center[self: Self]() -> (i32, i32);
  fn Draw[self: Self]();
}
interface EndOfGame {
  fn Draw[self: Self]();
  fn Winner[self: Self](player: i32);
}
fn F[T:! Renderable & EndOfGame](x: T) {
  // ❌ Error: Ambiguous, use either `(Renderable.Draw)`
  //           or `(EndOfGame.Draw)`.
  x.Draw();
}
```

Conflicts can be resolved at the call site using a
[qualified member access expression](#qualified-member-names-and-compound-member-access),
or by defining a named constraint explicitly and renaming the methods:

```carbon
constraint RenderableAndEndOfGame {
  require Self impls Renderable;
  require Self impls EndOfGame;
  alias Center = Renderable.Center;
  alias RenderableDraw = Renderable.Draw;
  alias TieGame = EndOfGame.Draw;
  alias Winner = EndOfGame.Winner;
}

fn RenderTieGame[T:! RenderableAndEndOfGame](x: T) {
  // ✅ Calls `Renderable.Draw`:
  x.RenderableDraw();
  // ✅ Calls `EndOfGame.Draw`:
  x.TieGame();
}
```

Note that `&` is associative and commutative, and so it is well defined on sets
of interfaces, or other facet types, independent of order.

Note that we do _not_ consider two facet types using the same name to mean the
same thing to be a conflict. For example, combining a facet type with itself
gives itself, `MyTypeOfType & MyTypeOfType == MyTypeOfType`. Also, given two
[interface extensions](#interface-extension) of a common base interface, the
combination should not conflict on any names in the common base.

To add to the requirements of a facet type without affecting the names, and so
avoid the possibility of name conflicts, names, use a
[`where .Self impls` clause](#implements-constraints).

```
// `Printable where .Self impls Renderable` is equivalent to:
constraint {
  require Self impls Printable;
  require Self impls Renderable;
  alias Print = Printable.Print;
}
```

You might use this to add requirements on interfaces used for
[operator overloading](#operator-overloading), where merely implementing the
interface is enough to be able to use the operator to access the functionality.

Note that the expressions `A & B` and `A where .Self impls B` have the same
requirements, and so you would be able to switch a function declaration between
them without affecting callers.

**Alternatives considered:** See
[Carbon: Access to interface methods](https://docs.google.com/document/d/17IXDdu384x1t9RimQ01bhx4-nWzs4ZEeke4eO6ImQNc/edit?resourcekey=0-Fe44R-0DhQBlw0gs2ujNJA).

**Rejected alternative:** Instead of using `&` as the combining operator, we
considered using `+`,
[like Rust](https://rust-lang.github.io/rfcs/0087-trait-bounds-with-plus.html).
The main difference from Rust's `+` is how you
[qualify names when there is a conflict](https://doc.rust-lang.org/rust-by-example/trait/disambiguating.html).
See [issue #531](https://github.com/carbon-language/carbon-lang/issues/531) for
the discussion.

## Interface requiring other interfaces

Some interfaces depend on other interfaces being implemented for the same type.
For example, in C++,
[the `Container` concept](https://en.cppreference.com/w/cpp/named_req/Container#Other_requirements)
requires all containers to also satisfy the requirements of
`DefaultConstructible`, `CopyConstructible`, `Eq`, and `Swappable`. This is
already a capability for [facet types in general](#facet-types). For consistency
we use the same semantics and `require Self impls` syntax as we do for
[named constraints](#named-constraints):

```carbon
interface Equatable { fn Equals[self: Self](rhs: Self) -> bool; }

interface Iterable {
  fn Advance[addr self: Self*]() -> bool;
  require Self impls Equatable;
}

def DoAdvanceAndEquals[T:! Iterable](x: T) {
  // `x` has type `T` that implements `Iterable`, and so has `Advance`.
  x.Advance();
  // `Iterable` requires an implementation of `Equatable`,
  // so `T` also implements `Equatable`.
  x.(Equatable.Equals)(x);
}

class Iota {
  extend impl as Iterable { fn Advance[self: Self]() { ... } }
  extend impl as Equatable { fn Equals[self: Self](rhs: Self) -> bool { ... } }
}
var x: Iota;
DoAdvanceAndEquals(x);
```

Like with named constraints, an interface implementation requirement doesn't by
itself add any names to the interface, but again those can be added with `alias`
declarations:

```carbon
interface Hashable {
  fn Hash[self: Self]() -> u64;
  require Self impls Equatable;
  alias Equals = Equatable.Equals;
}

def DoHashAndEquals[T:! Hashable](x: T) {
  // Now both `Hash` and `Equals` are available directly:
  x.Hash();
  x.Equals(x);
}
```

**Comparison with other languages:**
[This feature is called "Supertraits" in Rust](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#using-supertraits-to-require-one-traits-functionality-within-another-trait).

**Note:** The design for this feature is continued in
[a later section](#interface-requiring-other-interfaces-revisited).

### Interface extension

When implementing an interface, we allow implementing the aliased names as well.
In the case of `Hashable` above, this includes all the members of `Equatable`,
obviating the need to implement `Equatable` itself:

```carbon
class Song {
  extend impl as Hashable {
    fn Hash[self: Self]() -> u64 { ... }
    fn Equals[self: Self](rhs: Self) -> bool { ... }
  }
}
var y: Song;
DoHashAndEquals(y);
```

This allows us to say that `Hashable`
["extends"](terminology.md#extending-an-interface) `Equatable`, with some
benefits:

-   This allows `Equatable` to be an implementation detail of `Hashable`.
-   This allows types implementing `Hashable` to implement all of its API in one
    place.
-   This reduces the boilerplate for types implementing `Hashable`.

We expect this concept to be common enough to warrant dedicated `interface`
syntax:

```carbon
interface Equatable { fn Equals[self: Self](rhs: Self) -> bool; }

interface Hashable {
  extend Equatable;
  fn Hash[self: Self]() -> u64;
}
// is equivalent to the definition of Hashable from before:
// interface Hashable {
//   require Self impls Equatable;
//   alias Equals = Equatable.Equals;
//   fn Hash[self: Self]() -> u64;
// }
```

No names in `Hashable` are allowed to conflict with names in `Equatable` (unless
those names are marked as `upcoming` or `deprecated` as in
[evolution future work](#evolution)). Hopefully this won't be a problem in
practice, since interface extension is a very closely coupled relationship, but
this may be something we will have to revisit in the future.

Examples:

-   The C++
    [Boost.Graph library](https://www.boost.org/doc/libs/1_74_0/libs/graph/doc/)
    [graph concepts](https://www.boost.org/doc/libs/1_74_0/libs/graph/doc/graph_concepts.html#fig:graph-concepts)
    has many refining relationships between concepts.
    [Carbon generics use case: graph library](https://docs.google.com/document/d/15Brjv8NO_96jseSesqer5HbghqSTJICJ_fTaZOH0Mg4/edit?usp=sharing&resourcekey=0-CYSbd6-xF8vYHv9m1rolEQ)
    shows how those concepts might be translated into Carbon interfaces.
-   The [C++ concepts](https://en.cppreference.com/w/cpp/named_req) for
    containers, iterators, and concurrency include many requirement
    relationships.
-   Swift protocols, such as
    [Collection](https://developer.apple.com/documentation/swift/collection).

To write an interface extending multiple interfaces, use multiple `extend`
declarations. For example, the
[`BinaryInteger` protocol in Swift](https://developer.apple.com/documentation/swift/binaryinteger)
inherits from `CustomStringConvertible`, `Hashable`, `Numeric`, and `Stridable`.
The [`SetAlgebra` protocol](https://swiftdoc.org/v5.1/protocol/setalgebra/)
extends `Equatable` and `ExpressibleByArrayLiteral`, which would be declared in
Carbon:

```carbon
interface SetAlgebra {
  extend Equatable;
  extend ExpressibleByArrayLiteral;
}
```

**Alternative considered:** The `extend` declarations are in the body of the
`interface` definition instead of the header so we can use
[associated constants](terminology.md#associated-entity) also defined in the
body in parameters or constraints of the interface being extended.

```carbon
// A type can implement `ConvertibleTo` many times,
// using different values of `T`.
interface ConvertibleTo(T:! type) { ... }

// A type can only implement `PreferredConversion` once.
interface PreferredConversion {
  let AssociatedFacet:! type;
  // `extend` is in the body of an `interface`
  // definition. This allows extending an expression
  // that uses an associated facet.
  extend ConvertibleTo(AssociatedFacet);
}
```

#### `extend` and `impl` with named constraints

The `extend` declaration makes sense with the same meaning inside a
[`constraint`](#named-constraints) definition, and so is also supported.

```carbon
interface Media {
  fn Play[self: Self]();
}
interface Job {
  fn Run[self: Self]();
}

constraint Combined {
  extend Media;
  extend Job;
}
```

This definition of `Combined` is equivalent to requiring both the `Media` and
`Job` interfaces being implemented, and aliases their methods.

```carbon
// Equivalent
constraint Combined {
  require Self impls Media;
  alias Play = Media.Play;
  require Self impls Job;
  alias Run = Job.Run;
}
```

Notice how `Combined` has aliases for all the methods in the interfaces it
requires. That condition is sufficient to allow a type to `impl` the named
constraint:

```carbon
class Song {
  extend impl as Combined {
    fn Play[self: Self]() { ... }
    fn Run[self: Self]() { ... }
  }
}
```

This is equivalent to implementing the required interfaces directly:

```carbon
class Song {
  extend impl as Media {
    fn Play[self: Self]() { ... }
  }
  extend impl as Job {
    fn Run[self: Self]() { ... }
  }
}
```

This is just like when you get an implementation of `Equatable` by implementing
`Hashable` when `Hashable` extends `Equatable`. This provides a tool useful for
[evolution](#evolution).

Conversely, an `interface` can extend a `constraint`:

```carbon
interface MovieCodec {
  extend Combined;

  fn Load[addr self: Self*](filename: String);
}
```

This gives `MovieCodec` the same requirements and names as `Combined`, and so is
equivalent to:

```carbon
interface MovieCodec {
  require Self impls Media;
  alias Play = Media.Play;
  require Self impls Job;
  alias Run = Job.Run;

  fn Load[addr self: Self*](filename: String);
}
```

#### Diamond dependency issue

Consider this set of interfaces, simplified from
[this example generic graph library doc](https://docs.google.com/document/d/15Brjv8NO_96jseSesqer5HbghqSTJICJ_fTaZOH0Mg4/edit?usp=sharing&resourcekey=0-CYSbd6-xF8vYHv9m1rolEQ):

```carbon
interface Graph {
  fn Source[addr self: Self*](e: EdgeDescriptor) -> VertexDescriptor;
  fn Target[addr self: Self*](e: EdgeDescriptor) -> VertexDescriptor;
}

interface IncidenceGraph {
  extend Graph;
  fn OutEdges[addr self: Self*](u: VertexDescriptor)
    -> (EdgeIterator, EdgeIterator);
}

interface EdgeListGraph {
  extend Graph;
  fn Edges[addr self: Self*]() -> (EdgeIterator, EdgeIterator);
}
```

We need to specify what happens when a graph type implements both
`IncidenceGraph` and `EdgeListGraph`, since both interfaces extend the `Graph`
interface.

```carbon
class MyEdgeListIncidenceGraph {
  extend impl as IncidenceGraph { ... }
  extend impl as EdgeListGraph { ... }
}
```

The rule is that we need one definition of each method of `Graph`. Each method
though could be defined in the `impl` block of `IncidenceGraph`,
`EdgeListGraph`, or `Graph`. These would all be valid:

-   `IncidenceGraph` implements all methods of `Graph`, `EdgeListGraph`
    implements none of them.

    ```carbon
    class MyEdgeListIncidenceGraph {
      extend impl as IncidenceGraph {
        fn Source[self: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
        fn Target[self: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
        fn OutEdges[addr self: Self*](u: VertexDescriptor)
            -> (EdgeIterator, EdgeIterator) { ... }
      }
      extend impl as EdgeListGraph {
        fn Edges[addr self: Self*]() -> (EdgeIterator, EdgeIterator) { ... }
      }
    }
    ```

-   `IncidenceGraph` and `EdgeListGraph` implement all methods of `Graph`
    between them, but with no overlap.

    ```carbon
    class MyEdgeListIncidenceGraph {
      extend impl as IncidenceGraph {
        fn Source[self: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
        fn OutEdges[addr self: Self*](u: VertexDescriptor)
            -> (EdgeIterator, EdgeIterator) { ... }
      }
      extend impl as EdgeListGraph {
        fn Target[self: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
        fn Edges[addr self: Self*]() -> (EdgeIterator, EdgeIterator) { ... }
      }
    }
    ```

-   Explicitly implementing `Graph`.

    ```carbon
    class MyEdgeListIncidenceGraph {
      extend impl as Graph {
        fn Source[self: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
        fn Target[self: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
      }
      extend impl as IncidenceGraph { ... }
      extend impl as EdgeListGraph { ... }
    }
    ```

-   Implementing `Graph` out-of-line.

    ```carbon
    class MyEdgeListIncidenceGraph {
      extend impl as IncidenceGraph { ... }
      extend impl as EdgeListGraph { ... }
    }
    impl MyEdgeListIncidenceGraph as Graph {
      fn Source[self: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
      fn Target[self: Self](e: EdgeDescriptor) -> VertexDescriptor { ... }
    }
    ```

This last point means that there are situations where we can only detect a
missing method definition by the end of the file. This doesn't delay other
aspects of semantic checking, which will just assume that these methods will
eventually be provided.

**Open question:** We could require that the `impl` of the required interface be
declared lexically in the class scope in this case. That would allow earlier
detection of missing definitions.

### Use case: detecting unreachable matches

If interface `E` extends another interface `I`, that gives the information to
the compiler that the any type implementing `E` also implements `I`. This can be
used to detect unreachable code.

For example, the [`impl` prioritization rule](#prioritization-rule) is used to
pick between `impl` declarations based on an explicit priority ordering given by
the developer. If the broader interface `I` is prioritized over the more
specific interface `E`, the compiler can conclude that the more specific
declaration will never be selected and report an error. Similar situations could
be detected in function overloading.

## Adapting types

Since interfaces may only be implemented for a type once, and we limit where
implementations may be added to a type, there is a need to allow the user to
switch the type of a value to access different interface implementations. Carbon
therefore provides a way to create new types
[compatible with](terminology.md#compatible-types) existing types with different
APIs, in particular with different interface implementations, by
[adapting](terminology.md#adapting-a-type) them:

```carbon
interface Printable {
  fn Print[self: Self]();
}
interface Ordered {
  fn Less[self: Self](rhs: Self) -> bool;
}
class Song {
  extend impl as Printable { fn Print[self: Self]() { ... } }
}
class SongByTitle {
  adapt Song;
  extend impl as Ordered {
    fn Less[self: Self](rhs: Self) -> bool { ... }
  }
}
class FormattedSong {
  adapt Song;
  extend impl as Printable { fn Print[self: Self]() { ... } }
}
class FormattedSongByTitle {
  adapt Song;
  extend impl as Printable = FormattedSong;
  extend impl as Ordered = SongByTitle;
}
```

This allows developers to provide implementations of new interfaces (as in
`SongByTitle`), provide different implementations of the same interface (as in
`FormattedSong`), or mix and match implementations from other compatible types
(as in `FormattedSongByTitle`). The rules are:

-   You can add any declaration that you could add to a class except for
    declarations that would change the representation of the type. This means
    you can add methods, functions, interface implementations, and aliases, but
    not fields, base classes, or virtual functions. The specific implementations
    of virtual functions are part of the type representation, and so no virtual
    functions may be overridden in an adapter either.
-   The adapted type is compatible with the original type, and that relationship
    is an equivalence class, so all of `Song`, `SongByTitle`, `FormattedSong`,
    and `FormattedSongByTitle` end up compatible with each other.
-   Since adapted types are compatible with the original type, you may
    explicitly cast between them, but there is no implicit conversion between
    these types.

Inside an adapter, the `Self` type matches the adapter. Members of the original
type may be accessed either by a cast:

```carbon
class SongByTitle {
  adapt Song;
  extend impl as Ordered {
    fn Less[self: Self](rhs: Self) -> bool {
      return (self as Song).Title() < (rhs as Song).Title();
    }
  }
}
```

or using a qualified member access expression:

```carbon
class SongByTitle {
  adapt Song;
  extend impl as Ordered {
    fn Less[self: Self](rhs: Self) -> bool {
      return self.(Song.Title)() < rhs.(Song.Title)();
    }
  }
}
```

**Comparison with other languages:** This matches the Rust idiom called
"newtype", which is used to implement traits on types while avoiding
[coherence](terminology.md#coherence) problems, see
[here](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#using-the-newtype-pattern-to-implement-external-traits-on-external-types)
and
[here](https://github.com/Ixrec/rust-orphan-rules#user-content-why-are-the-orphan-rules-controversial).
Rust's mechanism doesn't directly support reusing implementations, though some
of that is provided by macros defined in libraries. Haskell has a
[`newtype` feature](https://wiki.haskell.org/Newtype) as well. Haskell's feature
doesn't directly support reusing implementations either, but the most popular
compiler provides it as
[an extension](https://ghc.gitlab.haskell.org/ghc/doc/users_guide/exts/newtype_deriving.html).

### Adapter compatibility

Consider a [type with a facet parameter, like a hash map](#parameterized-types):

```carbon
interface Hashable { ... }
class HashMap(KeyT:! Hashable, ValueT:! type) {
  fn Find[self: Self](key: KeyT) -> Optional(ValueT);
  // ...
}
```

A user of this type will provide specific values for the key and value types:

```carbon
class Song {
  extend impl as Hashable { ... }
  // ...
}

var play_count: HashMap(Song, i32) = ...;
var thriller_count: Optional(i32) =
    play_count.Find(Song("Thriller"));
```

Since the `KeyT` and `ValueT` are symbolic parameters, the `Find` function is a
checked generic, and it can only use the capabilities of `KeyT` and `ValueT`
specified as requirements. This allows us to evaluate when we can convert
between two different arguments to a parameterized type. Consider two adapters
of `Song` that implement `Hashable`:

```carbon
class PlayableSong {
  adapt Song;
  extend impl as Hashable = Song;
  extend impl as Media { ... }
}
class SongHashedByTitle {
  adapt Song;
  extend impl as Hashable { ... }
}
```

`Song` and `PlayableSong` have the same implementation of `Hashable` in addition
to using the same data representation. This means that it is safe to convert
between `HashMap(Song, i32)` and `HashMap(PlayableSong, i32)`, because the
implementation of all the methods will use the same implementation of the
`Hashable` interface. Carbon permits this conversion with an explicit cast.

On the other hand, `SongHashedByTitle` has a different implementation of
`Hashable` than `Song`. So even though `Song` and `SongHashedByTitle` are
compatible types, `HashMap(Song, i32)` and `HashMap(SongHashedByTitle, i32)` are
incompatible. This is important because we know that in practice the invariants
of a `HashMap` implementation rely on the hashing function staying the same.

### Extending adapter

Frequently we expect that the adapter type will want to preserve most or all of
the API of the original type. The two most common cases expected are adding and
replacing an interface implementation. Users would indicate that an adapter
starts from the original type's existing API by using the `extend` keyword
before `adapt`:

```carbon
class Song {
  extend impl as Hashable { ... }
  extend impl as Printable { ... }
}

class SongByArtist {
  extend adapt Song;

  // Add an implementation of a new interface
  extend impl as Ordered { ... }

  // Replace an existing implementation of an interface
  // with an alternative.
  extend impl as Hashable { ... }
}
```

The resulting type `SongByArtist` would:

-   implement `Ordered`, unlike `Song`,
-   implement `Hashable`, but differently than `Song`, and
-   implement `Printable`, inherited from `Song`.

The rule is that when looking up if `SongByArtist` implements an interface `I`
and no implementation is found, the compiler repeats the search to see if `Song`
implements `I`. If that is found, it is reused if possible. The reuse will be
successful if all types that reference `Self` in the signatures of interface's
functions can be cast to the corresponding type with `SongByArtist` substituted
in for `Song`.

Unlike the similar `class B { extend base: A; }` notation,
`class B { extend adapt A; }` is permitted even if `A` is a final class. Also,
there is no implicit conversion from `B` to `A`, matching `adapt` without
`extend` but unlike class extension.

To avoid or resolve name conflicts between interfaces, an `impl` may be declared
without [`extend`](#extend-impl). The names in that interface may then be pulled
in individually or renamed using `alias` declarations.

```carbon
class SongRenderToPrintDriver {
  extend adapt Song;

  // Add a new `Print()` member function.
  fn Print[self: Self]() { ... }

  // Avoid name conflict with new `Print`
  // function by implementing the `Printable`
  // interface without `extend`.
  impl as Printable = Song;

  // Make the `Print` function from `Printable`
  // available under the name `PrintToScreen`.
  alias PrintToScreen = Printable.Print;
}
```

### Use case: Using independent libraries together

Imagine we have two packages that are developed independently. Package
`CompareLib` defines an interface `CompareLib.Comparable` and a checked-generic
algorithm `CompareLib.Sort` that operates on types that implement
`CompareLib.Comparable`. Package `SongLib` defines a type `SongLib.Song`.
Neither has a dependency on the other, so neither package defines an
implementation for `CompareLib.Comparable` for type `SongLib.Song`. A user that
wants to pass a value of type `SongLib.Song` to `CompareLib.Sort` has to define
an adapter that provides an implementation of `CompareLib.Comparable` for
`SongLib.Song`. This adapter will probably use the
[`extend` facility of adapters](#extending-adapter) to preserve the
`SongLib.Song` API.

```carbon
import CompareLib;
import SongLib;

class Song {
  extend adapt SongLib.Song;
  extend impl as CompareLib.Comparable { ... }
}
// Or, to keep the names from CompareLib.Comparable out of Song's API:
class Song {
  extend adapt SongLib.Song;
}
impl Song as CompareLib.Comparable { ... }
// Or, equivalently:
class Song {
  extend adapt SongLib.Song;
  impl as CompareLib.Comparable { ... }
}
```

The caller can either convert `SongLib.Song` values to `Song` when calling
`CompareLib.Sort` or just start with `Song` values in the first place.

```carbon
var lib_song: SongLib.Song = ...;
CompareLib.Sort((lib_song as Song,));

var song: Song = ...;
CompareLib.Sort((song,));
```

### Use case: Defining an impl for use by other types

Let's say we want to provide a possible implementation of an interface for use
by types for which that implementation would be appropriate. We can do that by
defining an adapter implementing the interface that is parameterized on the type
it is adapting. That impl may then be pulled in using the `impl as ... = ...;`
syntax.

For example, given an interface `Comparable` for deciding which value is
smaller:

```carbon
interface Comparable {
  fn Less[self: Self](rhs: Self) -> bool;
}
```

We might define an adapter that implements `Comparable` for types that define
another interface `Difference`:

```carbon
interface Difference {
  fn Sub[self: Self](rhs: Self) -> i32;
}
class ComparableFromDifference(T:! Difference) {
  adapt T;
  extend impl as Comparable {
    fn Less[self: Self](rhs: Self) -> bool {
      return (self as T).Sub(rhs) < 0;
    }
  }
}
class IntWrapper {
  var x: i32;
  impl as Difference {
    fn Sub[self: Self](rhs: Self) -> i32 {
      return left.x - right.x;
    }
  }
  impl as Comparable = ComparableFromDifferenceFn(IntWrapper);
}
```

**TODO:** If we support function types, we could potentially pass a function to
use to the adapter instead:

```carbon
class ComparableFromDifferenceFn
    (T:! type, Difference:! fnty(T, T)->i32) {
  adapt T;
  extend impl as Comparable {
    fn Less[self: Self](rhs: Self) -> bool {
      return Difference(self as T, rhs as T) < 0;
    }
  }
}
class IntWrapper {
  var x: i32;
  fn Difference(left: Self, right: Self) {
    return left.x - right.x;
  }
  impl as Comparable =
      ComparableFromDifferenceFn(IntWrapper, Difference);
}
```

### Use case: Private impl

Adapter types can be used when a library publicly exposes a type, but only wants
to say that type implements an interface as a private detail internal to the
implementation of the type. In that case, instead of implementing the interface
for the public type, the library can create a private adapter for that type and
implement the interface on that instead. Any member of the class can cast its
`self` parameter to the adapter type when it wants to make use of the private
impl.

```carbon
// Public, in API file
class Complex64 {
  // ...
  fn CloserToOrigin[self: Self](them: Self) -> bool;
}

// Private

class ByReal {
  extend adapt Complex64;

  // Complex numbers are not generally comparable,
  // but this comparison function is useful for some
  // method implementations.
  extend impl as Comparable {
    fn Less[self: Self](that: Self) -> bool {
      return self.Real() < that.Real();
    }
  }
}

fn Complex64.CloserToOrigin[self: Self](them: Self) -> bool {
  var self_mag: ByReal = self * self.Conj() as ByReal;
  var them_mag: ByReal = them * them.Conj() as ByReal;
  return self_mag.Less(them_mag);
}
```

### Use case: Accessing interface names

Consider a case where a function will call several functions from an interface
that the type does not
[extend the implementation of](terminology.md#extending-an-impl).

```carbon
interface DrawingContext {
  fn SetPen[self: Self](...);
  fn SetFill[self: Self](...);
  fn DrawRectangle[self: Self](...);
  fn DrawLine[self: Self](...);
  ...
}
impl Window as DrawingContext { ... }
```

An adapter can make that more convenient by making a compatible type that does
extend the implementation of the interface. This avoids having to
[qualify](terminology.md#qualified-member-access-expression) each call to
methods in the interface.

```carbon
class DrawInWindow {
  adapt Window;
  extend impl as DrawingContext = Window;
}
fn Render(w: Window) {
  let d: DrawInWindow = w as DrawInWindow;
  d.SetPen(...);
  d.SetFill(...);
  d.DrawRectangle(...);
  ...
}
```

**Note:** Another way to achieve this is to use a
[local symbolic facet constant](#compile-time-let).

```carbon
fn Render(w: Window) {
  let DrawInWindow:! Draw = Window;
  // Implicit conversion to `w as DrawInWindow`.
  let d: DrawInWindow = w;
  d.SetPen(...);
  d.SetFill(...);
  d.DrawRectangle(...);
  ...
}
```

### Future work: Adapter with stricter invariants

**Future work:** Rust also uses the newtype idiom to create types with
additional invariants or other information encoded in the type
([1](https://doc.rust-lang.org/rust-by-example/generics/new_types.html),
[2](https://doc.rust-lang.org/book/ch19-04-advanced-types.html#using-the-newtype-pattern-for-type-safety-and-abstraction),
[3](https://www.worthe-it.co.za/blog/2020-10-31-newtype-pattern-in-rust.html)).
This is used to record in the type system that some data has passed validation
checks, like `ValidDate` with the same data layout as `Date`. Or to record the
units associated with a value, such as `Seconds` versus `Milliseconds` or `Feet`
versus `Meters`. We should have some way of restricting the casts between a type
and an adapter to address this use case. One possibility would be to add the
keyword `private` before `adapt`, so you might write
`extend private adapt Date;`.

## Associated constants

In addition to associated methods, we allow other kinds of
[associated entities](terminology.md#associated-entity). For consistency, we use
the same syntax to describe a compile-time constant in an interface as in a type
without assigning a value. As constants, they are declared using the `let`
introducer. For example, a fixed-dimensional point type could have the dimension
as an associated constant.

```carbon
interface NSpacePoint {
  let N:! i32;
  // The following require: 0 <= i < N.
  fn Get[addr self: Self*](i: i32) -> f64;
  fn Set[addr self: Self*](i: i32, value: f64);
  // Associated constants may be used in signatures:
  fn SetAll[addr self: Self*](value: Array(f64, N));
}
```

An implementation of an interface specifies values for associated constants with
a [`where` clause](#where-constraints). For example, implementations of
`NSpacePoint` for different types might have different values for `N`:

```carbon
class Point2D {
  extend impl as NSpacePoint where .N = 2 {
    fn Get[addr self: Self*](i: i32) -> f64 { ... }
    fn Set[addr self: Self*](i: i32, value: f64) { ... }
    fn SetAll[addr self: Self*](value: Array(f64, 2)) { ... }
  }
}

class Point3D {
  extend impl as NSpacePoint where .N = 3 {
    fn Get[addr self: Self*](i: i32) -> f64 { ... }
    fn Set[addr self: Self*](i: i32, value: f64) { ... }
    fn SetAll[addr self: Self*](value: Array(f64, 3)) { ... }
  }
}
```

Multiple assignments to associated constants may be joined using the `and`
keyword. The list of assignments is subject to two restrictions:

-   An implementation of an interface cannot specify a value for a
    [`final`](#final-members) associated constant.
-   If an associated constant doesn't have a
    [default value](#interface-defaults), every implementation must specify its
    value.

These values may be accessed as members of the type:

```carbon
Assert(Point2D.N == 2);
Assert(Point3D.N == 3);

fn PrintPoint[PointT:! NSpacePoint](p: PointT) {
  var i: i32 = 0
  while (i < PointT.N) {
    if (i > 0) { Print(", "); }
    Print(p.Get(i));
    ++i;
  }
}

fn ExtractPoint[PointT:! NSpacePoint](
    p: PointT,
    dest: Array(f64, PointT.N)*) {
  var i: i32 = 0;
  while (i < PointT.N) {
    (*dest)[i] = p.Get(i);
    ++i;
  }
}
```

**Comparison with other languages:** This feature is also called
[associated constants in Rust](https://doc.rust-lang.org/reference/items/associated-items.html#associated-constants).

**Aside:** The use of `:!` here means these `let` declarations will only have
compile-time and not runtime storage associated with them.

### Associated class functions

To be consistent with normal
[class function](/docs/design/classes.md#class-functions) declaration syntax,
associated class functions are written using a `fn` declaration:

```carbon
interface DeserializeFromString {
  fn Deserialize(serialized: String) -> Self;
}

class MySerializableType {
  var i: i32;

  extend impl as DeserializeFromString {
    fn Deserialize(serialized: String) -> Self {
      return {.i = StringToInt(serialized)};
    }
  }
}

var x: MySerializableType = MySerializableType.Deserialize("3");

fn Deserialize(T:! DeserializeFromString, serialized: String) -> T {
  return T.Deserialize(serialized);
}
var y: MySerializableType = Deserialize(MySerializableType, "4");
```

This is instead of declaring an associated constant using `let` with a function
type.

Together associated methods and associated class functions are called
_associated functions_, much like together methods and class functions are
called [member functions](/docs/design/classes.md#member-functions).

## Associated facets

Associated facets are [associated constants](#associated-constants) that happen
to have a [facet type](terminology.md#facet-type). These are particularly
interesting since they can be used in the signatures of associated methods or
functions, to allow the signatures of methods to vary from implementation to
implementation. We already have one example of this: the `Self` type discussed
[in the "Interfaces" section](#interfaces). For other cases, we can say that the
interface declares that each implementation will provide a facet constant under
a specified name. For example:

```carbon
interface StackAssociatedFacet {
  let ElementType:! type;
  fn Push[addr self: Self*](value: ElementType);
  fn Pop[addr self: Self*]() -> ElementType;
  fn IsEmpty[addr self: Self*]() -> bool;
}
```

Here we have an interface called `StackAssociatedFacet` which defines two
methods, `Push` and `Pop`. The signatures of those two methods declare them as
accepting or returning values with the type `ElementType`, which any implementer
of `StackAssociatedFacet` must also define. For example, maybe a `DynamicArray`
[parameterized type](#parameterized-types) implements `StackAssociatedFacet`:

```carbon
class DynamicArray(T:! type) {
  class IteratorType { ... }
  fn Begin[addr self: Self*]() -> IteratorType;
  fn End[addr self: Self*]() -> IteratorType;
  fn Insert[addr self: Self*](pos: IteratorType, value: T);
  fn Remove[addr self: Self*](pos: IteratorType);

  // Set the associated facet `ElementType` to `T`.
  extend impl as StackAssociatedFacet where .ElementType = T {
    fn Push[addr self: Self*](value: ElementType) {
      self->Insert(self->End(), value);
    }
    fn Pop[addr self: Self*]() -> ElementType {
      var pos: IteratorType = self->End();
      Assert(pos != self->Begin());
      --pos;
      returned var ret: ElementType = *pos;
      self->Remove(pos);
      return var;
    }
    fn IsEmpty[addr self: Self*]() -> bool {
      return self->Begin() == self->End();
    }
  }
}
```

The keyword `Self` can be used after the `as` in an `impl` declaration as a
shorthand for the type being implemented, including in the `where` clause
specifying the values of associated facets, as in:

```carbon
impl VeryLongTypeName as Add
    // `Self` here means `VeryLongTypeName`
    where .Result == Self {
  ...
}
```

> **Alternatives considered:** See
> [other syntax options considered in #731 for specifying associated facets](/proposals/p0731.md#syntax-for-associated-constants).
> In particular, it was deemed that
> [Swift's approach of inferring an associated facet from method signatures in the impl](https://docs.swift.org/swift-book/LanguageGuide/Generics.html#ID190)
> was unneeded complexity.

The definition of the `StackAssociatedFacet` is sufficient for writing a
checked-generic function that operates on anything implementing that interface,
for example:

```carbon
fn PeekAtTopOfStack[StackType:! StackAssociatedFacet](s: StackType*)
    -> StackType.ElementType {
  var top: StackType.ElementType = s->Pop();
  s->Push(top);
  return top;
}
```

Inside the checked-generic function `PeekAtTopOfStack`, the `ElementType`
associated facet member of `StackType` is an
[archetype](terminology.md#archetype), like other
[symbolic facet bindings](#symbolic-facet-bindings). This means
`StackType.ElementType` has the API dictated by the declaration of `ElementType`
in the interface `StackAssociatedFacet`.

Outside the checked-generic, associated facets have the concrete facet values
determined by impl lookup, rather than the erased version of that facet used
inside a checked-generic.

```carbon
var my_array: DynamicArray(i32) = (1, 2, 3);
// PeekAtTopOfStack's `StackType` is set to `DynamicArray(i32)`
// with `StackType.ElementType` set to `i32`.
Assert(PeekAtTopOfStack(my_array) == 3);
```

This is another part of achieving
[the goal that generic functions can be used in place of regular functions without changing the return type that callers see](goals.md#path-from-regular-functions)
discussed in the [return type section](#return-type).

Associated facets can also be implemented using a
[member type](/docs/design/classes.md#member-type).

```carbon
interface Container {
  let IteratorType:! Iterator;
  ...
}

class DynamicArray(T:! type) {
  ...
  extend impl as Container {
    class IteratorType {
      extend impl as Iterator { ... }
    }
    ...
  }
}
```

For context, see
["Interface parameters and associated constants" in the generics terminology document](terminology.md#interface-parameters-and-associated-constants).

**Comparison with other languages:** Both
[Rust](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#specifying-placeholder-types-in-trait-definitions-with-associated-types)
and [Swift](https://docs.swift.org/swift-book/LanguageGuide/Generics.html#ID189)
support these, but call them "associated types."

## Parameterized interfaces

Associated constants don't change the fact that a type can only implement an
interface at most once.

If instead you want a family of related interfaces, one per possible value of a
type parameter, multiple of which could be implemented for a single type, you
would use
[_parameterized interfaces_](terminology.md#interface-parameters-and-associated-constants),
also known as _generic interfaces_. To write a parameterized version of the
stack interface, instead of using associated constants, write a parameter list
after the name of the interface:

```carbon
interface StackParameterized(ElementType:! type) {
  fn Push[addr self: Self*](value: ElementType);
  fn Pop[addr self: Self*]() -> ElementType;
  fn IsEmpty[addr self: Self*]() -> bool;
}
```

Then `StackParameterized(Fruit)` and `StackParameterized(Veggie)` would be
considered different interfaces, with distinct implementations.

```carbon
class Produce {
  var fruit: DynamicArray(Fruit);
  var veggie: DynamicArray(Veggie);
  extend impl as StackParameterized(Fruit) {
    fn Push[addr self: Self*](value: Fruit) {
      self->fruit.Push(value);
    }
    fn Pop[addr self: Self*]() -> Fruit {
      return self->fruit.Pop();
    }
    fn IsEmpty[addr self: Self*]() -> bool {
      return self->fruit.IsEmpty();
    }
  }
  extend impl as StackParameterized(Veggie) {
    fn Push[addr self: Self*](value: Veggie) {
      self->veggie.Push(value);
    }
    fn Pop[addr self: Self*]() -> Veggie {
      return self->veggie.Pop();
    }
    fn IsEmpty[addr self: Self*]() -> bool {
      return self->veggie.IsEmpty();
    }
  }
}
```

Unlike associated constants in interfaces and parameters to types, interface
parameters can't be deduced. For example, if we were to rewrite
[the `PeekAtTopOfStack` example in the "associated facets" section](#associated-facets)
for `StackParameterized(T)` it would generate a compile error:

```carbon
// ❌ Error: can't deduce interface parameter `T`.
fn BrokenPeekAtTopOfStackParameterized
    [T:! type, StackType:! StackParameterized(T)]
    (s: StackType*) -> T { ... }
```

This error is because the compiler can not determine if `T` should be `Fruit` or
`Veggie` when passing in argument of type `Produce*`. Either `T` should be
replaced by a concrete type, like `Fruit`:

```carbon
fn PeekAtTopOfFruitStack
    [StackType:! StackParameterized(Fruit)]
    (s: StackType*) -> T { ... }

var produce: Produce = ...;
var top_fruit: Fruit =
    PeekAtTopOfFruitStack(&produce);
```

Or the value for `T` would be passed explicitly, using `where` constraints
described [in this section](#another-type-implements-parameterized-interface):

```carbon
fn PeekAtTopOfStackParameterizedImpl
    (T:! type, StackType:! StackParameterized(T), s: StackType*) -> T {
  ...
}
fn PeekAtTopOfStackParameterized[StackType:! type]
    (s: StackType*, T:! type where StackType is StackParameterized(T)) -> T {
  return PeekAtTopOfStackParameterizedImpl(T, StackType, s);
}

var produce: Produce = ...;
var top_fruit: Fruit =
    PeekAtTopOfStackParameterized(&produce, Fruit);
var top_veggie: Veggie =
    PeekAtTopOfStackParameterized(&produce, Veggie);
```

> **Note:** Alternative ways of declaraing `PeekAtTopOfStackParameterized` are
> described and discussed in
> [#578: Value patterns as function parameters](https://github.com/carbon-language/carbon-lang/issues/578).

Parameterized interfaces are useful for
[operator overloads](#operator-overloading). For example, the `EqWith(T)` and
`OrderedWith(T)` interfaces have a parameter that allows type to be comparable
with multiple other types, as in:

```carbon
interface EqWith(T:! type) {
  fn Equal[self: Self](rhs: T) -> bool;
  ...
}
class Complex {
  var real: f64;
  var imag: f64;
  // Can implement this interface more than once
  // as long as it has different arguments.
  extend impl as EqWith(f64) { ... }
  // Same as: impl as EqWith(Complex) { ... }
  extend impl as EqWith(Self) { ... }
}
```

All interface parameters must be marked as "symbolic", using the `:!` binding
pattern syntax. This reflects these two properties of these parameters:

-   They must be resolved at compile-time, and so can't be passed regular
    dynamic values.
-   We allow either symbolic or template values to be passed in.

**Future work:** We might also allow `template` bindings for interface
parameters, once we have a use case.

**Note:** Interface parameters aren't required to be facets, but that is the
vast majority of cases. As an example, if we had an interface that allowed a
type to define how the tuple-member-read operator would work, the index of the
member could be an interface parameter:

```carbon
interface ReadTupleMember(index:! u32) {
  let T:! type;
  // Returns self[index]
  fn Get[self: Self]() -> T;
}
```

This requires that the index be known at compile time, but allows different
indices to be associated with different values of `T`.

**Caveat:** When implementing an interface twice for a type, the interface
parameters are required to always be different. For example:

```carbon
interface Map(FromType:! type, ToType:! type) {
  fn Map[addr self: Self*](needle: FromType) -> Optional(ToType);
}
class Bijection(FromType:! type, ToType:! type) {
  extend impl as Map(FromType, ToType) { ... }
  extend impl as Map(ToType, FromType) { ... }
}
// ❌ Error: Bijection has two different impl definitions of
// interface Map(String, String)
var oops: Bijection(String, String) = ...;
```

In this case, it would be better to have an [adapting type](#adapting-types) to
contain the `impl` for the reverse map lookup, instead of implementing the `Map`
interface twice:

```carbon
class Bijection(FromType:! type, ToType:! type) {
  extend impl as Map(FromType, ToType) { ... }
}
class ReverseLookup(FromType:! type, ToType:! type) {
  adapt Bijection(FromType, ToType);
  extend impl as Map(ToType, FromType) { ... }
}
```

**Comparison with other languages:** Rust calls
[traits with parameters "generic traits"](https://doc.rust-lang.org/reference/items/traits.html#generic-traits)
and
[uses them for operator overloading](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#default-generic-type-parameters-and-operator-overloading).

[Rust uses the term "type parameters"](https://github.com/rust-lang/rfcs/blob/master/text/0195-associated-items.md#clearer-trait-matching)
for both interface facet parameters and associated facets. The difference is
that interface parameters are "inputs" since they _determine_ which `impl` to
use, and associated constants are "outputs" since they are determined _by_ the
`impl`, but play no role in selecting the `impl`.

### Parameterized named constraints

Carbon also allows the [named constraint](#named-constraints) construct to
support parameters. Those parameters work the same way as for interfaces.

## Where constraints

So far, we have restricted a [symbolic facet binding](#symbolic-facet-bindings)
by saying it has to implement an interface or a set of interfaces. There are a
variety of other constraints we would like to be able to express, such as
applying restrictions to associated constants. This is done using the `where`
operator that adds constraints to a [facet type](#facet-types).

The where operator can be applied to a facet type in a declaration context:

```carbon
// Constraints on generic function parameters:
fn F[V:! D where ...](v: V) { ... }

// Constraints on a class parameter:
class S(T:! B where ...) {
  // Constraints on a method:
  fn G[self: Self, V:! D where ...](v: V);
}

// Constraints on an interface parameter:
interface A(T:! B where ...) {
  // Constraints on an associated facet:
  let U:! C where ...;
  // Constraints on an associated method:
  fn G[self: Self, V:! D where ...](v: V);
}
```

We also allow you to name constraints using a `where` operator in a `let` or
`constraint` definition. The expressions that can follow the `where` keyword are
described in the ["kinds of `where` constraints"](#kinds-of-where-constraints)
section, but generally look like boolean expressions that should evaluate to
`true`.

The result of applying a `where` operator to a facet type is another facet type.
Note that this expands the kinds of requirements that facet types can have from
just interface requirements to also include the various kinds of constraints
discussed later in this section. In addition, it can introduce relationships
between different type variables, such as that a member of one is equal to a
member of another. The `where` operator is not associative, so a type expression
using multiple must use round parens `(`...`)` to specify grouping.

> **Comparison with other languages:** Both Swift and Rust use `where` clauses
> on declarations instead of in the expression syntax. These happen after the
> type that is being constrained has been given a name and use that name to
> express the constraint.
>
> Rust also supports
> [directly passing in the values for associated types](https://rust-lang.github.io/rfcs/0195-associated-items.html#constraining-associated-types)
> when using a trait as a constraint. This is helpful when specifying concrete
> types for all associated types in a trait in order to
> [make it object safe so it can be used to define a trait object type](https://rust-lang.github.io/rfcs/0195-associated-items.html#trait-objects).
>
> Rust is adding trait aliases
> ([RFC](https://github.com/rust-lang/rfcs/blob/master/text/1733-trait-alias.md),
> [tracking issue](https://github.com/rust-lang/rust/issues/41517)) to support
> naming some classes of constraints.

**References:** `where` constraints were added in proposal
[#818: Constraints for generics (generics details 3)](https://github.com/carbon-language/carbon-lang/pull/818).

### Kinds of `where` constraints

There are three kinds of `where` constraints, each of which uses a different
binary operator:

-   _Rewrite constraints_: `where`...`=`...
-   _Same-type constraints_: `where`...`==`...
-   _Implements constraints_: `where`...`impls`...

A rewrite constraint is written `where .A = B`, where `A` is the name of an
[associated constant](#associated-constants) which is rewritten to `B`.

The "dot followed by the name of a member" construct, like `.A`, is called a
_designator_. The name of the designator is looked up in the constraint, and
refers to the value of that member for whatever type is to satisfy this
constraint.

> **Concern:** Using `=` for this use case is not consistent with other `where`
> clauses that write a boolean expression that evaluates to `true` when the
> constraint is satisfied.

A same-type constraint is written `where X == Y`, where `X` and `Y` both name
facets. The constraint is that `X as type` must be the same as `Y as type`. In
cases where a constraint may be written in either form, prefer a rewrite
constraint over a same-type constraint. Note that switching between the two
forms does not change which types satisfies the constraint, and so is a
compatible change for callers.

An implements constraint is written `where T impls C`, where `T` is a facet and
`C` is a facet type. The constraint is that `T` satisfies the requirements of
`C`.

**References:** The definition of rewrite and same-type constraints were in
[proposal #2173](https://github.com/carbon-language/carbon-lang/pull/2173).
Implements constraints switched to using the `impls` keyword in
[proposal #2483](https://github.com/carbon-language/carbon-lang/pull/2483).

**Alternatives considered:**

-   [Different equality constraint operators for symbolic and constants](/proposals/p2173.md#status-quo)
-   [Single one-step equality constraint operators that merges constraints](/proposals/p2173.md#equal-types-with-different-interfaces)
-   [Restrict constraints to allow computable type equality](/proposals/p2173.md#restrict-constraints-to-allow-computable-type-equality)
-   [Find a fully transitive approach to type equality](/proposals/p2173.md#find-a-fully-transitive-approach-to-type-equality)
-   [Different syntax for rewrite constraint](/proposals/p2173.md#different-syntax-for-rewrite-constraint)
-   [Different syntax for same-type constraint](/proposals/p2173.md#different-syntax-for-same-type-constraint)
-   [Required ordering for rewrites](/proposals/p2173.md#required-ordering-for-rewrites)
-   [Multi-constraint `where` clauses](/proposals/p2173.md#multi-constraint-where-clauses)
-   [Rewrite constraints in `require` constraints](/proposals/p2173.md#rewrite-constraints-in-impl-as-constraints)

#### Recursive constraints

We sometimes need to constrain a type to equal one of its associated facets. In
this first example, we want to represent the function `Abs` which will return
`Self` for some but not all types, so we use an associated facet `MagnitudeType`
to encode the return type:

```carbon
interface HasAbs {
  extend Numeric;
  let MagnitudeType:! Numeric;
  fn Abs[self: Self]() -> MagnitudeType;
}
```

For types representing subsets of the real numbers, such as `i32` or `f32`, the
`MagnitudeType` will match `Self`, the type implementing an interface. For types
representing complex numbers, the types will be different. For example, the
`Abs()` function applied to a `Complex64` value would produce a `f32` result.
The goal is to write a constraint to restrict to the first case.

In a second example, when you take the slice of a type implementing `Container`
you get a type implementing `Container` which may or may not be the same type as
the original container type. However, taking the slice of a slice always gives
you the same type, and some functions want to only operate on containers whose
slice type is the same as the container type.

To solve this problem, we think of `Self` as an actual associated facet member
of every interface. We can then address it using `.Self` in a `where` clause,
like any other associated facet member.

```carbon
fn Relu[T:! HasAbs where .MagnitudeType = .Self](x: T) {
  // T.MagnitudeType == T so the following is allowed:
  return (x.Abs() + x) / 2;
}
fn UseContainer[T:! Container where .SliceType = .Self](c: T) -> bool {
  // T.SliceType == T so `c` and `c.Slice(...)` can be compared:
  return c == c.Slice(...);
}
```

Notice that in an interface definition, `Self` refers to the type implementing
this interface while `.Self` refers to the associated facet currently being
defined.

```carbon
interface Container;
constraint SliceConstraint(E:! type, S:! Container);

interface Container {
  let ElementType:! type;
  let IteratorType:! Iterator where .ElementType = ElementType;

  // `.Self` means `SliceType`.
  let SliceType:! Container where .Self is SliceConstraint(ElementType, .Self);

  // `Self` means the type implementing `Container`.
  fn GetSlice[addr self: Self*]
      (start: IteratorType, end: IteratorType) -> SliceType;
}

constraint SliceConstraint(E:! type, S:! Container) {
  extend Container where .ElementType = E and
                         .SliceType = S;
}
```

Note that [naming](#named-constraint-constants) a recursive constraint using the
[`constraint` introducer](#named-constraints) approach, we can name the
implementing type using `Self` instead of `.Self`, since they refer to the same
type. Note though they are different facets with different facet types:

```carbon
constraint RealAbs {
  extend HasAbs where .MagnitudeType = Self;
  // Satisfied by the same types:
  extend HasAbs where .MagnitudeType = .Self;

  // While `Self as type` is the same as `.Self as type`,
  // they are different as facets: `Self` has type
  // `RealAbs` and `.Self` has type `HasAbs`.
}

constraint ContainerIsSlice {
  extend Container where .SliceType = Self;
  // Satisfied by the same types:
  extend Container where .SliceType = .Self;

  // `Self` has type `ContainerIsSlice` and
  // `.Self` has type `Container`.
}
```

The `.Self` construct follows these rules:

-   `X :!` introduces `.Self:! type`, where references to `.Self` are resolved
    to `X`. This allows you to use `.Self` as an interface parameter as in
    `X:! I(.Self)`.
-   `A where` introduces `.Self:! A` and a `.Foo` _designator_ for each member
    `Foo` of `A`.
-   It's an error to reference `.Self` if it refers to more than one different
    thing or isn't a facet.
-   You get the innermost, most-specific type for `.Self` if it is introduced
    twice in a scope. By the previous rule, it is only legal if they all refer
    to the same facet binding.
-   `.Self` may not be on the left side of the `=` in a rewrite constraint.

So in `X:! A where ...`, `.Self` is introduced twice, after the `:!` and the
`where`. This is allowed since both times it means `X`. After the `:!`, `.Self`
has the type `type`, which gets refined to `A` after the `where`. In contrast,
it is an error if `.Self` could mean two different things, as in:

```carbon
// ❌ Illegal: `.Self` could mean `T` or `T.A`.
fn F[T:! InterfaceA where .A impls
           (InterfaceB where .B = .Self)](x: T);
```

These two meanings can be disambiguated by defining a
[`constraint`](#named-constraints):

```carbon
constraint InterfaceBWithSelf {
  extend InterfaceB where .B = Self;
}
constraint InterfaceBWith(U:! InterfaceA) {
  extend InterfaceB where .B = U;
}
// `T.A impls InterfaceB where .B = T.A`
fn F[T:! InterfaceA where .A impls InterfaceBWithSelf](x: T);
// `T.A impls InterfaceB where .B = T`
fn F[T:! InterfaceA where .A impls InterfaceBWith(.Self)](x: T);
```

#### Rewrite constraints

In a rewrite constraint, the left-hand operand of `=` must be a `.` followed by
the name of an associated constant. `.Self` is not permitted.

```carbon
interface RewriteSelf {
  // ❌ Error: `.Self` is not the name of an associated constant.
  let Me:! type where .Self = Self;
}
interface HasAssoc {
  let Assoc:! type;
}
interface RewriteSingleLevel {
  // ✅ Uses of `A.Assoc` will be rewritten to `i32`.
  let A:! HasAssoc where .Assoc = i32;
}
interface RewriteMultiLevel {
  // ❌ Error: Only one level of associated constant is permitted.
  let B:! RewriteSingleLevel where .A.Assoc = i32;
}
```

This notation is permitted anywhere a constraint can be written, and results in
a new constraint with a different interface: the named member effectively no
longer names an associated constant of the constrained type, and is instead
treated as a rewrite rule that expands to the right-hand side of the constraint,
with any mentioned parameters substituted into that type.

```carbon
interface Container {
  let Element:! type;
  let Slice:! Container where .Element = Element;
  fn Add[addr self: Self*](x: Element);
}
// `T.Slice.Element` rewritten to `T.Element`
//     because type of `T.Slice` says `.Element = Element`.
// `T.Element` rewritten to `i32`
//     because type of `T` says `.Element = i32`.
fn Add[T:! Container where .Element = i32](p: T*, y: T.Slice.Element) {
  // ✅ Argument `y` has the same type `i32` as parameter `x` of
  // `T.(Container.Add)`, which is also rewritten to `i32`.
  p->Add(y);
}
```

Rewrites aren't performed on the left-hand side of such an `=`, so
`where .A = .B and .A = C` is not rewritten to `where .A = .B and .B = C`.
Instead, such a `where` clause is invalid when the constraint is
[resolved](appendix-rewrite-constraints.md#rewrite-constraint-resolution) unless
each rule for `.A` specifies the same rewrite.

Note that `T:! C where .R = i32` can result in a type `T.R` whose behavior is
different from the behavior of `T.R` given `T:! C`. For example, member lookup
into `T.R` can find different results and operations can therefore have
different behavior. However, this does not violate
[coherence](/proposals/p2173.md#coherence) because the facet types `C` and
`C where .R = i32` don't differ by merely having more type information; rather,
they are different facet types that have an isomorphic set of values, somewhat
like `i32` and `u32`. An `=` constraint is not merely learning a new fact about
a type, it is requesting different behavior.

This approach has some good properties that
[same-type constraints](#same-type-constraints) have problems with:

-   [Equal types with different interfaces](/proposals/p2173.md#equal-types-with-different-interfaces):
    When an associated facet is constrained to be a concrete type, it is
    desirable for the associated facet to behave like that concrete type.
-   [Type canonicalization](/proposals/p2173.md#type-canonicalization): to
    enable efficient type equality.
-   [Transitivity of equality of types](/proposals/p2173.md#transitivity-of-equality)

The precise rules governing rewrite constraints are described in
[an appendix](appendix-rewrite-constraints.md).

#### Same-type constraints

A same-type constraint describes that two type expressions are known to evaluate
to the same value. Unlike a [rewrite constraint](#rewrite-constraints), however,
the two type expressions are treated as distinct types when type-checking a
symbolic expression that refers to them.

Same-type constraints are brought into scope, looked up, and resolved exactly as
if there were a `SameAs(U:! type)` interface and a `T == U` impl corresponded to
`T is SameAs(U)`, except that `==` is commutative.

Further, same-type equalities apply to type components, so that `X(A, B, C)` is
`SameType(X(D, E, F))` if we know that `A == D`, `B == E`, and `C == F`. Stated
differently, if `F` is any pure type function, `T impls SameAs(U)` implies
`F(T) impls SameAs(F(U))`. For example, if we know that `T == i32` then we also
have `Vector(T)` is single-step equal to `Vector(i32)`.

This relationship is not transitive, though, so it's not possible to ask for a
list of types that are the same as a given type, nor to ask whether there exists
a type that is the same as a given type and has some property. But it is
possible to ask whether two types are (non-transitively) known to be the same.

In order for same-type constraints to be useful, they must allow the two types
to be treated as actually being the same in some context. This can be
accomplished by the use of `==` constraints in an `impl`, such as in the
built-in implementation of `ImplicitAs`:

```carbon
final impl forall [T:! type, U:! type where .Self == T] T as ImplicitAs(U) {
  fn Convert[self: Self](other: U) -> U { ... }
}
```

> **Alternative considered:** It superficially seems like it would be convenient
> if such implementations were made available implicitly –- for example, by
> writing `impl forall [T:! type] T as ImplicitAs(T)` -– but in more complex
> examples that turns out to be problematic. Consider:
>
> ```carbon
> interface CommonTypeWith(U:! type) {
>   let Result:! type;
> }
> final impl forall [T:! type] T as CommonTypeWith(T) where .Result = T {}
>
> fn F[T:! Potato, U:! Hashable where .Self == T](x: T, y: U) -> auto {
>   // What is T.CommonTypeWith(U).Result? Is it T or U?
>  return (if cond then x else y).Hash();
> }
> ```
>
> With this alternative, `impl` validation for `T as CommonTypeWith(U)` fails:
> we cannot pick a common type when given two distinct type expressions, even if
> we know they evaluate to the same type, because we would not know which API
> the result should have.

##### Implementation of same-type `ImplicitAs`

It is possible to implement the above `impl` of `ImplicitAs` directly in Carbon,
without a compiler builtin, by taking advantage of the built-in conversion
between `C where .A = X` and `C where .A == X`:

```carbon
interface EqualConverter {
  let T:! type;
  fn Convert(t: T) -> Self;
}
fn EqualConvert[T:! type](t: T, U:! EqualConverter where .T = T) -> U {
  return U.Convert(t);
}
impl forall [U:! type] U as EqualConverter where .T = U {
  fn Convert(u: U) -> U { return u; }
}

impl forall [T:! type, U:! type where .Self == T] T as ImplicitAs(U) {
  fn Convert[self: Self]() -> U { return EqualConvert(self, U); }
}
```

The transition from `(T as ImplicitAs(U)).Convert`, where we know that `U == T`,
to `EqualConverter.Convert`, where we know that `.T = U`, allows a same-type
constraint to be used to perform a rewrite.

##### Manual type equality

A same-type constraint establishes
[type expressions](terminology.md#type-expression) are equal, and allows
implicit conversions between them. However, determining whether two type
expressions are _transitively_ equal is in general undecidable, as
[has been shown in Swift](https://forums.swift.org/t/swift-type-checking-is-undecidable/39024).

Carbon does not combine these equalities between type expressions. This means
that if two type expressions are only transitively equal, the user will need to
include a sequence of casts or use an
[`observe` declaration](#observe-declarations) to convert between them.

Given this interface `Transitive` that has associated facets that are
constrained to all be equal, with interfaces `P`, `Q`, and `R`:

```carbon
interface P { fn InP[self: Self](); }
interface Q { fn InQ[self: Self](); }
interface R { fn InR[self: Self](); }

interface Transitive {
  let A:! P;
  let B:! Q where .Self == A;
  let C:! R where .Self == B;

  fn GetA[self: Self]() -> A;
  fn TakesC[self: Self](c: C);
}
```

A cast to `B` is needed to call `TakesC` with a value of type `A`, so each step
only relies on one equality:

```carbon
fn F[T:! Transitive](t: T) {
  // ✅ Allowed
  t.TakesC(t.GetA() as T.B);

  // ✅ Allowed
  let b: T.B = t.GetA();
  t.TakesC(b);

  // ❌ Not allowed: t.TakesC(t.GetA());
}
```

The compiler may have several different `where` clauses to consider,
particularly when an interface has associated facets that recursively satisfy
the same interface, or mutual recursion between multiple interfaces. For
example, given these `Edge` and `Node` interfaces (similar to those defined in
[the section on interfaces with cyclic references](#example-of-declaring-interfaces-with-cyclic-references),
but using `==` same-type constraints):

```carbon
interface Edge;
interface Node;

private constraint EdgeFor(NodeT:! Node);
private constraint NodeFor(EdgeT:! Edge);

interface Edge {
  let N:! NodeFor(Self);
  fn GetN[self: Self]() -> N;
}
interface Node {
  let E:! EdgeFor(Self);
  fn GetE[self: Self]() -> E;
  fn AddE[addr self: Self*](e: E);
  fn NearN[self: Self](n: Self) -> bool;
}

constraint EdgeFor(NodeT:! Node) {
  extend Edge where .N == NodeT;
}
constraint NodeFor(EdgeT:! Edge) {
  extend Node where .E == EdgeT;
}
```

and a function `H` taking a value with some type implementing the `Node`
interface, then the following would be legal statements in `H`:

```carbon
fn H[N:! Node](n: N) {
  // ✅ Legal: argument has type `N.E`, matches parameter
  n.AddE(n.GetE());

  // ✅ Legal:
  // - argument has type `N.E.N`
  // - `N.E` has type `EdgeFor(Self)` where `Self`
  //   is `N`, which means `Edge where .N == N`
  // - so we have the constraint `N.E.N == N`
  // - which means the argument type `N.E.N`
  //   is equal to the parameter type `N` using a
  //   single `==` constraint.
  n.NearN(n.GetE().GetN());

  // ✅ Legal:
  // - type `N.E.N.E.N` may be cast to `N.E.N`
  //   using a single `where ==` clause, either
  //   `(N.E.N).E.N == (N).E.N` or
  //   `N.E.(N.E.N) == N.E.(N)`
  // - argument of type `N.E.N` may be passed to
  //   function expecting `N`, using a single
  //   `where ==` clause, as in the previous call.
  n.NearN(n.GetE().GetN().GetE().GetN() as N.E.N);
}
```

That last call would not be legal without the cast, though.

**Comparison with other languages:** Other languages such as Swift and Rust
instead perform automatic type equality. In practice this means that their
compiler can reject some legal programs based on heuristics simply to avoid
running for an unbounded length of time.

The benefits of the manual approach include:

-   fast compilation, since the compiler does not need to explore a potentially
    large set of combinations of equality restrictions, supporting
    [Carbon's goal of fast and scalable development](/docs/project/goals.md#fast-and-scalable-development);
-   expressive and predictable semantics, since there are no limitations on how
    complex a set of constraints can be supported; and
-   simplicity.

The main downsides are:

-   manual work for the source code author to prove to the compiler that types
    are equal; and
-   verbosity.

We expect that rich error messages and IDE tooling will be able to suggest
changes to the source code when a single equality constraint is not sufficient
to show two type expressions are equal, but a more extensive automated search
can find a sequence that prove they are equal.

##### Observe declarations

Same-type constraints are non-transitive, just like `ImplicitAs`. The developer
can use an `observe` declaration to bring a new same-type constraint into scope:

```carbon
observe A == B == C;
```

notionally does much the same thing as

```carbon
impl A as SameAs(C) { ... }
```

where the `impl` makes use of `A is SameAs(B)` and `B is SameAs(C)`.

In general, an `observe` declaration lists a sequence of
[type expressions](terminology.md#type-expression) that are equal by some
same-type `where` constraints. These `observe` declarations may be included in
an `interface` definition or a function body, as in:

```carbon
interface Edge {
  let N:! type;
}
interface Node {
  let E:! type;
}
interface Graph {
  let E:! Edge;
  let N:! Node where .E == E and E.N == .Self;
  observe E == N.E == E.N.E == N.E.N.E;
  // ...
}

fn H[G: Graph](g: G) {
  observe G.N == G.E.N == G.N.E.N == G.E.N.E.N;
  // ...
}
```

Every type expression after the first must be equal to some earlier type
expression in the sequence by a single `where` equality constraint. In this
example,

```carbon
fn J[G: Graph](g: G) {
  observe G.E.N == G.N.E.N == G.N == G.E.N.E.N;
  // ...
}
```

the expression `G.E.N.E.N` is one equality away from `G.N.E.N` and so it is
allowed. This is true even though `G.N.E.N` isn't the type expression
immediately prior to `G.E.N.E.N`.

After an `observe` declaration, all of the listed type expressions are
considered equal to each other using a single `where` equality. In this example,
the `observe` declaration in the `Transitive` interface definition provides the
link between associated facets `A` and `C` that allows function `F` to type
check.

```carbon
interface P { fn InP[self: Self](); }
interface Q { fn InQ[self: Self](); }
interface R { fn InR[self: Self](); }

interface Transitive {
  let A:! P;
  let B:! Q where .Self == A;
  let C:! R where .Self == B;

  fn GetA[self: Self]() -> A;
  fn TakesC[self: Self](c: C);

  // Without this `observe` declaration, the
  // calls in `F` below would not be allowed.
  observe A == B == C;
}

fn F[T:! Transitive](t: T) {
  var a: T.A = t.GetA();

  // ✅ Allowed: `T.A` values implicitly convert to
  // `T.C` using `observe` in interface definition.
  t.TakesC(a);

  // ✅ Allowed: `T.C` extends and implements `R`.
  (a as T.C).InR();
}
```

Only the current type is searched for interface implementations, so the call to
`InR()` would be illegal without the cast. However, an
[`observe`...`==`...`impls` declaration](#observing-equal-to-a-type-implementing-an-interface)
can be used to identify interfaces that must be implemented through some equal
type. This does not [extend](terminology.md#extending-an-impl) the API of the
type, that is solely determined by the definition of the type. Continuing the
previous example:

```carbon
fn TakesPQR[U:! P & Q & R](u: U);

fn G[T:! Transitive](t: T) {
  var a: T.A = t.GetA();

  // ✅ Allowed: `T.A` implements `P` and
  // includes its API, as if it extends `P`.
  a.InP();

  // ❌ Illegal: only the current type is
  // searched for interface implementations.
  a.(Q.InQ());

  // ✅ Allowed: values of type `T.A` may be cast
  // to `T.B`, which extends and implements `Q`.
  (a as T.B).InQ();

  // ✅ Allowed: `T.A` == `T.B` that implements `Q`.
  observe T.A == T.B impls Q;
  a.(Q.InQ());

  // ❌ Illegal: `T.A` still does not extend `Q`.
  a.InQ();

  // ✅ Allowed: `T.A` implements `P`,
  // `T.A` == `T.B` that implements `Q` (observe above),
  // and `T.A` == `T.C` that implements `R`.
  observe T.A == T.C impls R;
  TakesPQR(a);
}
```

Since adding an `observe`...`impls` declaration only adds non-extending
implementations of interfaces to symbolic facets, they may be added without
breaking existing code.

#### Implements constraints

An _implements constraint_ is written `where T impls C`, and expresses that the
facet `T` must implement the requirements of facet type `C`. This is more
flexible than using
[`&` to add a constraint](#combining-interfaces-by-anding-facet-types) since it
can be applied to [associated facet](#associated-facets) members as well.

In the following example, normally the `ElementType` of a `Container` can be any
type. The `SortContainer` function, however, takes a pointer to a type
satisfying `Container` with the additional constraint that its `ElementType`
must satisfy the `Ordered` interface, using an `impls` constraint:

```carbon
interface Container {
  let ElementType:! type;
  ...
}

fn SortContainer
    [ContainerType:! Container where .ElementType impls Ordered]
    (container_to_sort: ContainerType*);
```

In contrast to a [rewrite constraint](#rewrite-constraints) or a
[same-type constraint](#same-type-constraints), this does not say what type
`ElementType` exactly is, just that it must satisfy the requirements of some
facet type.

The specific case of a clause of the form
`where .AssociatedFacet impls AConstraint`, where the constraint is applied to a
direct associated facet member of the facet type being constrained (similar to
the restriction on [rewrite constraints](#rewrite-constraints)), gets special
treatment. In this case, the type of the associated facet is
[combined](#combining-interfaces-by-anding-facet-types) with the constraint. In
the above example, `Container` defines `ElementType` as having type `type`, but
`ContainerType.ElementType` has type `type & Ordered` (which is equivalent to
`Ordered`). This is because `ContainerType` has type
`Container where .ElementType impls Ordered`, not `Container`. This means we
need to be a bit careful when talking about the type of `ContainerType` when
there is a `where` clause modifying it.

> **Future work:** We may want to use a different operator in this case, such as
> `&=`, in place of `impls`, to reflect the change in the type. This is
> analogous to rewrite constraints using `=` instead of `==` to visibly reflect
> the different impact on the type.

An implements constraint can be applied to [`.Self`](#recursive-constraints), as
in `I where .Self impls C`. This has the same requirements as `I & C`, but that
`where` clause does not affect the API. This means that a
[symbolic facet binding](#symbolic-facet-bindings) with that facet type, so `T`
in `T:! I where .Self impls C`, is represented by an
[archetype](terminology.md#archetype) that implements both `I` and `C`, but only
[extends](terminology.md#extending-an-impl) `I`.

##### Implied constraints

Imagine we have a checked-generic function that accepts an arbitrary
[`HashMap` parameterized type](#parameterized-types):

```carbon
fn LookUp[KeyT:! type](hm: HashMap(KeyT, i32)*,
                       k: KeyT) -> i32;

fn PrintValueOrDefault[KeyT:! Printable,
                       ValueT:! Printable & HasDefault]
    (map: HashMap(KeyT, ValueT), key: KeyT);
```

The `KeyT` in these declarations does not visibly satisfy the requirements of
`HashMap`, which requires the type implement `Hashable` and other interfaces:

```carbon
class HashMap(
    KeyT:! Hashable & Eq & Movable,
    ...) { ... }
```

In this case, `KeyT` gets `Hashable` and so on as _implied constraints_.
Effectively that means that these functions are automatically rewritten to add a
`where .Self impls` constraint on `KeyT`:

```carbon
fn LookUp[
    KeyT:! type
        where .Self impls Hashable & Eq & Movable]
    (hm: HashMap(KeyT, i32)*, k: KeyT) -> i32;

fn PrintValueOrDefault[
    KeyT:! Printable
        where .Self impls Hashable & Eq & Movable,
    ValueT:! Printable & HasDefault]
    (map: HashMap(KeyT, ValueT), key: KeyT);
```

In this case, Carbon will accept the definition and infer the needed constraints
on the symbolic facet parameter. This is both more concise for the author of the
code and follows the
["don't repeat yourself" principle](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself).
This redundancy is undesirable since it means if the needed constraints for
`HashMap` are changed, then the code has to be updated in more locations.
Further it can add noise that obscures relevant information. In practice, any
user of these functions will have to pass in a valid `HashMap` instance, and so
will have already satisfied these constraints.

This implied constraint is equivalent to the explicit constraint that each
parameter and return type [is legal](#must-be-legal-type-argument-constraints).

> **Note:** These implied constraints affect the _requirements_ of a symbolic
> facet parameter, but not its _member names_. This way you can always look at
> the declaration to see how name resolution works, without having to look up
> the definitions of everything it is used as an argument to.

**Limitation:** To limit readability concerns and ambiguity, this feature is
limited to a single signature. Consider this interface declaration:

```carbon
interface GraphNode {
  let Edge:! type;
  fn EdgesFrom[self: Self]() -> HashSet(Edge);
}
```

One approach would be to say the use of `HashSet(Edge)` in the signature of the
`EdgesFrom` function would imply that `Edge` satisfies the requirements of an
argument to `HashSet`, such as being `Hashable`. Another approach would be to
say that the `EdgesFrom` would only be conditionally available when `Edge` does
satisfy the constraints on `HashSet` arguments. Instead, Carbon will reject this
definition, requiring the user to include all the constraints required for the
other declarations in the interface in the declaration of the `Edge` associated
facet. Similarly, a parameter to a class must be declared with all the
constraints needed to declare the members of the class that depend on that
parameter.

**Comparison with other languages:** Both Swift
([1](https://www.swiftbysundell.com/tips/inferred-generic-type-constraints/),
[2](https://github.com/apple/swift/blob/main/docs/Generics.rst#constraint-inference))
and
[Rust](https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=0b2d645bd205f24a7a6e2330d652c32e)
support some form of this feature as part of their type inference (and
[the Rust community is considering expanding support](http://smallcultfollowing.com/babysteps//blog/2022/04/12/implied-bounds-and-perfect-derive/#expanded-implied-bounds)).

#### Combining constraints

Constraints can be combined by separating constraint clauses with the `and`
keyword. This example expresses a constraint that two associated facets are
equal and satisfy an interface:

```carbon
fn EqualContainers
    [CT1:! Container,
     CT2:! Container where .ElementType impls HasEquality
                       and .ElementType = CT1.ElementType]
    (c1: CT1*, c2: CT2*) -> bool;
```

**Comparison with other languages:** Swift and Rust use commas `,` to separate
constraint clauses, but that only works because they place the `where` in a
different position in a declaration. In Carbon, the `where` is attached to a
type in a parameter list that is already using commas to separate parameters.

### Satisfying both facet types

If the two facet bindings being constrained to be equal, using either a
[rewrite constraint](#rewrite-constraints) or a
[same-type constraint](#same-type-constraints), have been declared with
different facet types, then the actual type value they are set to will have to
satisfy the requirements of both facet types. For example, if
`SortedContainer.ElementType` is declared to have a `Ordered` requirement, then
in these declarations:

```carbon
// With `=` rewrite constraint:
fn Contains_Rewrite
    [SC:! SortedContainer,
     CT:! Container where .ElementType = SC.ElementType]
    (haystack: SC, needles: CT) -> bool;

// With `==` same-type constraint:
fn Contains_SameType
    [SC:! SortedContainer,
     CT:! Container where .ElementType == SC.ElementType]
    (haystack: SC, needles: CT) -> bool;
```

the `where` constraints in both cases mean `CT.ElementType` must satisfy
`Ordered` as well. However, the behavior inside the body of these two inside the
body of the two functions is different.

In `Contains_Rewrite`, `CT.ElementType` is rewritten to `SC.ElementType` and
uses the facet type of `SC.ElementType`.

In `Contains_SameType`, the `where` clause does not affect the API of
`CT.ElementType`, and it would not even be considered to implement `Ordered`
unless there is some declaration like
`observe CT.ElementType == SC.ElementType impls Ordered`. Even then, the items
from the `needles` container won't directly have a `Compare` method member.

The rule is that a same-type `where` constraint between two type variables does
not modify the set of member names of either type. This is in contrast to
rewrite constraints like `where .ElementType = String` with a `=`, then
`.ElementType` is actually set to `String` including the complete `String` API.

Note that `==` constraints are symmetric, so the previous declaration of
`Contains_SameType` is equivalent to an alternative declaration where `CT` is
declared first and the `where` clause is attached to `SortedContainer`:

```carbon
fn Contains_SameType_Equivalent
    [CT:! Container,
     SC:! SortedContainer where .ElementType == CT.ElementType]
    (haystack: SC, needles: CT) -> bool;
```

### Constraints must use a designator

We don't allow a `where` constraint unless it applies a restriction to the
current type. This means referring to some
[designator](#kinds-of-where-constraints), like `.MemberName`, or
[`.Self`](#recursive-constraints). Examples:

-   `Container where .ElementType = i32`
-   `type where Vector(.Self) impls Sortable`
-   `Addable where i32 impls AddableWith(.Result)`

Constraints that only refer to other types should be moved to the type that is
declared last. So:

```carbon
// ❌ Error: `where A == B` does not use `.Self` or a designator
fn F[A:! type, B:! type, C:! type where A == B](a: A, b: B, c: C);
```

must be replaced by:

```carbon
// ✅ Allowed
fn F[A:! type, B:! type where A == .Self, C:! type](a: A, b: B, c: C);
```

This includes `where` clauses used in an `impl` declaration:

```carbon
// ❌ Error: `where T impls B` does not use `.Self` or a designator
impl forall [T:! type] T as A where T impls B {}
// ✅ Allowed
impl forall [T:! type where .Self impls B] T as A {}
// ✅ Allowed
impl forall [T:! B] T as A {}
```

This clarifies the meaning of the `where` clause and reduces the number of
redundant ways to express a restriction, following the
[one-way principle](/docs/project/principles/one_way.md).

**Alternative considered:** This rule was added in proposal
[#2376](https://github.com/carbon-language/carbon-lang/pull/2376), which
[considered whether this rule should be added](/proposals/p2376.md#alternatives-considered).

### Referencing names in the interface being defined

The constraint in a `where` clause is required to only reference earlier names
from this scope, as in this example:

```carbon
// ❌ Illegal: `E` references `V` declared later.
interface Graph {
  let E: Edge where .V = V;
  let V: Vert where .E = E;
}

// ✅ Allowed: Only references to earlier names.
interface Graph {
  let E: Edge;
  let V: Vert where .E = E and .Self == E.V;
}
```

### Constraint examples and use cases

-   **Set [associated constant](#associated-constants) to a constant:** For
    example in `NSpacePoint where .N = 2`, the associated constant `N` of
    `NSpacePoint` must be `2`. This syntax is also used to specify the values of
    associated constants when implementing an interface for a type, as in
    `impl MyPoint as NSpacePoint where .N = 2 {`...`}`.

-   **Set an [associated facet](#associated-facets) to a specific value:**
    Associated facets are treated like any other associated constant. So
    `Stack where .ElementType = i32` is a facet type that restricts to types
    that implement the `Stack` interface with integer elements, as in:

    ```carbon
    fn SumIntStack[T:! Stack where .ElementType = i32]
        (s: T*) -> i32 {
      var sum: i32 = 0;
      while (!s->IsEmpty()) {
        // s->Pop() returns a value of type
        // `T.ElementType` which is `i32`:
        sum += s->Pop();
      }
      return sum;
    }
    ```

    Note that this is a case that can use an `==` same-type constraint instead
    of an `=` rewrite constraint.

-   **One [associated constant](#associated-constants) must equal another:** For
    example with this definition of the interface `PointCloud`:

    ```carbon
    interface PointCloud {
      let Dim:! i32;
      let PointT:! NSpacePoint where .N = Dim;
    }
    ```

    an implementation of `PointCloud` for a type `T` will have
    `T.PointT.N == T.Dim`.

-   **Equal facet bindings:**

    For example, we could make the `ElementType` of an `Iterator` interface
    equal to the `ElementType` of a `Container` interface as follows:

    ```carbon
    interface Iterator {
      let ElementType:! type;
      ...
    }
    interface Container {
      let ElementType:! type;
      let IteratorType:! Iterator where .ElementType = ElementType;
      ...
    }
    ```

    In a function signature, this may be done by referencing an earlier
    parameter:

    ```carbon
    fn Map[CT:! Container,
           FT:! Function where .InputType = CT.ElementType]
          (c: CT, f: FT) -> Vector(FT.OutputType);
    ```

    In that example, `FT.InputType` is constrained to equal `CT.InputType`.
    Given an interface with two associated facets

    ```carbon
    interface PairInterface {
      let Left:! type;
      let Right:! type;
    }
    ```

    we can constrain them to be equal using
    `PairInterface where .Left = .Right`.

    Note that this is a case that can use an `==` same-type constraint instead
    of an `=` rewrite constraint.

-   **[Associated facet](#associated-facets) implements interface:** Given these
    definitions (omitting `ElementType` for brevity):

    ```carbon
    interface IteratorInterface { ... }
    interface ContainerInterface {
      let IteratorType:! IteratorInterface;
      // ...
    }
    interface RandomAccessIterator {
      extend IteratorInterface;
      // ...
    }
    ```

    We can then define a function that only accepts types that implement
    `ContainerInterface` where its `IteratorType` associated facet implements
    `RandomAccessIterator`:

    ```carbon
    fn F[ContainerType:! ContainerInterface
         where .IteratorType impls RandomAccessIterator]
        (c: ContainerType);
    ```

#### Parameterized type implements interface

There are times when a function will pass a
[symbolic facet parameter](#symbolic-facet-bindings) of the function as an
argument to a [parameterized type](#parameterized-types), and the function needs
the result to implement a specific interface.

```carbon
// A parameterized type
class DynArray(T:! type) { ... }

interface Printable { fn Print[self: Self](); }

// The parameterized type `DynArray` implements interface
// `Printable` only for some arguments.
impl DynArray(String) as Printable { ... }

// Constraint: `T` such that `DynArray(T)` implements `Printable`
fn PrintThree
    [T:! type where DynArray(.Self) impls Printable]
    (a: T, b: T, c: T) {
  // Create a `DynArray(T)` of size 3.
  var v: auto = DynArray(T).Make(a, b, c);
  // Known to be implemented due to the constraint on `T`.
  v.(Printable.Print)();
}

// ✅ Allowed: `DynArray(String)` implements `Printable`.
let s: String = "Ai ";
PrintThree(s, s, s);
// ❌ Forbidden: `DynArray(i32)` doesn't implement `Printable`.
let i: i32 = 3;
PrintThree(i, i, i);
```

**Comparison with other languages:** This use case was part of the
[Rust rationale for adding support for `where` clauses](https://rust-lang.github.io/rfcs/0135-where.html#motivation).

#### Another type implements parameterized interface

In this case, we need some other type to implement an interface parameterized by
a [symbolic facet parameter](#symbolic-facet-bindings). The syntax for this case
follows the previous case, except now the `.Self` parameter is on the interface
to the right of the `impls`. For example, we might need a type parameter `T` to
support explicit conversion from an `i32`:

```carbon
interface As(T:! type) {
  fn Convert[self: Self]() -> T;
}

fn Double[T:! Mul where i32 impls As(.Self)](x: T) -> T {
  return x * ((2 as i32) as T);
}
```

#### Must be legal type argument constraints

Now consider the case that the symbolic facet parameter is going to be used as
an argument to a [parameterized type](#parameterized-types) in a function body,
but not in the signature. If the parameterized type was explicitly mentioned in
the signature, the [implied constraint](#implied-constraints) feature would
ensure all of its requirements were met. To say a parameterized type is allowed
to be passed a specific argument, just write that it `impls type`, which all
types do. This is a trivial case of a
[parameterized type implements interface](#parameterized-type-implements-interface)
`where` constraint.

For example, a function that adds its parameters to a `HashSet` to deduplicate
them, needs them to be `Hashable` and so on. To say "`T` is a type where
`HashSet(T)` is legal," we can write:

```carbon
fn NumDistinct[T:! type where HashSet(.Self) impls type]
    (a: T, b: T, c: T) -> i32 {
  var set: HashSet(T);
  set.Add(a);
  set.Add(b);
  set.Add(c);
  return set.Size();
}
```

This has the same advantages over repeating the constraints on `HashSet`
arguments in the type of `T` as other
[implied constraints](#implied-constraints).

### Named constraint constants

A facet type with a `where` constraint, such as `C where <condition>`, can be
named two different ways:

-   Using `let template` as in:

    ```carbon
    let template NameOfConstraint:! auto = C where <condition>;
    ```

    or, since the type of a facet type is `type`:

    ```carbon
    let template NameOfConstraint:! type = C where <condition>;
    ```

-   Using a [named constraint](#named-constraints) with the `constraint` keyword
    introducer:

    ```carbon
    constraint NameOfConstraint {
      extend C where <condition>;
    }
    ```

Whichever approach is used, the result is `NameOfConstraint` is a compile-time
constant that is equivalent to `C where <condition>`.

## Other constraints as facet types

There are some constraints that Carbon naturally represents as named facet
types. These can either be used directly to constrain a facet binding, or in a
`where ... impls ...` [implements constraint](#implements-constraints) to
constrain an associated facet.

The compiler determines which types implement these interfaces, developers are
not permitted to explicitly implement these interfaces for their own types.

These facet types extend the requirements that facet types are allowed to
include beyond [interfaces implemented](#facet-types) and
[`where` clauses](#where-constraints).

**Open question:** Are these names part of the prelude or in a standard library?

### Is a derived class

Given a type `T`, `Extends(T)` is a facet type whose values are facets that are
(transitively) [derived from](/docs/design/classes.md#inheritance) `T`. That is,
`U:! Extends(T)` means `U` has an `extend base: T;` declaration, or there is a
chain of `extend base` declarations connecting `T` to `U`.

```carbon
base class BaseType { ... }

fn F[T:! Extends(BaseType)](p: T*);
fn UpCast[U:! type]
    (p: U*, V:! type where U impls Extends(.Self)) -> V*;
fn DownCast[X:! type](p: X*, Y:! Extends(X)) -> Y*;

class DerivedType {
  extend base: BaseType;
}
var d: DerivedType = {};
// `T` is set to `DerivedType`
// `DerivedType impls Extends(BaseType)`
F(&d);

// `U` is set to `DerivedType`
let p: BaseType* = UpCast(&d, BaseType);

// `X` is set to `BaseType`
// `Y` is set to facet `DerivedType as Extends(BaseType)`.
Assert(DownCast(p, DerivedType) == &d);
```

**Open question:** Alternatively, we could define a new `extends` operator for
use in `where` clauses:

```carbon
fn F[T:! type where .Self extends BaseType](p: T*);
fn UpCast[T:! type](p: T*, U:! type where T extends .Self) -> U*;
fn DownCast[T:! type](p: T*, U:! type where .Self extends T) -> U*;
```

**Comparison to other languages:** In Swift, you can
[add a required superclass to a type bound using `&`](https://docs.swift.org/swift-book/LanguageGuide/Protocols.html#ID282).

### Type compatible with another type

Given a type `U`, define the facet type `CompatibleWith(U)` as follows:

> `CompatibleWith(U)` is a facet type whose values are facets `T` such that
> `T as type` and `U as type` are
> [compatible types](terminology.md#compatible-types). That is values of `T` and
> `U` as types can be cast back and forth without any change in representation
> (for example `T` is an [adapter](#adapting-types) for `U`).

`CompatibleWith` determines an equivalence relationship between types.
Specifically, given two types `T1` and `T2`, they are equivalent if
`T1 impls CompatibleWith(T2)`, which is true if and only if
`T2 impls CompatibleWith(T1)`.

**Note:** Just like interface parameters, we require the user to supply `U`, it
may not be deduced. Specifically, this code would be illegal:

```carbon
fn Illegal[U:! type, T:! CompatibleWith(U)](x: T*) ...
```

In general there would be multiple choices for `U` given a specific `T` here,
and no good way of picking one. However, similar code is allowed if there is
another way of determining `U`:

```carbon
fn Allowed[U:! type, T:! CompatibleWith(U)](x: U*, y: T*) ...
```

#### Same implementation restriction

In some cases, we need to restrict to types that implement certain interfaces
the same way as the type `U`.

> The values of facet type `CompatibleWith(U, C)` are facets satisfying
> `CompatibleWith(U)` that have the same implementation of `C` as `U`.

For example, if we have a type `HashSet(T)`:

```carbon
class HashSet(T:! Hashable) { ... }
```

Then `HashSet(T)` may be cast to `HashSet(U)` if
`T impls CompatibleWith(U, Hashable)`. The one-parameter interpretation of
`CompatibleWith(U)` is recovered by letting the default for the second parameter
(`C`) be `type`.

#### Example: Multiple implementations of the same interface

This allows us to represent functions that accept multiple implementations of
the same interface for a type.

```carbon
choice CompareResult { Less, Equal, Greater }
interface Ordered {
  fn Compare[self: Self](rhs: Self) -> CompareResult;
}
fn CombinedLess[T:! type](a: T, b: T,
                          U:! CompatibleWith(T) & Ordered,
                          V:! CompatibleWith(T) & Ordered) -> bool {
  match ((a as U).Compare(b as U)) {
    case .Less => { return True; }
    case .Greater => { return False; }
    case .Equal => {
      return (a as V).Compare(b as V) == CompareResult.Less;
    }
  }
}
```

Used as:

```carbon
class Song { ... }
class SongByArtist { adapt Song; impl as Ordered { ... } }
class SongByTitle { adapt Song; impl as Ordered { ... } }
let s1: Song = ...;
let s2: Song = ...;
assert(CombinedLess(s1, s2, SongByArtist, SongByTitle) == True);
```

> **Open question:** We might generalize this to a list of implementations using
> variadics:
>
> ```carbon
> fn CombinedCompare[T:! type]
>     (a: T, b: T, ... each CompareT:! CompatibleWith(T) & Ordered)
>     -> CompareResult {
>   ... block {
>     let result: CompareResult =
>         (a as each CompareT).Compare(b as each CompareT);
>     if (result != CompareResult.Equal) {
>       return result;
>     }
>   }
>   return CompareResult.Equal;
> }
>
> assert(CombinedCompare(s1, s2, SongByArtist, SongByTitle)
>        == CompareResult.Less);
> ```
>
> However, [variadic support](#variadic-arguments) is still future work.

#### Example: Creating an impl out of other implementations

And then to package this functionality as an implementation of `Ordered`, we
combine `CompatibleWith` with [type adaptation](#adapting-types) and
[variadics](#variadic-arguments):

```carbon
class ThenCompare(
      T:! type,
      ... each CompareT:! CompatibleWith(T) & Ordered) {
  adapt T;
  extend impl as Ordered {
    fn Compare[self: Self](rhs: Self) -> CompareResult {
      ... block {
        let result: CompareResult =
            (self as each CompareT).Compare(rhs as each CompareT);
        if (result != CompareResult.Equal) {
          return result;
        }
      }
      return CompareResult.Equal;
    }
  }
}

let template SongByArtistThenTitle:! auto =
    ThenCompare(Song, SongByArtist, SongByTitle);
var s1: Song = ...;
var s2: SongByArtistThenTitle =
    ({ ... } as Song) as SongByArtistThenTitle;
assert((s1 as SongByArtistThenTitle).Compare(s2) ==
       CompareResult.Less);
```

### Sized types and facet types

What is the size of a type?

-   It could be fully known and fixed at compile time -- this is true of
    primitive types (`i32`, `f64`, and so on), most
    [classes](/docs/design/classes.md), and most other concrete types.
-   It could be known symbolically. This means that it will be known at codegen
    time, but not at type-checking time.
-   It could be dynamic. For example, it could be a
    [dynamic type](#runtime-type-fields), a slice, variable-sized type (such as
    [found in Rust](https://doc.rust-lang.org/nomicon/exotic-sizes.html#dynamically-sized-types-dsts)),
    or you could dereference a pointer to a base class that could actually point
    to a [derived class](/docs/design/classes.md#inheritance).
-   It could be unknown which category the type is in. In practice this will be
    essentially equivalent to having dynamic size.

A type is called _sized_ if it is in the first two categories, and _unsized_
otherwise. Note: something with size 0 is still considered "sized". The facet
type `Sized` is defined as follows:

> `Sized` is a type whose values are types `T` that are "sized" -- that is the
> size of `T` is known, though possibly only symbolically

Knowing a type is sized is a precondition to declaring variables of that type,
taking values of that type as parameters, returning values of that type, and
defining arrays of that type. Users will not typically need to express the
`Sized` constraint explicitly, though, since it will usually be a dependency of
some other constraint the type will need such as `Movable` or `Concrete`.

Example:

```carbon
// In the Carbon standard library
interface DefaultConstructible {
  // Types must be sized to be default constructible.
  require Self impls Sized;
  fn Default() -> Self;
}

// Classes are "sized" by default.
class Name {
  extend impl as DefaultConstructible {
    fn Default() -> Self { ... }
  }
  ...
}

fn F[T:! type](x: T*) {  // T is unsized.
  // ✅ Allowed: may access unsized values through a pointer.
  var y: T* = x;
  // ❌ Illegal: T is unsized.
  var z: T;
}

// T is sized, but its size is only known symbolically.
fn G[T: DefaultConstructible](x: T*) {
  // ✅ Allowed: T is default constructible, which means sized.
  var y: T = T.Default();
}

var z: Name = Name.Default();;
// ✅ Allowed: `Name` is sized and implements `DefaultConstructible`.
G(&z);
```

**Open question:** Should the `Sized` facet type expose an associated constant
with the size? So you could say `T.ByteSize` in the above example to get a
symbolic integer value with the size of `T`. Similarly you might say
`T.ByteStride` to get the number of bytes used for each element of an array of
`T`.

### Destructor constraints

There are four facet types related to
[the destructors of types](/docs/design/classes.md#destructors):

-   `Concrete` types may be local or member variables.
-   `Deletable` types may be safely deallocated by pointer using the `Delete`
    method on the `Allocator` used to allocate it.
-   `Destructible` types have a destructor and may be deallocated by pointer
    using the `UnsafeDelete` method on the correct `Allocator`, but it may be
    unsafe. The concerning case is deleting a pointer to a derived class through
    a pointer to its base class without a virtual destructor.
-   `TrivialDestructor` types have empty destructors. This facet type may be
    used with [specialization](#lookup-resolution-and-specialization) to unlock
    specific optimizations.

**Note:** The names `Deletable` and `Destructible` are
[**placeholders**](/proposals/p1154.md#type-of-type-naming) since they do not
conform to the decision on
[question-for-leads issue #1058: "How should interfaces for core functionality be named?"](https://github.com/carbon-language/carbon-lang/issues/1058).

The facet types `Concrete`, `Deletable`, and `TrivialDestructor` all extend
`Destructible`. Combinations of them may be formed using
[the `&` operator](#combining-interfaces-by-anding-facet-types). For example, a
checked-generic function that both instantiates and deletes values of a type `T`
would require `T` implement `Concrete & Deletable`.

Types are forbidden from explicitly implementing these facet types directly.
Instead they use
[`destructor` declarations in their class definition](/docs/design/classes.md#destructors)
and the compiler uses them to determine which of these facet types are
implemented.

## Compile-time `let`

A `let` statement inside a function body may be used to get the change in type
behavior of calling a checked-generic function without having to introduce a
function call.

```carbon
fn SymbolicLet(...) {
  ...
  let T:! C = U;
  X;
  Y;
  Z;
}
```

This introduces a symbolic constant `T` with type `C` and value `U`. This
implicitly includes an [`observe T == U;` declaration](#observe-declarations),
when `T` and `U` are facets, which allows values to implicitly convert from the
concrete type `U` to the erased type `T`, as in:

```carbon
let x: i32 = 7;
let T:! Add = i32;
// ✅ Allowed to convert `i32` values to `T`.
let y: T = x;
```

> **TODO:** The implied `observe` declaration is from question-for-leads issue
> [#996](https://github.com/carbon-language/carbon-lang/issues/996) and should
> be approved in a proposal.

This makes the `SymbolicLet` function roughly equivalent to:

```carbon
fn SymbolicLet(...) {
  ...
  fn Closure(T:! C where .Self == U) {
    X;
    Y;
    Z;
  }
  Closure(U);
}
```

The `where .Self == U` modifier captures the `observe` declaration introduced by
the `let` (at the cost of changing the type of `T`).

A symbolic `let` can be used to switch to the API of `C` when `U` does not
extend `C`, as an alternative to
[using an adapter](#use-case-accessing-interface-names), or to simplify inlining
of a generic function while preserving semantics.

To get a template binding instead of symbolic binding, add the `template`
keyword before the binding pattern, as in:

```carbon
fn TemplateLet(...) {
  ...
  let template T:! C = U;
  X;
  Y;
  Z;
}
```

which introduces a template constant `T` with type `C` and value `U`. This is
roughly equivalent to:

```carbon
fn TemplateLet(...) {
  ...
  fn Closure(template T:! C) {
    X;
    Y;
    Z;
  }
  Closure(U);
}
```

In this case, the `where .Self == U` modifier is superfluous.

> **References:**
>
> -   Proposal
>     [#950: Generics details 6: remove facets #950](https://github.com/carbon-language/carbon-lang/pull/950)
> -   Question-for-leads issue
>     [#996: Generic `let` with `auto`?](https://github.com/carbon-language/carbon-lang/issues/996)

## Parameterized impl declarations

There are cases where an `impl` definition should apply to more than a single
type and interface combination. The solution is to parameterize the `impl`
definition, so it applies to a family of types, interfaces, or both. This
includes:

-   Defining an `impl` that applies to multiple arguments to a
    [parameterized type](#parameterized-types).
-   _Conditional conformance_ where a parameterized type implements some
    interface if the parameter to the type satisfies some criteria, like
    implementing the same interface.
-   _Blanket_ `impl` declarations where an interface is implemented for all
    types that implement another interface, or some other criteria beyond being
    a specific type.
-   _Wildcard_ `impl` declarations where a family of interfaces are implemented
    for single type.

The syntax for an out-of-line parameterized `impl` declaration is:

<!-- prettier-ignore-start -->

<!-- The following triggers a bug in prettier where it adds an `>` -->

> `impl forall [`_<parameter-bindings>_`]` _<type-expression>_ `as`
> _<facet-type-expression> [_ `where` _<optional-rewrite-constraints> ]_ `;`

<!-- prettier-ignore-end -->

This may also be called a _generic `impl` declaration_.

### Impl for a parameterized type

Interfaces may be implemented for a [parameterized type](#parameterized-types).
This can be done lexically in the class's scope:

```carbon
class Vector(T:! type) {
  impl as Iterable where .ElementType = T {
    ...
  }
}
```

This is equivalent to naming the implementing type between `impl` and `as`,
though this form is not allowed after `extend`:

```carbon
class Vector(T:! type) {
  impl Vector(T) as Iterable where .ElementType = T {
    ...
  }
}
```

An out-of-line `impl` declaration must declare all parameters in a `forall`
clause:

```carbon
impl forall [T:! type] Vector(T) as Iterable
    where .ElementType = T {
  ...
}
```

The parameter for the type can be used as an argument to the interface being
implemented, with or without `extend`:

```carbon
class HashMap(KeyT:! Hashable, ValueT:! type) {
  extend impl as Has(KeyT) { ... }
  impl as Contains(HashSet(KeyT)) { ... }
}
```

or out-of-line the same `forall` parameter can be passed to both:

```carbon
class HashMap(KeyT:! Hashable, ValueT:! type) { ... }
impl forall [KeyT:! Hashable, ValueT:! type]
    HashMap(KeyT, ValueT) as Has(KeyT) { ... }
impl forall [KeyT:! Hashable, ValueT:! type]
    HashMap(KeyT, ValueT) as Contains(HashSet(KeyT)) { ... }
```

### Conditional conformance

[Conditional conformance](terminology.md#conditional-conformance) is expressing
that we have an `impl` of some interface for some type, but only if some
additional type restrictions are met. Examples where this would be useful
include being able to say that a container type, like `Vector`, implements some
interface when its element type satisfies the same interface:

-   A container is printable if its elements are.
-   A container could be compared to another container with the same element
    type using a
    [lexicographic comparison](https://en.wikipedia.org/wiki/Lexicographic_order)
    if the element type is comparable.
-   A container is copyable if its elements are.

This may be done by specifying a more specific implementing type to the left of
the `as` in the declaration:

```carbon
interface Printable {
  fn Print[self: Self]();
}
class Vector(T:! type) { ... }

// By saying "T:! Printable" instead of "T:! type" here,
// we constrain `T` to be `Printable` for this impl.
impl forall [T:! Printable] Vector(T) as Printable {
  fn Print[self: Self]() {
    for (let a: T in self) {
      // Can call `Print` on `a` since the constraint
      // on `T` ensures it implements `Printable`.
      a.Print();
    }
  }
}
```

Note that no `forall` clause or type may be specified when declaring an `impl`
with the [`extend`](#extend-impl) keyword:

```carbon
class Array(T:! type, template N:! i64) {
  // ❌ Illegal: nothing allowed before `as` after `extend impl`:
  extend impl forall [P:! Printable] Array(P, N) as Printable { ... }
}
```

**Note:** This was changed in
[proposal #2760](https://github.com/carbon-language/carbon-lang/pull/2760).

Instead, the class can declare aliases to members of the interface. Those
aliases will only be usable when the type implements the interface.

```carbon
class Array(T:! type, template N:! i64) {
  alias Print = Printable.Print;
}
impl forall [P:! Printable] Array(P, N) as Printable { ... }

impl String as Printable { ... }
var can_print: Array(String, 2) = ("Hello ", "world");
// ✅ Allowed: `can_print.Print` resolves to
// `can_print.(Printable.Print)`, which exists as long as
// `Array(String, 2) impls Printable`, which exists since
// `String impls Printable`.
can_print.Print();

var no_print: Array(Unprintable, 2) = ...;
// ❌ Illegal: `no_print.Print` resolves to
// `no_print.(Printable.Print)`, but there is no
// implementation of `Printable` for `Array(Unprintable, 2)`
// as long as `Unprintable` doesn't implement `Printable`.
no_print.Print();
```

It is legal to declare or define a conditional impl lexically inside the class
scope without `extend`, as in:

```carbon
class Array(T:! type, template N:! i64) {
  // ✅ Allowed: non-extending impl defined in class scope may
  // use `forall` and may specify a type.
  impl forall [P:! Printable] Array(P, N) as Printable { ... }
}
```

Inside the scope of this `impl` definition, both `P` and `T` refer to the same
type, but `P` has the facet type of `Printable` and so has a `Print` member. The
relationship between `T` and `P` is as if there was a
[`where P == T` clause](#same-type-constraints).

**Open question:** Need to resolve whether the `T` name can be reused, or if we
require that you need to use new names, like `P`, when creating new type
variables.

**Example:** Consider a type with two parameters, like `Pair(T, U)`. In this
example, the interface `Foo(T)` is only implemented when the two types are
equal.

```carbon
interface Foo(T:! type) { ... }
class Pair(T:! type, U:! type) { ... }
impl forall [T:! type] Pair(T, T) as Foo(T) { ... }
```

As before, you may also define the `impl` inline, but it may not be combined
with the `extend` keyword:

```carbon
class Pair(T:! type, U:! type) {
  impl Pair(T, T) as Foo(T) { ... }
}
```

**Clarification:** The same interface may be implemented multiple times as long
as there is no overlap in the conditions:

```carbon
class X(T:! type) {
  // ✅ Allowed: `X(T).F` consistently means `X(T).(Foo.F)`
  // even though that may have different definitions for
  // different values of `T`.
  alias F = Foo.F;
}
impl X(i32) as Foo {
  fn F[self: Self]() { DoOneThing(); }
}
impl X(i64) as Foo {
  fn F[self: Self]() { DoADifferentThing(); }
}
```

This allows a type to express that it implements an interface for a list of
types, possibly with different implementations. However, in general, `X(T).F`
can only mean one thing, regardless of `T`.

**Comparison with other languages:**
[Swift supports conditional conformance](https://github.com/apple/swift-evolution/blob/master/proposals/0143-conditional-conformances.md),
but bans cases where there could be ambiguity from overlap.
[Rust also supports conditional conformance](https://doc.rust-lang.org/rust-by-example/generics/where.html).

### Blanket impl declarations

A _blanket impl declaration_ is an `impl` declaration that could apply to more
than one root type, so the `impl` declaration will use a type variable for the
`Self` type. Here are some examples where blanket impl declarations arise:

-   Any type implementing `Ordered` should get an implementation of
    `PartiallyOrdered`.

    ```carbon
    impl forall [T:! Ordered] T as PartiallyOrdered { ... }
    ```

-   `T` implements `CommonType(T)` for all `T`

    ```carbon
    impl forall [T:! type] T as CommonType(T)
        where .Result = T { }
    ```

    This means that every type is the common type with itself.

Blanket impl declarations may never be declared using [`extend`](#extend-impl)
and must always be defined lexically [out-of-line](#out-of-line-impl).

#### Difference between a blanket impl and a named constraint

A blanket impl declaration can be used to say "any type implementing
`interface I` also implements `interface B`." Compare this with defining a
`constraint C` that requires `I`. In that case, `C` will also be implemented any
time `I` is. There are differences though:

-   There can be other implementations of `interface B` without a corresponding
    implementation of `I`, unless `B` has a requirement on `I`. However, the
    types implementing `C` will be the same as the types implementing `I`.
-   More specialized implementations of `B` can override the blanket
    implementation.

### Wildcard impl declarations

A _wildcard impl declaration_ is an `impl` declaration that defines how a family
of interfaces are implemented for a single `Self` type. For example, the
`BigInt` type might implement `AddTo(T)` for all `T` that implement
`ImplicitAs(i32)`. The implementation would first convert `T` to `i32` and then
add the `i32` to the `BigInt` value.

```carbon
class BigInt {
  impl forall [T:! ImplicitAs(i32)] as AddTo(T) { ... }
}
// Or out-of-line:
impl forall [T:! ImplicitAs(i32)] BigInt as AddTo(T) { ... }
```

Wildcard impl declarations may never be declared using [`extend`](#extend-impl),
to avoid having the names in the interface defined for the type multiple times.

### Combinations

The different kinds of parameters to an `impl` declarations may be combined. For
example, if `T` implements `As(U)`, then this implements `As(Optional(U))` for
`Optional(T)`:

```carbon
impl forall [U:! type, T:! As(U)]
  Optional(T) as As(Optional(U)) { ... }
```

This has a wildcard parameter `U`, and a condition on parameter `T`.

### Lookup resolution and specialization

As much as possible, we want rules for where an `impl` is allowed to be defined
and for selecting which `impl` definition to use that achieve these three goals:

-   Implementations have coherence, as
    [defined in terminology](terminology.md#coherence). This is
    [a goal for Carbon](goals.md#coherence). More detail can be found in
    [this appendix with the rationale and alternatives considered](appendix-coherence.md).
-   Libraries will work together as long as they pass their separate checks.
-   A checked-generic function can assume that some `impl` definition will be
    successfully selected if it can see an `impl` declaration that applies, even
    though another more specific `impl` definition may be selected.

For this to work, we need a rule that picks a single `impl` definition in the
case where there are multiple `impl` definitions that match a particular type
and interface combination. This is called _specialization_ when the rule is that
most specific implementation is chosen, for some definition of "specific."

#### Type structure of an impl declaration

Given an impl declaration, find the type structure by deleting deduced
parameters and replacing type parameters by a `?`. The type structure of this
declaration:

```carbon
impl forall [T:! ..., U:! ...] Foo(T, i32) as Bar(String, U) { ... }
```

is:

```carbon
impl Foo(?, i32) as Bar(String, ?)
```

To get a uniform representation across different `impl` definitions, before type
parameters are replaced the declarations are normalized as follows:

-   For `impl` declarations that are lexically inline in a class definition, the
    type is added between the `impl` and `as` keywords if the type is left out.
-   Pointer types `T*` are replaced with `Ptr(T)`.
-   The `extend` keyword is removed, if present.
-   The `forall` clause introducing type parameters is removed, if present.
-   Any `where` clauses that are setting associated constants or types are
    removed.

The type structure will always contain a single interface name, which is the
name of the interface being implemented, and some number of type names. Type
names can be in the `Self` type to the left of the `as` keyword, or as
parameters to other types or the interface. These names must always be defined
either in the current library or be publicly defined in some library this
library depends on.

#### Orphan rule

To achieve [coherence](terminology.md#coherence), we need to ensure that any
given impl can only be defined in a library that must be imported for it to
apply. Specifically, given a specific type and specific interface, `impl`
declarations that can match can only be in libraries that must have been
imported to name that type or interface. This is achieved with the _orphan
rule_.

**Orphan rule:** Some name from the type structure of an `impl` declaration must
be defined in the same library as the `impl`, that is some name must be _local_.

Let's say you have some interface `I(T, U(V))` being implemented for some type
`A(B(C(D), E))`. To satisfy the orphan rule for coherence, that `impl` must be
defined in some library that must be imported in any code that looks up whether
that interface is implemented for that type. This requires that `impl` is
defined in the same library that defines the interface or one of the names
needed by the type. That is, the `impl` must be defined with one of `I`, `T`,
`U`, `V`, `A`, `B`, `C`, `D`, or `E`. We further require anything looking up
this `impl` to import the _definitions_ of all of those names. Seeing a forward
declaration of these names is insufficient, since you can presumably see forward
declarations without seeing an `impl` with the definition. This accomplishes a
few goals:

-   The compiler can check that there is only one definition of any `impl` that
    is actually used, avoiding
    [One Definition Rule (ODR)](https://en.wikipedia.org/wiki/One_Definition_Rule)
    problems.
-   Every attempt to use an `impl` will see the exact same `impl` definition,
    making the interpretation and semantics of code consistent no matter its
    context, in accordance with the
    [low context-sensitivity principle](/docs/project/principles/low_context_sensitivity.md).
-   Allowing the `impl` to be defined with either the interface or the type
    partially addresses the
    [expression problem](https://eli.thegreenplace.net/2016/the-expression-problem-and-its-solutions).

Note that [the rules for specialization](#lookup-resolution-and-specialization)
do allow there to be more than one `impl` to be defined for a type, by
unambiguously picking one as most specific.

> **References:** Implementation coherence is
> [defined in terminology](terminology.md#coherence), and is
> [a goal for Carbon generics](goals.md#coherence). More detail can be found in
> [this appendix with the rationale and alternatives considered](appendix-coherence.md).

Only the implementing interface and types (self type and type parameters) in the
type structure are relevant here; an interface mentioned in a constraint is not
sufficient since it
[need not be imported](/proposals/p0920.md#orphan-rule-could-consider-interface-requirements-in-blanket-impls).

Since Carbon in addition requires there be no cyclic library dependencies, we
conclude that there is at most one library that can contain `impl` definitions
with a particular type structure.

#### Overlap rule

Given a specific concrete type, say `Foo(bool, i32)`, and an interface, say
`Bar(String, f32)`, the overlap rule picks, among all the matching `impl`
declarations, which type structure is considered "most specific" to use as the
implementation of that type for that interface.

Given two different type structures of impl declarations matching a query, for
example:

```carbon
impl Foo(?, i32) as Bar(String, ?)
impl Foo(?, ?) as Bar(String, f32)
```

We pick the type structure with a non-`?` at the first difference as most
specific. Here we see a difference between `Foo(?, i32)` and `Foo(?, ?)`, so we
select the one with `Foo(?, i32)`, ignoring the fact that it has another `?`
later in its type structure

This rule corresponds to a depth-first traversal of the type tree to identify
the first difference, and then picking the most specific choice at that
difference.

#### Prioritization rule

Since at most one library can contain `impl` definitions with a given type
structure, all `impl` definitions with a given type structure must be in the
same library. Furthermore by the [`impl` declaration access rules](#access),
they will be defined in the API file for the library if they could match any
query from outside the library. If there is more than one `impl` with that type
structure, they must be [defined](#implementing-interfaces) or
[declared](#declaring-implementations) together in a prioritization block. Once
a type structure is selected for a query, the first `impl` declaration in the
prioritization block that matches is selected.

> **Open question:** How are prioritization blocks written? A block starts with
> a keyword like `match_first` or `impl_priority` and then a sequence of impl
> declarations inside matching curly braces `{` ... `}`.
>
> ```carbon
> match_first {
>   // If T is Foo prioritized ahead of T is Bar
>   impl forall [T:! Foo] T as Bar { ... }
>   impl forall [T:! Baz] T as Bar { ... }
> }
> ```

To increase expressivity, Carbon allows prioritization blocks to contain a mix
of type structures, which is resolved using this rule:

> The compiler first picks the `impl` declaration with the type structure most
> favored for the query, and then picks the highest priority (earliest) matching
> `impl` declaration in the same prioritization block.

> **Alternatives considered:** We considered two other options:
>
> -   "Intersection rule:" Prioritization blocks implicitly define all non-empty
>     intersections of contained `impl` declarations, which are then selected by
>     their type structure.
> -   "Same type structure rule:" All the `impl` declarations in a
>     prioritization block are required to have the same type structure, at a
>     cost in expressivity. This option was not chosen since it wouldn't support
>     the different type structures created by the
>     [`like` operator](#like-operator-for-implicit-conversions).
>
> To see the difference from the first option, consider two libraries with type
> structures as follows:
>
> -   Library B has `impl (A, ?, ?, D) as I` and `impl (?, B, ?, D) as I` in the
>     same prioritization block.
> -   Library C has `impl (A, ?, C, ?) as I`.
>
> For the query `(A, B, C, D) as I`, using the intersection rule, library B is
> considered to have the intersection impl with type structure
> `impl (A, B, ?, D) as I` which is the most specific. If we instead just
> considered the rules mentioned explicitly, then `impl (A, ?, C, ?) as I` from
> library C is the most specific. The advantage of the implicit intersection
> rule is that if library B is changed to add an impl with type structure
> `impl (A, B, ?, D) as I`, it won't shift which library is serving that query.
> Ultimately we decided that it was too surprising to prioritize based on the
> implicit intersection of `impl` declarations, rather than something explicitly
> written in the code.
>
> We chose between these alternatives in
> [the open discussion on 2023-07-18](https://docs.google.com/document/d/1gnJBTfY81fZYvI_QXjwKk1uQHYBNHGqRLI2BS_cYYNQ/edit?resourcekey=0-ql1Q1WvTcDvhycf8LbA9DQ#heading=h.7jxges9ojgy3).
> **TODO:** This decision needs to be approved in a proposal.

#### Acyclic rule

A cycle is when a query, such as "does type `T` implement interface `I`?",
considers an `impl` declaration that might match, and whether that `impl`
declaration matches is ultimately dependent on whether that query is true. These
are cycles in the graph of (type, interface) pairs where there is an edge from
pair A to pair B if whether type A implements interface A determines whether
type B implements interface B.

The test for whether something forms a cycle needs to be precise enough, and not
erase too much information when considering this graph, that these `impl`
declarations are not considered to form cycles with themselves:

```carbon
impl forall [T:! Printable] Optional(T) as Printable;
impl forall [T:! type, U:! ComparableTo(T)] U as ComparableTo(Optional(T));
```

**Example:** If `T` implements `ComparableWith(U)`, then `U` should implement
`ComparableWith(T)`.

```carbon
impl forall [U:! type, T:! ComparableWith(U)]
    U as ComparableWith(T);
```

This is a cycle where which types implement `ComparableWith` determines which
types implement the same interface.

**Example:** Cycles can create situations where there are multiple ways of
selecting `impl` declarations that are inconsistent with each other. Consider an
interface with two blanket `impl` declarations:

```carbon
class Y {}
class N {}
interface True {}
impl Y as True {}
interface Z(T:! type) { let Cond:! type; }
match_first {
  impl forall [T:! type, U:! Z(T) where .Cond impls True] T as Z(U)
      where .Cond = N { }
  impl forall [T:! type, U:! type] T as Z(U)
      where .Cond = Y { }
}
```

What is `i8.(Z(i16).Cond)`? It depends on which of the two blanket impl
declarations are selected.

-   An implementation of `Z(i16)` for `i8` could come from the first blanket
    impl with `T == i8` and `U == i16` if `i16 impls Z(i8)` and
    `(i16 as Z(i8)).Cond == Y`. This condition is satisfied if `i16` implements
    `Z(i8)` using the second blanket impl. In this case,
    `(i8 as Z(i16)).Cond == N`.
-   Equally well `Z(i8)` could be implemented for `i16` using the first blanket
    impl and `Z(i16)` for `i8` using the second. In this case,
    `(i8 as Z(i16)).Cond == Y`.

There is no reason to to prefer one of these outcomes over the other.

**Example:** Further, cycles can create contradictions in the type system:

```carbon
class A {}
class B {}
class C {}
interface D(T:! type) { let Cond:! type; }
match_first {
  impl forall [T:! type, U:! D(T) where .Cond = B] T as D(U)
      where .Cond = C { }
  impl forall [T:! type, U:! D(T) where .Cond = A] T as D(U)
      where .Cond = B { }
  impl forall [T:! type, U:! type] T as D(U)
      where .Cond = A { }
}
```

What is `(i8 as D(i16)).Cond`? The answer is determined by which blanket impl is
selected to implement `D(i16)` for `i8`:

-   If the third blanket impl is selected, then `(i8 as D(i16)).Cond == A`. This
    implies that `(i16 as D(i8)).Cond == B` using the second blanket impl. If
    that is true, though, then our first impl choice was incorrect, since the
    first blanket impl applies and is higher priority. So
    `(i8 as D(i16)).Cond == C`. But that means that `i16 as D(i8)` can't use the
    second blanket impl.
-   For the second blanket impl to be selected, so `(i8 as D(i16)).Cond == B`,
    `(i16 as D(i8)).Cond` would have to be `A`. This happens when `i16`
    implements `D(i8)` using the third blanket impl. However,
    `(i8 as D(i16)).Cond == B` means that there is a higher priority
    implementation of `D(i8).Cond` for `i16`.

In either case, we arrive at a contradiction.

The workaround for this problem is to either split an interface in the cycle in
two, with a blanket implementation of one from the other, or move some of the
criteria into a [named constraint](#named-constraints).

**Concern:** Cycles could be spread out across libraries with no dependencies
between them. This means there can be problems created by a library that are
only detected by its users.

**Open question:** Should Carbon reject cycles in the absence of a query? The
two options here are:

-   Combining `impl` declarations gives you an immediate error if there exists
    queries using them that have cycles.
-   Only when a query reveals a cyclic dependency is an error reported.

**Open question:** In the second case, should we ignore cycles if they don't
affect the result of the query? For example, the cycle might be among
implementations that are lower priority.

#### Termination rule

It is possible to have a set of `impl` declarations where there isn't a cycle,
but the graph is infinite. Without some rule to prevent exhaustive exploration
of the graph, determining whether a type implements an interface could run
forever.

**Example:** It could be that `A` implements `B`, so `A impls B` if
`Optional(A) impls B`, if `Optional(Optional(A)) impls B`, and so on. This could
be the result of a single impl:

```carbon
impl forall [A:! type where Optional(.Self) impls B] A as B { ... }
```

This problem can also result from a chain of `impl` declarations, as in
`A impls B` if `A* impls C`, if `Optional(A) impls B`, and so on.

Determining whether a particular set of `impl` declarations terminates is
equivalent to the halting problem (content warning: contains many instances of
an obscene word as part of a programming language name
[1](https://sdleffler.github.io/RustTypeSystemTuringComplete/),
[2](https://forums.swift.org/t/two-more-undecidable-problems-in-the-swift-type-system/64814)),
and so is undecidable in general. Carbon adopts an approximation that guarantees
termination, but may mistakenly report an error when the query would terminate
if left to run long enough. The hope is that this criteria is accurate on code
that occurs in practice.

Rule: the types in the `impl` query must never get strictly more complicated
when considering the same `impl` declaration again. The way we measure the
complexity of a set of types is by counting how many of each base type appears.
A base type is the name of a type without its parameters. For example, the base
types in this query `Pair(Optional(i32), bool) impls AddWith(Optional(i32))`
are:

-   `Pair`
-   `Optional` twice
-   `i32` twice
-   `bool`
-   `AddWith`

A query is strictly more complicated if at least one count increases, and no
count decreases. So `Optional(Optional(i32))` is strictly more complicated than
`Optional(i32)` but not strictly more complicated than `Optional(bool)`.

This rule, when combined with [the acyclic rule](#acyclic-rule) that a query
can't repeat exactly,
[guarantees termination](/proposals/p2687.md#proof-of-termination).

Consider the example from before,

```carbon
impl forall [A:! type where Optional(.Self) impls B] A as B;
```

This `impl` declaration matches the query `i32 impls B` as long as
`Optional(i32) impls B`. That is a strictly more complicated query, though,
since it contains all the base types of the starting query (`i32` and `B`), plus
one more (`Optional`). As a result, an error can be given after one step, rather
than after hitting a large recursion limit. And that error can state explicitly
what went wrong: we went from a query with no `Optional` to one with one,
without anything else decreasing.

Note this only triggers a failure when the same `impl` declaration is considered
with the strictly more complicated query. For example, if the declaration is not
considered since there is a more specialized `impl` declaration that is
preferred by the [type-structure overlap rule](#overlap-rule), as in:

```
impl forall [A:! type where Optional(.Self) impls B] A as B;
impl Optional(bool) as B;
// OK, because we never consider the first `impl`
// declaration when looking for `Optional(bool) impls I`.
let U:! B = bool;
// Error: cycle with `i32 impls B` depending on
// `Optional(i32) impls B`, using the same `impl`
// declaration, as before.
let V:! B = i32;
```

<!-- prettier-ignore-start -->

<!-- The following triggers a bug in prettier where it adds an `>` -->

> **Note:**
> [Issue #2880](https://github.com/carbon-language/carbon-lang/issues/2880) is a
> tracking bug for known issues with this "strictly more complex" rule for
> `impl` termination. We are using that issue to track any code that arises in
> practice that would terminate but is rejected by this rule.

<!-- prettier-ignore-end -->

> **Comparison with other languages:** Rust solves this problem by imposing a
> recursion limit, much like C++ compilers use to terminate template recursion.
> This goes against
> [Carbon's goal of predictability in generics](goals.md#predictability),
> because of the concern that increasing the number of steps needed to resolve
> an `impl` query could cause far away code to hit the recursion limit.
>
> Carbon's approach is robust in the face of refactoring:
>
> -   It does not depend on the specifics of how an `impl` declaration is
>     parameterized, only on the query.
> -   It does not depend on the length of the chain of queries.
> -   It does not depend on a measure of type-expression complexity, like depth.
>
> Carbon's approach also results in identifying the minimal steps in the loop,
> which makes error messages as short and understandable as possible.

> **Alternatives considered:**
>
>     -   [Recursion limit](/proposals/p2687.md#problem)
>     -   [Measure complexity using type tree depth](/proposals/p2687.md#measure-complexity-using-type-tree-depth)
>     -   [Consider each type parameter in an `impl` declaration separately](/proposals/p2687.md#consider-each-type-parameter-in-an-impl-declaration-separately)
>     -   [Consider types in the interface being implemented as distinct](/proposals/p2687.md#consider-types-in-the-interface-being-implemented-as-distinct)
>     -   [Require some count to decrease](/proposals/p2687.md#require-some-count-to-decrease)
>     -   [Require non-type values to stay the same](/proposals/p2687.md#require-non-type-values-to-stay-the-same)

> **References:** This algorithm is from proposal
> [#2687: Termination algorithm for impl selection](https://github.com/carbon-language/carbon-lang/pull/2687),
> replacing the recursion limit originally proposed in
> [#920: Generic parameterized impls (details 5)](https://github.com/carbon-language/carbon-lang/pull/920)
> before we came up with this algorithm.

##### Non-facet arguments

For non-facet arguments we have to expand beyond base types to consider other
kinds of keys. These other keys are in a separate namespace from base types.

-   Values with an integral type use the name of the type as the key and the
    absolute value as a count. This means integer arguments are considered more
    complicated if they increase in absolute value. For example, if the values
    `2` and `-3` are used as arguments to parameters with type `i32`, then the
    `i32` key will have count `5`.
-   Every option of a choice type is its own key, counting how many times a
    value using that option occurs. Any parameters to the option are recorded as
    separate keys. For example, the `Optional(i32)` value of `.Some(7)` is
    recorded as keys `.Some` (with a count of `1`) and `i32` (with a count of
    `7`).
-   Yet another namespace of keys is used to track counts of variadic arguments,
    under the base type. This is to defend against having a variadic type `V`
    that takes any number of `i32` arguments, with an infinite set of distinct
    instantiations: `V(0)`, `V(0, 0)`, `V(0, 0, 0)`, ...
    -   A `tuple` key in this namespace is used to track the total number of
        components of tuple values. The values of those elements will be tracked
        using their own keys.

Non-facet argument values not covered by these cases are deleted from the query
entirely for purposes of the termination algorithm. This requires that two
queries that only differ by non-facet arguments are considered identical and
therefore are rejected by the acyclic rule. Otherwise, we could construct an
infinite family of non-facet argument values that could be used to avoid
termination.

### `final` impl declarations

There are cases where knowing that a parameterized impl won't be specialized is
particularly valuable. This could let the compiler know the return type of a
call to a generic function, such as using an operator:

```carbon
// Interface defining the behavior of the prefix-* operator
interface Deref {
  let Result:! type;
  fn Op[self: Self]() -> Result;
}

// Types implementing `Deref`
class Ptr(T:! type) {
  ...
  impl as Deref where .Result = T {
    fn Op[self: Self]() -> Result { ... }
  }
}
class Optional(T:! type) {
  ...
  impl as Deref where .Result = T {
    fn Op[self: Self]() -> Result { ... }
  }
}

fn F[T:! type](x: T) {
  // uses Ptr(T) and Optional(T) in implementation
}
```

The concern is the possibility of specializing `Optional(T) as Deref` or
`Ptr(T) as Deref` for a more specific `T` means that the compiler can't assume
anything about the return type of `Deref.Op` calls. This means `F` would in
practice have to add a constraint, which is both verbose and exposes what should
be implementation details:

```carbon
fn F[T:! type where Optional(T).(Deref.Result) == .Self
                and Ptr(T).(Deref.Result) == .Self](x: T) {
  // uses Ptr(T) and Optional(T) in implementation
}
```

To mark an impl as not able to be specialized, prefix it with the keyword
`final`:

```carbon
class Ptr(T:! type) {
  ...
  // Note: added `final`
  final impl as Deref where .Result = T {
    fn Op[self: Self]() -> Result { ... }
  }
}
class Optional(T:! type) {
  ...
  // Note: added `final`
  final impl as Deref where .Result = T {
    fn Op[self: Self]() -> Result { ... }
  }
}

// ❌ Illegal: impl Ptr(i32) as Deref { ... }
// ❌ Illegal: impl Optional(i32) as Deref { ... }
```

This prevents any higher-priority impl that overlaps a final impl from being
defined unless it agrees with the `final` impl on the overlap. Overlap is
computed between two non-`template` `impl` declaration by
[unifying](<https://en.wikipedia.org/wiki/Unification_(computer_science)>) the
corresponding parts. For example, the intersection of these two declarations

```carbon
final impl forall [T:! type]
    T as CommonTypeWith(T)
    where .Result = T {}

impl forall [V:! type, U:! CommonTypeWith(V)]
    Vec(U) as CommonTypeWith(Vec(V))
    where .Result = Vec(U.Result) {}
```

is found by unifying `T` with `Vec(U)` and `CommonTypeWith(T)` with
`CommonTypeWith(Vec(V))`. In this case, the intersection is when `T == Vec(U)`
and `U == V`. For templated `impl` declarations, overlap and agreement is
delayed until the template is instantiated with concrete types.

Since we do not require the compiler to compare the definitions of functions,
agreement is only possible for interfaces without any function members.

If the Carbon compiler sees a matching `final` impl, it can assume it won't be
specialized so it can use the assignments of the associated constants in that
impl definition.

```carbon
fn F[T:! type](x: T) {
  var p: Ptr(T) = ...;
  // *p has type `T`
  var o: Optional(T) = ...;
  // *o has type `T`
}
```

> **Alternatives considered:**
>
> -   [Allow interfaces with member functions to compare equal](/proposals/p2868.md#allow-interfaces-with-member-functions-to-compare-equal)
> -   Mark associated constants as `final` instead of an `impl` declaration, in
>     proposals
>     [#983](/proposals/p0983.md#final-associated-constants-instead-of-final-impls)
>     and
>     [#2868](/proposals/p2868.md#mark-associated-constants-as-final-instead-of-an-impl-declaration)
> -   [Prioritize a `final impl` over a more specific `impl` on the overlap](/proposals/p2868.md#prioritize-a-final-impl-over-a-more-specific-impl-on-the-overlap)

#### Libraries that can contain a `final` impl

To prevent the possibility of two unrelated libraries defining conflicting
`impl` declarations, Carbon restricts which libraries may declare an impl as
`final` to only:

-   the library declaring the impl's interface and
-   the library declaring the root of the `Self` type.

This means:

-   A blanket impl with type structure `impl ? as MyInterface(...)` may only be
    defined in the same library as `MyInterface`.
-   An impl with type structure `impl MyType(...) as MyInterface(...)` may be
    defined in the library with `MyType` or `MyInterface`.

These restrictions ensure that the Carbon compiler can locally check that no
higher-priority impl is defined superseding a `final` impl.

-   An impl with type structure `impl MyType(...) as MyInterface(...)` defined
    in the library with `MyType` must import the library defining `MyInterface`,
    and so will be able to see any final blanket impl declarations.
-   A blanket impl with type structure
    `impl ? as MyInterface(...ParameterType(...)...)` may be defined in the
    library with `ParameterType`, but that library must import the library
    defining `MyInterface`, and so will be able to see any `final` blanket impl
    declarations that might overlap. A final impl with type structure
    `impl MyType(...) as MyInterface(...)` would be given priority over any
    overlapping blanket impl defined in the `ParameterType` library.
-   An impl with type structure
    `impl MyType(...ParameterType(...)...) as MyInterface(...)` may be defined
    in the library with `ParameterType`, but that library must import the
    libraries defining `MyType` and `MyInterface`, and so will be able to see
    any `final` `impl` declarations that might overlap.

### Comparison to Rust

Rust has been designing a specialization feature, but it has not been completed.
Luckily, Rust team members have done a lot of blogging during their design
process, so Carbon can benefit from the work they have done. However, getting
specialization to work for Rust is complicated by the need to maintain
compatibility with existing Rust code. This motivates a number of Rust rules
where Carbon can be simpler. As a result there are both similarities and
differences between the Carbon design and Rust plans:

-   A Rust `impl` defaults to not being able to be specialized, with a `default`
    keyword used to opt-in to allowing specialization, reflecting the existing
    code base developed without specialization. Carbon `impl` declarations
    default to allowing specialization, with restrictions on which may be
    declared `final`.
-   Since a Rust impl is not specializable by default, generic functions can
    assume that if a matching blanket impl declaration is found, the associated
    constants from that impl will be used. In Carbon, if a checked-generic
    function requires an associated constant to have a particular value, the
    function commonly will need to state that using an explicit constraint.
-   Carbon will not have the "fundamental" attribute used by Rust on types or
    traits, as described in
    [Rust RFC 1023: "Rebalancing Coherence"](https://rust-lang.github.io/rfcs/1023-rebalancing-coherence.html).
-   Carbon will not use "covering" rules, as described in
    [Rust RFC 2451: "Re-Rebalancing Coherence"](https://rust-lang.github.io/rfcs/2451-re-rebalancing-coherence.html)
    and
    [Little Orphan Impls: The covered rule](http://smallcultfollowing.com/babysteps/blog/2015/01/14/little-orphan-impls/#the-covered-rule).
-   Like Rust, Carbon does use ordering, favoring the `Self` type and then the
    parameters to the interface in left-to-right order, see
    [Rust RFC 1023: "Rebalancing Coherence"](https://rust-lang.github.io/rfcs/1023-rebalancing-coherence.html)
    and
    [Little Orphan Impls: The ordered rule](http://smallcultfollowing.com/babysteps/blog/2015/01/14/little-orphan-impls/#the-ordered-rule),
    but the specifics are different.
-   Carbon is not planning to support any inheritance of implementation between
    `impl` definitions. This is more important to Rust since Rust does not
    support class inheritance for implementation reuse. Rust has considered
    multiple approaches here, see
    [Aaron Turon: "Specialize to Reuse"](http://aturon.github.io/tech/2015/09/18/reuse/)
    and
    [Supporting blanket impls in specialization](http://smallcultfollowing.com/babysteps/blog/2016/10/24/supporting-blanket-impls-in-specialization/).
-   [Supporting blanket impls in specialization](http://smallcultfollowing.com/babysteps/blog/2016/10/24/supporting-blanket-impls-in-specialization/)
    proposes a specialization rule for Rust that considers type structure before
    other constraints, as in Carbon, though the details differ.
-   Rust has more orphan restrictions to avoid there being cases where it is
    ambiguous which impl should be selected. Carbon instead has picked a total
    ordering on type structures, picking one as higher priority even without one
    being more specific in the sense of only applying to a subset of types.

## Forward declarations and cyclic references

Interfaces, named constraints, and their implementations may be forward declared
and then later defined. This is needed to allow cyclic references, for example
when declaring the edges and nodes of a graph. It is also a tool that may be
used to make code more readable.

The [interface](#interfaces), [named constraint](#named-constraints), and
[implementation](#implementing-interfaces) sections describe the syntax for
their _definition_, which consists of a declaration followed by a body contained
in curly braces `{` ... `}`. A _forward declaration_ is a declaration followed
by a semicolon `;`. A forward declaration is a promise that the entity being
declared will be defined later. Between the first declaration of an entity,
which may be in a forward declaration or the first part of a definition, and the
end of the definition the interface or implementation is called _incomplete_.
There are additional restrictions on how the name of an incomplete entity may be
used.

### Declaring interfaces and named constraints

The declaration for an interface or named constraint consists of:

-   an optional access-control keyword like `private`,
-   the keyword introducer `interface`, `constraint`, or `template constraint`,
-   the name of the interface or constraint, and
-   the parameter list, if any.

The name of an interface or constraint can not be used until its first
declaration is complete. In particular, it is illegal to use the name of the
interface in its parameter list. There is a
[workaround](#interfaces-with-parameters-constrained-by-the-same-interface) for
the use cases when this would come up.

An expression forming a constraint, such as `C & D`, is incomplete if any of the
interfaces or constraints used in the expression are incomplete.

An interface or named constraint may be forward declared subject to these rules:

-   The definition must be in the same file as the declaration.
-   Only the first declaration may have an access-control keyword.
-   An incomplete interface or named constraint may be used as constraints in
    declarations of types, functions, interfaces, or named constraints. This
    includes an `require` or `extend` declaration inside an interface or named
    constraint, but excludes specifying the values for associated constants
    because that would involve name lookup into the incomplete constraint.
-   An attempt to define the body of a generic function using an incomplete
    interface or named constraint in its signature is illegal.
-   An attempt to call a generic function using an incomplete interface or named
    constraint in its signature is illegal.
-   Any name lookup into an incomplete interface or named constraint is an
    error. For example, it is illegal to attempt to access a member of an
    interface using `MyInterface.MemberName` or constrain a member using a
    [`where` clause](#where-constraints).

If `C` is the name of an incomplete interface or named constraint, then it can
be used in the following contexts:

-   ✅ `T:! C`
-   ✅ `C & D`
    -   There may be conflicts between `C` and `D` making this invalid that will
        only be discovered once they are both complete.
-   ✅ `interface `...` { require` ... `impls C; }` or
    `constraint `...` { require` ... `impls C; }`
    -   Nothing implied by implementing `C` will be visible until `C` is
        complete.
-   ✅ `T:! C` ... `T impls C`
-   ✅ `T:! A & C` ... `T impls C`
    -   This includes constructs requiring `T impls C` such as `T as C` or
        `U:! C = T`.
-   ✅ `impl `...` as C;`
    -   Checking that all associated constants of `C` are correctly assigned
        values will be delayed until `C` is complete.

An incomplete `C` cannot be used in the following contexts:

-   ❌ `T:! C` ... `T.X`
-   ❌ `T:! C where `...
-   ❌ `class `...` { extend impl as C; }`
    -   The names of `C` are added to the class, and so those names need to be
        known.
-   ❌ `T:! C` ... `T impls A` where `A` is an interface or named constraint
    different from `C`
    -   Need to see the definition of `C` to see if it implies `A`.
-   ❌ `impl` ... `as C {` ... `}`

> **Future work:** It is currently undecided whether an interface needs to be
> complete to be extended, as in:
>
> ```carbon
> interface I { extend C; }
> ```
>
> There are three different approaches being considered:
>
> -   If we detect name collisions between the members of the interface `I` and
>     `C` when the interface `I` is defined, then we need `C` to be complete.
> -   If we instead only generate errors on ambiguous use of members with the
>     same name, as we do with `A & B`, then we don't need to require `C` to be
>     complete.
> -   Another option, being discussed in
>     [#2745](https://github.com/carbon-language/carbon-lang/issues/2745), is
>     that names in interface `I` shadow the names in any interface being
>     extended, then `C` would not be required to be complete.

### Declaring implementations

The declaration of an interface implementation consists of:

-   optional modifier keyword `final`,
-   the keyword introducer `impl`,
-   an optional `forall` followed by a deduced parameter list in square brackets
    `[`...`]`,
-   a type, including an optional [argument list](#parameterized-types),
-   the keyword `as`, and
-   a [facet type](#facet-types), including an optional
    [argument list](#parameterized-interfaces) and
    [`where` clause](#where-constraints) assigning
    [associated constants](#associated-constants) including
    [associated facets](#associated-facets).

**Note:** The `extend` keyword, when present, is not part of the `impl`
declaration. It precedes the `impl` declaration in class scope.

An implementation of an interface for a type may be forward declared, subject to
these rules:

-   The definition must be in the same library as the declaration. They must
    either be in the same file, or the declaration can be in the API file and
    the definition in an impl file. **Future work:** Carbon may require
    [parameterized `impl` definitions](#parameterized-impl-declarations) to be
    in the API file, to support separate compilation.
-   If there is both a forward declaration and a definition, only the first
    declaration must specify the assignment of associated constants with a
    `where` clause. Later declarations may omit the `where` clause by writing
    `where _` instead.
-   You can't forward declare an implementation of an incomplete interface. This
    allows the assignment of associated constants in the `impl` declaration to
    be verified with the declaration. An `impl` forward declaration may be for
    any declared type, whether it is incomplete or defined.
-   Every [extending implementation](#extend-impl) must be declared (or defined)
    inside the scope of the class definition. It may also be declared before the
    class definition or defined afterwards. Note that the class itself is
    incomplete in the scope of the class definition, but member function bodies
    defined inline are processed
    [as if they appeared immediately after the end of the outermost enclosing class](/docs/project/principles/information_accumulation.md#exceptions).
-   For [coherence](terminology.md#coherence), we require that any `impl`
    declaration that matches an impl lookup query in the same file, must be
    declared before the query. This can be done with a definition or a forward
    declaration. This matches the
    [information accumulation principle](/docs/project/principles/information_accumulation.md).

### Matching and agreeing

Carbon needs to determine if two declarations match in order to say which
definition a forward declaration corresponds to and to verify that nothing is
defined twice. Declarations that match must also agree, meaning they are
consistent with each other.

Interface and named constraint declarations match if their names are the same
after name and alias resolution. To agree:

-   The introducer keyword or keywords much be the same.
-   The types and order of parameters in the parameter list, if any, must match.
    The parameter names may be omitted, but if they are included in both
    declarations, they must match.
-   Types agree if they correspond to the same expression tree, after name and
    alias resolution and canonicalization of parentheses. Note that no other
    evaluation of expressions is performed.

Interface implementation declarations match if the type and interface
expressions match along with
[the `forall` clause](#parameterized-impl-declarations), if any:

-   If the type part is omitted, it is rewritten to `Self` in the context of the
    declaration.
-   `Self` is rewritten to its meaning in the scope it is used. In a class
    scope, this should match the type name and
    [optional parameter expression](#parameterized-types) after `class`. So in
    `class MyClass { ... }`, `Self` is rewritten to `MyClass`. In
    `class Vector(T:! Movable) { ... }`, `Self` is rewritten to
    `forall [T:! Movable] Vector(T)`.
-   Types match if they have the same name after name and alias resolution and
    the same parameters, or are the same type parameter.
-   Interfaces match if they have the same name after name and alias resolution
    and the same parameters. Note that a named constraint that is equivalent to
    an interface, as in `constraint Equivalent { extend MyInterface; }`, is not
    considered to match.

For implementations to agree:

-   The presence of the modifier keyword `final` before `impl` must match
    between a forward declaration and definition.
-   If either declaration includes a `where` clause, they must both include one.
    If neither uses `where _`, they must match in that they produce the
    associated constants with the same values considered separately.

### Declaration examples

```carbon
// Forward declaration of interfaces
interface Interface1;
interface Interface2;
interface Interface3;
interface Interface4;
interface Interface5;
interface Interface6;

// Forward declaration of class type
class MyClass;

// ❌ Illegal: Can't declare implementation of incomplete
//             interface.
// impl MyClass as Interface1;

// Definition of interfaces that were previously declared
interface Interface1 {
  let T1:! type;
}
interface Interface2 {
  let T2:! type;
}
interface Interface3 {
  let T3:! type;
}
interface Interface4 {
  let T4:! type;
}

// Out-of-line forward declarations
impl MyClass as Interface1 where .T1 = i32;
impl MyClass as Interface2 where .T2 = bool;
impl MyClass as Interface3 where .T3 = f32;
impl MyClass as Interface4 where .T4 = String;

interface Interface5 {
  let T5:! type;
}
interface Interface6 {
  let T6:! type;
}

// Definition of the previously declared class type
class MyClass {
  // Inline definition of previously declared impl.
  // Note: no need to repeat assignments to associated
  // constants.
  impl as Interface1 where _ { }

  // Inline extending definition of previously declared
  // impl.
  // Note: `extend` only appears on the declaration in
  // class scope
  // Note: allowed even though `MyClass` is incomplete.
  // Note: allowed but not required to repeat `where`
  // clause.
  extend impl as Interface3 where .T3 = f32 { }

  // Extending redeclaration of previously declared
  // impl. Every extending implementation must be
  // declared in the class definition.
  extend impl as Interface4 where _;

  // Inline forward declaration of implementation.
  impl MyClass as Interface5 where .T5 = u64;
  // or: impl as Interface5 where .T5 = u64;

  // Forward declaration of extending implementation.
  extend impl as Interface6 where .T6 = u8;
  // *Not*:
  //   extend impl MyClass as Interface6 where .T6 = u8;
  // No optional type after `extend impl`, it must be
  // followed immediately by `as`
}

// It would be legal to move the following definitions
// from the API file to the implementation file for
// this library.

// Definitions of previously declared implementations.
impl MyClass as Interface2 where _ { }
impl MyClass as Interface5 where _ { }

// Definition of previously declared extending
// implementations.
impl MyClass as Interface4 where _ { }
impl MyClass as Interface6 where _ { }
```

### Example of declaring interfaces with cyclic references

In this example, `Node` has an `EdgeT` associated facet that is constrained to
implement `Edge`, and `Edge` has a `NodeT` associated facet that is constrained
to implement `Node`. Furthermore, the `NodeT` of an `EdgeT` is the original
type, and the other way around. This is accomplished by naming and then forward
declaring the constraints that can't be stated directly:

```carbon
// Forward declare interfaces used in
// parameter lists of constraints.
interface Edge;
interface Node;

// Forward declare named constraints used in
// interface definitions.
private constraint EdgeFor(N:! Node);
private constraint NodeFor(E:! Edge);

// Define interfaces using named constraints.
interface Edge {
  let NodeT:! NodeFor(Self);
  fn Head[self: Self]() -> NodeT;
}
interface Node {
  let EdgeT:! EdgeFor(Self);
  fn Edges[self: Self]() -> DynArray(EdgeT);
}

// Now that the interfaces are defined, can
// refer to members of the interface, so it is
// now legal to define the named constraints.
constraint EdgeFor(N:! Node) {
  extend Edge where .NodeT = N;
}
constraint NodeFor(E:! Edge) {
  extend Node where .EdgeT = E;
}
```

> **Future work:** This approach has limitations. For example the compiler only
> knows `EdgeT` is convertible to `type` in the body of the `interface Node`
> definition, which may not be enough to satisfy the requirements to be an
> argument to `DynArray`. If this proves to be a problem, we may decided to
> expand what can be done with incomplete interfaces and types to allow the
> above to be written without the additional private constraints:
>
> ```carbon
> interface Node;
>
> interface Edge {
>   let NodeT:! Node where .EdgeT = Self;
>   fn Head[self: Self]() -> NodeT;
> }
>
> interface Node {
>   let EdgeT:! Movable & Edge where .NodeT = Self;
>   fn Edges[self: Self]() -> DynArray(EdgeT);
> }
> ```

### Interfaces with parameters constrained by the same interface

To work around
[the restriction about not being able to name an interface in its parameter list](#declaring-interfaces-and-named-constraints),
instead include that requirement in the body of the interface.

```carbon
// Want to require that `T` satisfies `CommonType(Self)`,
// but that can't be done in the parameter list.
interface CommonType(T:! type) {
  let Result:! type;
  // Instead add the requirement inside the definition.
  require T impls CommonType(Self);
}
```

Note however that `CommonType` is still incomplete inside its definition, so no
constraints on members of `CommonType` are allowed, and that this
`require T impls` declaration
[must involve `Self`](#interface-requiring-other-interfaces-revisited).

```carbon
interface CommonType(T:! type) {
  let Result:! type;
  // ❌ Illegal: `CommonType` is incomplete
  require T impls CommonType(Self) where .Result == Result;
}
```

Instead, a forward-declared named constraint can be used in place of the
constraint that can only be defined later. This is
[the same strategy used to work around cyclic references](#example-of-declaring-interfaces-with-cyclic-references).

```carbon
private constraint CommonTypeResult(T:! type, R:! type);

interface CommonType(T:! type) {
  let Result:! type;
  // ✅ Allowed: `CommonTypeResult` is incomplete, but
  //             no members are accessed.
  require T impls CommonTypeResult(Self, Result);
}

constraint CommonTypeResult(T:! type, R:! type) {
  extend CommonType(T) where .Result == R;
}
```

## Interface members with definitions

Interfaces may provide definitions for members, such as a function body for an
associated function or method or a value for an associated constant. If these
definitions may be overridden in implementations, they are called "defaults" and
prefixed with the `default` keyword. Otherwise they are called "final members"
and prefixed with the `final` keyword.

### Interface defaults

An interface may provide a default implementation of methods in terms of other
methods in the interface.

```carbon
interface Vector {
  fn Add[self: Self](b: Self) -> Self;
  fn Scale[self: Self](v: f64) -> Self;
  // Default definition of `Invert` calls `Scale`.
  default fn Invert[self: Self]() -> Self {
    return self.Scale(-1.0);
  }
}
```

A default function or method may also be defined out of line, later in the same
file as the interface definition:

```carbon
interface Vector {
  fn Add[self: Self](b: Self) -> Self;
  fn Scale[self: Self](v: f64) -> Self;
  default fn Invert[self: Self]() -> Self;
}
// `Vector` is considered complete at this point,
// even though `Vector.Invert` is still incomplete.
fn Vector.Invert[self: Self]() -> Self {
  return self.Scale(-1.0);
}
```

An impl of that interface for a type may omit a definition of `Invert` to use
the default, or provide a definition to override the default.

Interface defaults are helpful for [evolution](#evolution), as well as reducing
boilerplate. Defaults address the gap between the minimum necessary for a type
to provide the desired functionality of an interface and the breadth of API that
developers desire. As an example, in Rust the
[iterator trait](https://doc.rust-lang.org/std/iter/trait.Iterator.html) only
has one required method but dozens of "provided methods" with defaults.

Defaults may also be provided for associated constants, such as associated
facets, and interface parameters, using the `= <default value>` syntax.

```carbon
interface Add(Right:! type = Self) {
  default let Result:! type = Self;
  fn DoAdd[self: Self](right: Right) -> Result;
}

impl String as Add() {
  // Right == Result == Self == String
  fn DoAdd[self: Self](right: Self) -> Self;
}
```

Note that `Self` is a legal default value for an associated facet or facet
parameter. In this case the value of those names is not determined until `Self`
is, so `Add()` is equivalent to the constraint:

```carbon
// Equivalent to Add()
constraint AddDefault {
  extend Add(Self);
}
```

Note also that the parenthesis are required after `Add`, even when all
parameters are left as their default values.

More generally, default expressions may reference other associated constants or
`Self` as parameters to type constructors. For example:

```carbon
interface Iterator {
  let Element:! type;
  default let Pointer:! type = Element*;
}
```

Carbon does **not** support providing a default implementation of a required
interface.

```carbon
interface TotalOrder {
  fn TotalLess[self: Self](right: Self) -> bool;
  // ❌ Illegal: May not provide definition
  //             for required interface.
  require Self impls PartialOrder {
    fn PartialLess[self: Self](right: Self) -> bool {
      return self.TotalLess(right);
    }
  }
}
```

The workaround for this restriction is to use a
[blanket impl declaration](#blanket-impl-declarations) instead:

```carbon
interface TotalOrder {
  fn TotalLess[self: Self](right: Self) -> bool;
  // No `require` declaration, since implementers of
  // `TotalOrder` don't need to also implement
  // `PartialOrder`, since an implementation is provided.
}

// Any type that implements `TotalOrder` also has at
// least this implementation of `PartialOrder`:
impl forall [T:! TotalOrder] T as PartialOrder {
  fn PartialLess[self: Self](right: Self) -> bool {
    return self.TotalLess(right);
  }
}
```

Note that by the [orphan rule](#orphan-rule), this blanket impl must be defined
in the same library as `PartialOrder`.

**Comparison with other languages:** Rust supports specifying defaults for
[methods](https://doc.rust-lang.org/book/ch10-02-traits.html#default-implementations),
[interface parameters](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#default-generic-type-parameters-and-operator-overloading),
and
[associated constants](https://doc.rust-lang.org/reference/items/associated-items.html#associated-constants-examples).
Rust has found them valuable.

### `final` members

As an alternative to providing a definition of an interface member as a default,
members marked with the `final` keyword will not allow that definition to be
overridden in `impl` definitions.

```carbon
interface TotalOrder {
  fn TotalLess[self: Self](right: Self) -> bool;
  final fn TotalGreater[self: Self](right: Self) -> bool {
    return right.TotalLess(self);
  }
}

class String {
  extend impl as TotalOrder {
    fn TotalLess[self: Self](right: Self) -> bool { ... }
    // ❌ Illegal: May not provide definition of final
    //             method `TotalGreater`.
    fn TotalGreater[self: Self](right: Self) -> bool { ... }
  }
}

interface Add(T:! type = Self) {
  // `AddWith` *always* equals `T`
  final let AddWith:! type = T;
  // Has a *default* of `Self`
  default let Result:! type = Self;
  fn DoAdd[self: Self](right: AddWith) -> Result;
}
```

Final members may also be defined out-of-line:

```carbon
interface TotalOrder {
  fn TotalLess[self: Self](right: Self) -> bool;
  final fn TotalGreater[self: Self](right: Self) -> bool;
}
// `TotalOrder` is considered complete at this point, even
// though `TotalOrder.TotalGreater` is not yet defined.
fn TotalOrder.TotalGreater[self: Self](right: Self) -> bool {
 return right.TotalLess(self);
}
```

There are a few reasons for this feature:

-   When overriding would be inappropriate.
-   Matching the functionality of non-virtual methods in base classes, so
    interfaces can be a replacement for inheritance.
-   Potentially reduce dynamic dispatch when using the interface in a
    [`DynPtr`](#dynamic-types).

Note that this applies to associated entities, not interface parameters.

## Interface requiring other interfaces revisited

Recall that an
[interface can require another interface be implemented for the type](#interface-requiring-other-interfaces),
as in:

```carbon
interface Iterable {
  require Self impls Equatable;
  // ...
}
```

This states that the type implementing the interface `Iterable`, which in this
context is called `Self`, must also implement the interface `Equatable`. As is
done with [conditional conformance](#conditional-conformance), we allow another
type to be specified between `require` and `impls` to say some type other than
`Self` must implement an interface. For example,

```carbon
interface IntLike {
  require i32 impls As(Self);
  // ...
}
```

says that if `Self` implements `IntLike`, then `i32` must implement `As(Self)`.
Similarly,

```carbon
interface CommonTypeWith(T:! type) {
  require T impls CommonTypeWith(Self);
  // ...
}
```

says that if `Self` implements `CommonTypeWith(T)`, then `T` must implement
`CommonTypeWith(Self)`.

A `require`...`impls` constraint in an `interface`, or `constraint`, definition
must still use `Self` in some way. It can be an argument to either the
[type](#parameterized-types) or [interface](#parameterized-interfaces). For
example:

-   ✅ Allowed: `require Self impls Equatable`
-   ✅ Allowed: `require Vector(Self) impls Equatable`
-   ✅ Allowed: `require i32 impls CommonTypeWith(Self)`
-   ✅ Allowed: `require Self impls CommonTypeWith(Self)`
-   ❌ Error: `require i32 impls Equatable`
-   ❌ Error: `require T impls Equatable` where `T` is some parameter to the
    interface

This restriction allows the Carbon compiler to know where to look for facts
about a type. If `require i32 impls Equatable` could appear in any `interface`
definition, that implies having to search all of them when considering what
interfaces `i32` implements. This would create a
[coherence](terminology.md#coherence) problem, since then the set of facts true
for a type would depend on which interfaces have been imported.

When implementing an interface with an `require`...`impls` requirement, that
requirement must be satisfied by an implementation in an imported library, an
implementation somewhere in the same file, or a constraint in the impl
declaration. Implementing the requiring interface is a promise that the
requirement will be implemented. This is like a
[forward declaration of an impl](#declaring-implementations) except that the
definition can be broader instead of being required to match exactly.

```carbon
// `Iterable` requires `Equatable`, so there must be some
// impl of `Equatable` for `Vector(i32)` in this file.
impl Vector(i32) as Iterable { ... }

fn RequiresEquatable[T:! Equatable](x: T) { ... }
fn ProcessVector(v: Vector(i32)) {
  // ✅ Allowed since `Vector(i32)` is known to
  // implement `Equatable`.
  RequiresEquatable(v);
}

// Satisfies the requirement that `Vector(i32)` must
// implement `Equatable` since `i32 impls Equatable`.
impl forall [T:! Equatable] Vector(T) as Equatable { ... }
```

In some cases, the interface's requirement can be trivially satisfied by the
implementation itself, as in:

```carbon
impl forall [T:! type] T as CommonTypeWith(T) { ... }
```

Here is an example where the requirement of interface `Iterable` that the type
implements interface `Equatable` is satisfied by a constraint in the `impl`
declaration:

```carbon
class Foo(T:! type) {}
// This is allowed because we know that an `impl Foo(T) as Equatable`
// will exist for all types `T` for which this impl is used, even
// though there's neither an imported impl nor an impl in this file.
impl forall [T:! type where Foo(T) impls Equatable]
    Foo(T) as Iterable {}
```

This might be used to provide an implementation of `Equatable` for types that
already satisfy the requirement of implementing `Iterable`:

```carbon
class Bar {}
impl Foo(Bar) as Equatable {}
// Gives `Foo(Bar) impls Iterable` using the blanket impl of
// `Iterable` for `Foo(T)`.
```

### Requirements with `where` constraints

An interface implementation requirement with a `where` clause is harder to
satisfy. Consider an interface `B` that has a requirement that interface `A` is
also implemented.

```carbon
interface A(T:! type) {
  let Result:! type;
}
interface B(T:! type) {
  require Self impls A(T) where .Result == i32;
}
```

An implementation of `B` for a set of types can only be valid if there is a
visible implementation of `A` with the same `T` parameter for those types with
the `.Result` associated facet set to `i32`. That is
[not sufficient](/proposals/p1088.md#less-strict-about-requirements-with-where-clauses),
though, unless the implementation of `A` can't be specialized, either because it
is [marked `final`](#final-impl-declarations) or is not
[parameterized](#parameterized-impl-declarations). Implementations in other
libraries can't make `A` be implemented for fewer types, but can cause `.Result`
to have a different assignment.

## Observing a type implements an interface

An [`observe` declaration](#observe-declarations) can be used to show that two
types are equal so code can pass type checking without explicitly writing casts,
and without requiring the compiler to do a unbounded search that may not
terminate. An `observe` declaration can also be used to show that a type
implements an interface, in cases where the compiler will not work this out for
itself.

### Observing interface requirements

One situation where this occurs is when there is a chain of
[interfaces requiring other interfaces](#interface-requiring-other-interfaces-revisited).
During the `impl` validation done during type checking, Carbon will only
consider the interfaces that are direct requirements of the interfaces the type
is known to implement. An `observe`...`impls` declaration can be used to add an
interface that is a direct requirement to the set of interfaces whose direct
requirements will be considered for that type. This allows a developer to
provide a proof that there is a sequence of requirements that demonstrate that a
type implements an interface, as in this example:

```carbon
interface A { }
interface B { require Self impls A; }
interface C { require Self impls B; }
interface D { require Self impls C; }

fn RequiresA[T:! A](x: T);
fn RequiresC[T:! C](x: T);
fn RequiresD[T:! D](x: T) {
  // ✅ Allowed: `D` directly requires `C` to be implemented.
  RequiresC(x);

  // ❌ Illegal: No direct connection between `D` and `A`.
  // RequiresA(x);

  // `T impls D` and `D` directly requires `C` to be
  // implemented.
  observe T impls C;

  // `T impls C` and `C` directly requires `B` to be
  // implemented.
  observe T impls B;

  // ✅ Allowed: `T impls B` and `B` directly requires
  //             `A` to be implemented.
  RequiresA(x);
}
```

Note that `observe` statements do not affect which impl is selected during code
generation. For [coherence](terminology.md#coherence), the impl used for a
(type, interface) pair must always be the same, independent of context. The
[termination rule](#termination-rule) governs when compilation may fail when the
compiler can't determine the `impl` definition to select.

### Observing blanket impl declarations

An `observe`...`impls` declaration can also be used to observe that a type
implements an interface because there is a
[blanket impl declaration](#blanket-impl-declarations) in terms of requirements
a type is already known to satisfy. Without an `observe` declaration, Carbon
will only use blanket impl declarations that are directly satisfied.

```carbon
interface A { }
interface B { }
interface C { }
interface D { }

impl forall [T:! A] T as B { }
impl forall [T:! B] T as C { }
impl forall [T:! C] T as D { }

fn RequiresD(T:! D)(x: T);
fn RequiresB(T:! B)(x: T);

fn RequiresA(T:! A)(x: T) {
  // ✅ Allowed: There is a blanket implementation
  //             of `B` for types implementing `A`.
  RequiresB(x);

  // ❌ Illegal: No implementation of `D` for type
  //             `T` implementing `A`
  // RequiresD(x);

  // There is a blanket implementation of `B` for
  // types implementing `A`.
  observe T impls B;

  // There is a blanket implementation of `C` for
  // types implementing `B`.
  observe T impls C;

  // ✅ Allowed: There is a blanket implementation
  //             of `D` for types implementing `C`.
  RequiresD(x);
}
```

In the case of an error, a quality Carbon implementation will do a deeper search
for chains of requirements and blanket impl declarations and suggest `observe`
declarations that would make the code compile if any solution is found.

### Observing equal to a type implementing an interface

The [`observe`...`==` form](#observe-declarations) can be combined with the
`observe`...`impls` form to show that a type implements an interface because it
is equal to another type that is known to implement that interface.

```carbon
interface I {
  fn F();
}

fn G(T:! I, U:! type where .Self == T) {
  // ❌ Illegal: No implementation of `I` for `U`.
  U.(I.F)();

  // ✅ Allowed: Implementation of `I` for `U`
  //             through `T`.
  observe U == T impls I;
  U.(I.F)();

  // ❌ Illegal: `U` does not extend `I`.
  U.F();
}
```

Multiple `==` clauses are allowed in an `observe` declaration, so you may write
`observe A == B == C impls I;`.

## Operator overloading

Operations are overloaded for a type by implementing an interface specific to
that interface for that type. For example, types implement
[the `Negate` interface](/docs/design/expressions/arithmetic.md#extensibility)
to overload the unary `-` operator:

```carbon
// Unary `-`.
interface Negate {
  default let Result:! type = Self;
  fn Op[self: Self]() -> Result;
}
```

Expressions using operators are rewritten into calls to these interface methods.
For example, `-x` would be rewritten to `x.(Negate.Op)()`.

The interfaces and rewrites used for a given operator may be found in the
[expressions design](/docs/design/expressions/README.md).
[Question-for-leads issue #1058](https://github.com/carbon-language/carbon-lang/issues/1058)
defines the naming scheme for these interfaces, which was implemented in
[proposal #1178](https://github.com/carbon-language/carbon-lang/pull/1178).

### Binary operators

Binary operators will have an interface that is
[parameterized](#parameterized-interfaces) based on the second operand. For
example, to say a type may be converted to another type using an `as`
expression, implement the
[`As` interface](/docs/design/expressions/as_expressions.md#extensibility):

```carbon
interface As(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}
```

The expression `x as U` is rewritten to `x.(As(U).Convert)()`. Note that the
parameterization of the interface means it can be implemented multiple times to
support multiple operand types.

Unlike `as`, for most binary operators the interface's argument will be the
_type_ of the right-hand operand instead of its _value_. Consider
[the interface for a binary operator like `*`](/docs/design/expressions/arithmetic.md#extensibility):

```carbon
// Binary `*`.
interface MulWith(U:! type) {
  default let Result:! type = Self;
  fn Op[self: Self](other: U) -> Result;
}
```

A use of binary `*` in source code will be rewritten to use this interface:

```carbon
var left: Meters = ...;
var right: f64 = ...;
var result: auto = left * right;
// Equivalent to:
var equivalent: left.(MulWith(f64).Result)
    = left.(MulWith(f64).Op)(right);
```

Note that if the types of the two operands are different, then swapping the
order of the operands will result in a different implementation being selected.
It is up to the developer to make those consistent when that is appropriate. The
standard library will provide [adapters](#adapting-types) for defining the
second implementation from the first, as in:

```carbon
interface OrderedWith(U:! type) {
  fn Compare[self: Self](u: U) -> Ordering;
  // ...
}

class ReverseComparison(T:! type, U:! OrderedWith(T)) {
  adapt T;
  extend impl as OrderedWith(U) {
    fn Compare[self: Self](u: U) -> Ordering {
      match (u.Compare(self)) {
        case .Less         => return .Greater;
        case .Equivalent   => return .Equivalent;
        case .Greater      => return .Less;
        case .Incomparable => return .Incomparable;
      }
    }
  }
}

impl SongByTitle as OrderedWith(SongTitle) { ... }
impl SongTitle as OrderedWith(SongByTitle)
    = ReverseComparison(SongTitle, SongByTitle);
```

In some cases the reverse operation may not be defined. For example, a library
might support subtracting a vector from a point, but not the other way around.

Further note that even if the reverse implementation exists,
[the `impl` prioritization rule](#prioritization-rule) might not pick it. For
example, if we have two types that support comparison with anything implementing
an interface that the other implements:

```carbon
interface IntLike {
  fn AsInt[self: Self]() -> i64;
}

class EvenInt { ... }
impl EvenInt as IntLike;
impl EvenInt as OrderedWith(EvenInt);
// Allow `EvenInt` to be compared with anything that
// implements `IntLike`, in either order.
impl forall [T:! IntLike] EvenInt as OrderedWith(T);
impl forall [T:! IntLike] T as OrderedWith(EvenInt);

class PositiveInt { ... }
impl PositiveInt as IntLike;
impl PositiveInt as OrderedWith(PositiveInt);
// Allow `PositiveInt` to be compared with anything that
// implements `IntLike`, in either order.
impl forall [T:! IntLike] PositiveInt as OrderedWith(T);
impl forall [T:! IntLike] T as OrderedWith(PositiveInt);
```

Then the compiler will favor selecting the implementation based on the type of
the left-hand operand:

```carbon
var even: EvenInt = ...;
var positive: PositiveInt = ...;
// Uses `EvenInt as OrderedWith(T)` impl
if (even < positive) { ... }
// Uses `PositiveInt as OrderedWith(T)` impl
if (positive > even) { ... }
```

### `like` operator for implicit conversions

Because the type of the operands is directly used to select the operator
interface implementation, there are no automatic implicit conversions, unlike
with function or method calls. Given both a method and an interface
implementation for multiplying by a value of type `f64`:

```carbon
class Meters {
  fn Scale[self: Self](s: f64) -> Self;
}
// "Implementation One"
impl Meters as MulWith(f64)
    where .Result = Meters {
  fn Op[self: Self](other: f64) -> Result {
    return self.Scale(other);
  }
}
```

the method will work with any argument that can be implicitly converted to `f64`
but the operator overload will only work with values that have the specific type
of `f64`:

```carbon
var height: Meters = ...;
var scale: f32 = 1.25;
// ✅ Allowed: `scale` implicitly converted
//             from `f32` to `f64`.
var allowed: Meters = height.Scale(scale);
// ❌ Illegal: `Meters` doesn't implement
//             `MulWith(f32)`.
var illegal: Meters = height * scale;
```

The workaround is to define a parameterized implementation that performs the
conversion. The implementation is for types that implement the
[`ImplicitAs` interface](/docs/design/expressions/implicit_conversions.md#extensibility).

```carbon
// "Implementation Two"
impl forall [T:! ImplicitAs(f64)]
    Meters as MulWith(T) where .Result = Meters {
  fn Op[self: Self](other: T) -> Result {
    // Carbon will implicitly convert `other` from type
    // `T` to `f64` to perform this call.
    return self.((Meters as MulWith(f64)).Op)(other);
  }
}
// ✅ Allowed: uses `Meters as MulWith(T)` impl
//             with `T == f32` since `f32 impls ImplicitAs(f64)`.
var now_allowed: Meters = height * scale;
```

Observe that the [prioritization rule](#prioritization-rule) will still prefer
the unparameterized impl when there is an exact match.

To reduce the boilerplate needed to support these implicit conversions when
defining operator overloads, Carbon has the `like` operator. This operator can
only be used in the type or facet type part of an `impl` declaration, as part of
a forward declaration or definition, in a place of a type.

```carbon
// Notice `f64` has been replaced by `like f64`
// compared to "implementation one" above.
impl Meters as MulWith(like f64)
    where .Result = Meters {
  fn Op[self: Self](other: f64) -> Result {
    return self.Scale(other);
  }
}
```

This `impl` definition actually defines two implementations. The first is the
same as this definition with `like f64` replaced by `f64`, giving something
equivalent to "implementation one". The second implementation replaces the
`like f64` with a parameter that ranges over types that can be implicitly
converted to `f64`, equivalent to "implementation two".

> **Note:** We have decided to change the following in
> [a discussion on 2023-07-13](https://docs.google.com/document/d/1gnJBTfY81fZYvI_QXjwKk1uQHYBNHGqRLI2BS_cYYNQ/edit?resourcekey=0-ql1Q1WvTcDvhycf8LbA9DQ#heading=h.rs7m0kytcl4t).
> The new approach is to have one parameterized implementation replacing all of
> the `like` expressions on the left of the `as`, and another replacing all of
> the `like` expressions on the right of the `as`. However, in
> [a discussion on 2023-07-20](https://docs.google.com/document/d/1gnJBTfY81fZYvI_QXjwKk1uQHYBNHGqRLI2BS_cYYNQ/edit?resourcekey=0-ql1Q1WvTcDvhycf8LbA9DQ#heading=h.msdqbemd6axi),
> we decided that this change would not affect how we handle nested `like`
> expressions: `like Vector(like i32)` is still `like Vector(i32)` plus
> `Vector(like i32)`. These changes have not yet gone through the proposal
> process, and we may decide to reject nested `like` until we have a
> demonstrated need.

In general, each `like` adds one additional parameterized implementation. There
is always the impl defined with all of the `like` expressions replaced by their
arguments with the definition supplied in the source code. In addition, for each
`like` expression, there is an automatic `impl` definition with it replaced by a
new parameter. These additional automatic implementations will delegate to the
main `impl` definition, which will trigger implicit conversions according to
[Carbon's ordinary implicit conversion rules](/docs/design/expressions/implicit_conversions.md).
In this example, there are two uses of `like`, producing three implementations

```carbon
impl like Meters as MulWith(like f64)
    where .Result = Meters {
  fn Op[self: Self](other: f64) -> Result {
    return self.Scale(other);
  }
}
```

is equivalent to "implementation one", "implementation two", and:

```carbon
impl forall [T:! ImplicitAs(Meters)]
    T as MulWith(f64) where .Result = Meters {
  fn Op[self: Self](other: f64) -> Result {
    // Will implicitly convert `self` to `Meters` in
    // order to match the signature of this `Op` method.
    return self.((Meters as MulWith(f64)).Op)(other);
  }
}
```

`like` may be used in `impl` forward declarations in a way analogous to `impl`
definitions.

```carbon
impl like Meters as MulWith(like f64)
    where .Result = Meters;
}
```

is equivalent to:

```carbon
// All `like`s removed. Same as the declaration part of
// "implementation one", without the body of the definition.
impl Meters as MulWith(f64) where .Result = Meters;

// First `like` replaced with a wildcard.
impl forall [T:! ImplicitAs(Meters)]
    T as MulWith(f64) where .Result = Meters;

// Second `like` replaced with a wildcard. Same as the
// declaration part of "implementation two", without the
// body of the definition.
impl forall [T:! ImplicitAs(f64)]
    Meters as MulWith(T) where .Result = Meters;
```

In addition, the generated `impl` definition for a `like` is implicitly injected
at the end of the (unique) source file in which the `impl` is defined. That is,
it is injected in the API file if the `impl` definition is in an API file, and
in the sole impl file with the `impl` definition otherwise.

If one `impl` declaration uses `like`, other declarations must use `like` in the
same way to match.

The `like` operator may be nested, as in:

```carbon
impl like Vector(like String) as Printable;
```

Which will generate implementations with declarations:

```carbon
impl Vector(String) as Printable;
impl forall [T:! ImplicitAs(Vector(String))] T as Printable;
impl forall [T:! ImplicitAs(String)] Vector(T) as Printable;
```

The generated implementations must be legal or the `like` is illegal. For
example, it must be legal to have those `impl` definitions in this library by
the [orphan rule](#orphan-rule). In addition, the generated `impl` definitions
must only require implicit conversions that are guaranteed to exist. For
example, there existing an implicit conversion from `T` to `String` does not
imply that there is one from `Vector(T)` to `Vector(String)`, so the following
use of `like` is illegal:

```carbon
// ❌ Illegal: Can't convert a value with type
//             `Vector(T:! ImplicitAs(String))`
//             to `Vector(String)` for `self`
//             parameter of `Printable.Print`.
impl Vector(like String) as Printable;
```

Since the additional implementation definitions are generated eagerly, these
errors will be reported in the file with the first declaration.

The argument to `like` must either not mention any type parameters, or those
parameters must be able to be determined due to being repeated outside of the
`like` expression.

```carbon
// ✅ Allowed: no parameters
impl like Meters as Printable;

// ❌ Illegal: No other way to determine `T`
impl forall [T:! IntLike] like T as Printable;

// ❌ Illegal: `T` being used in a `where` clause
//             is insufficient.
impl forall [T:! IntLike] like T
    as MulWith(i64) where .Result = T;

// ❌ Illegal: `like` can't be used in a `where`
//             clause.
impl Meters as MulWith(f64)
    where .Result = like Meters;

// ✅ Allowed: `T` can be determined by another
//             part of the query.
impl forall [T:! IntLike] like T
    as MulWith(T) where .Result = T;
impl forall [T:! IntLike] T
    as MulWith(like T) where .Result = T;

// ✅ Allowed: Only one `like` used at a time, so this
//             is equivalent to the above two examples.
impl forall [T:! IntLike] like T
    as MulWith(like T) where .Result = T;
```

## Parameterized types

Generic types may be defined by giving them compile-time parameters. Those
parameters may be used to specify types in the declarations of its members, such
as data fields, member functions, and even interfaces being implemented. For
example, a container type might be parameterized by a facet describing the type
of its elements:

```carbon
class HashMap(
    KeyT:! Hashable & Eq & Movable,
    ValueT:! Movable) {
  // `Self` is `HashMap(KeyT, ValueT)`.

  // Class parameters may be used in function signatures.
  fn Insert[addr self: Self*](k: KeyT, v: ValueT);

  // Class parameters may be used in field types.
  private var buckets: DynArray((KeyT, ValueT));

  // Class parameters may be used in interfaces implemented.
  extend impl as Container where .ElementType = (KeyT, ValueT);
  impl as OrderedWith(HashMap(KeyT, ValueT));
}
```

Note that, unlike functions, every parameter to a type must be a compile-time
binding, either symbolic using `:!` or template using `template`...`:!`, not
runtime, with a plain `:`.

Two types are the same if they have the same name and the same arguments, after
applying aliases and [rewrite constraints](#rewrite-constraints). Carbon's
[manual type equality](#manual-type-equality) approach means that the compiler
may not always be able to tell when two
[type expressions](terminology.md#type-expression) are equal without help from
the user, in the form of [`observe` declarations](#observe-declarations). This
means Carbon will not in general be able to determine when types are unequal.

Unlike an [interface's parameters](#parameterized-interfaces), a type's
parameters may be [deduced](terminology.md#deduced-parameter), as in:

```carbon
fn ContainsKey[KeyT:! Movable, ValueT:! Movable]
    (haystack: HashMap(KeyT, ValueT), needle: KeyT)
    -> bool { ... }
fn MyMapContains(s: String) {
  var map: HashMap(String, i32) = (("foo", 3), ("bar", 5));
  // ✅ Deduces `KeyT` = `String as Movable` from the types of both arguments.
  // Deduces `ValueT` = `i32 as Movable` from the type of the first argument.
  return ContainsKey(map, s);
}
```

Note that restrictions on the type's parameters from the type's declaration can
be [implied constraints](#implied-constraints) on the function's parameters. In
the above example, the `KeyT` parameter to `ContainsKey` gets `Hashable & Eq`
implied constraints from the declaration of the corresponding parameter to
`HashMap`.

> **Future work:** We may want to support optional deduced parameters in square
> brackets `[`...`]` before the explicit parameters in round parens `(`...`)`.

> **References:** This feature is from
> [proposal #1146: Generic details 12: parameterized types](https://github.com/carbon-language/carbon-lang/pull/1146).

### Generic methods

A generic type may have methods with additional compile-time parameters. For
example, this `Set(T)` type may be compared to anything implementing the
`Container` interface as long as the element types match:

```carbon
class Set(T:! Ordered) {
  fn Less[U:! Container with .ElementType = T, self: Self](u: U) -> bool;
  // ...
}
```

The `Less` method is parameterized both by the `T` parameter to the `Set` type
and its own `U` parameter deduced from the type of its first argument.

### Conditional methods

A method could be defined conditionally for a generic type by using a more
specific type in place of `Self` in the method declaration. For example, this is
how to define a dynamically sized array type that only has a `Sort` method if
its elements implement the `Ordered` interface:

```carbon
class DynArray(T:! type) {
  // `DynArray(T)` has a `Sort()` method if `T impls Ordered`.
  fn Sort[C:! Ordered, addr self: DynArray(C)*]();
}
```

**Comparison with other languages:** In
[Rust](https://doc.rust-lang.org/book/ch10-02-traits.html#using-trait-bounds-to-conditionally-implement-methods)
this feature is part of conditional conformance. Swift supports conditional
methods using
[conditional extensions](https://docs.swift.org/swift-book/LanguageGuide/Generics.html#ID553)
or
[contextual where clauses](https://docs.swift.org/swift-book/LanguageGuide/Generics.html#ID628).

### Specialization

[Specialization](terminology.md#checked-generic-specialization) is used to
improve performance in specific cases when a general strategy would be
inefficient. For example, you might use
[binary search](https://en.wikipedia.org/wiki/Binary_search_algorithm) for
containers that support random access and keep their contents in sorted order
but [linear search](https://en.wikipedia.org/wiki/Linear_search) in other cases.
Types, like functions, may not be specialized directly in Carbon. This effect
can be achieved, however, through delegation.

For example, imagine we have a parameterized class `Optional(T)` that has a
default storage strategy that works for all `T`, but for some types we have a
more efficient approach. For pointers we can use a
[null value](https://en.wikipedia.org/wiki/Null_pointer) to represent "no
pointer", and for booleans we can support `True`, `False`, and `None` in a
single byte. Clients of the optional library may want to add additional
specializations for their own types. We make an interface that represents "the
storage of `Optional(T)` for type `T`," written here as `OptionalStorage`:

```carbon
interface OptionalStorage {
  let Storage:! type;
  fn MakeNone() -> Storage;
  fn Make(x: Self) -> Storage;
  fn IsNone(x: Storage) -> bool;
  fn Unwrap(x: Storage) -> Self;
}
```

The default implementation of this interface is provided by a
[blanket implementation](#blanket-impl-declarations):

```carbon
// Default blanket implementation
impl forall [T:! Movable] T as OptionalStorage
    where .Storage = (bool, T) {
  ...
}
```

This implementation can then be
[specialized](#lookup-resolution-and-specialization) for more specific type
patterns:

```carbon
// Specialization for pointers, using nullptr == None
final impl forall [T:! type] T* as OptionalStorage
    where .Storage = Array(Byte, sizeof(T*)) {
  ...
}
// Specialization for type `bool`.
final impl bool as OptionalStorage
    where .Storage = Byte {
  ...
}
```

Further, libraries can implement `OptionalStorage` for their own types, assuming
the interface is not marked `private`. Then the implementation of `Optional(T)`
can delegate to `OptionalStorage` for anything that can vary with `T`:

```carbon
class Optional(T:! Movable) {
  fn None() -> Self {
    return {.storage = T.(OptionalStorage.MakeNone)()};
  }
  fn Some(x: T) -> Self {
    return {.storage = T.(OptionalStorage.Make)(x)};
  }
  ...
  private var storage: T.(OptionalStorage.Storage);
}
```

Note that the constraint on `T` is just `Movable`, not
`Movable & OptionalStorage`, since the `Movable` requirement is
[sufficient to guarantee](#lookup-resolution-and-specialization) that some
implementation of `OptionalStorage` exists for `T`. Carbon does not require
callers of `Optional`, even checked-generic callers, to specify that the
argument type implements `OptionalStorage`:

```carbon
// ✅ Allowed: `T` just needs to be `Movable` to form `Optional(T)`.
//             A `T:! OptionalStorage` constraint is not required.
fn First[T:! Movable & Eq](v: Vector(T)) -> Optional(T);
```

Adding `OptionalStorage` to the constraints on the parameter to `Optional` would
obscure what types can be used as arguments. `OptionalStorage` is an
implementation detail of `Optional` and need not appear in its public API.

In this example, a `let` is used to avoid repeating `OptionalStorage` in the
definition of `Optional`, since it has no name conflicts with the members of
`Movable`:

```carbon
class Optional(T:! Movable) {
  private let U:! Movable & OptionalStorage = T;
  fn None() -> Self {
    return {.storage = U.MakeNone()};
  }
  fn Some(x: T) -> Self {
    return {.storage = U.Make(x)};
  }
  ...
  private var storage: U.Storage;
}
```

> **Alternative considered:** Direct support for specialization of types was
> considered in [proposal #1146](/proposals/p1146.md#alternatives-considered).

## Future work

### Dynamic types

Checked-generics provide enough structure to support runtime dispatch for values
with types that vary at runtime, without giving up type safety. Both Rust and
Swift have demonstrated the value of this feature.

#### Runtime type parameters

This feature is about allowing a function's type parameter to be passed in as a
dynamic (non-compile-time) parameter. All values of that type would still be
required to have the same type.

#### Runtime type fields

Instead of passing in a single type parameter to a function, we could store a
type per value. This changes the data layout of the value, and so is a somewhat
more invasive change. It also means that when a function operates on multiple
values they could have different real types.

### Abstract return types

This lets you return an anonymous type implementing an interface from a
function. In Rust this is the
[`impl Trait` return type](https://rust-lang.github.io/rfcs/1522-conservative-impl-trait.html).

In Swift, there are discussions about implementing this feature under the name
"reverse generics" or "opaque result types":
[1](https://forums.swift.org/t/improving-the-ui-of-generics/22814#heading--reverse-generics),
[2](https://forums.swift.org/t/reverse-generics-and-opaque-result-types/21608),
[3](https://forums.swift.org/t/se-0244-opaque-result-types/21252),
[4](https://forums.swift.org/t/se-0244-opaque-result-types-reopened/22942),
Swift is considering spelling this `<V: Collection> V` or `some Collection`.

### Evolution

There are a collection of use cases for making different changes to interfaces
that are already in use. These should be addressed either by describing how they
can be accomplished with existing generics features, or by adding features.

In addition, evolution from (C++ or Carbon) templates to checked generics needs
to be supported and made safe.

### Testing

The idea is that you would write tests alongside an interface that validate the
expected behavior of any type implementing that interface.

### Impl with state

A feature we might consider where an `impl` itself can have state.

### Generic associated facets and higher-ranked facets

This would be some way to express the requirement that there is a way to go from
a type to an implementation of an interface parameterized by that type.

#### Generic associated facets

Generic associated facets are about when this is a requirement of an interface.
These are also called
"[associated type constructors](https://smallcultfollowing.com/babysteps/blog/2016/11/02/associated-type-constructors-part-1-basic-concepts-and-introduction/)."

Rust has
[stabilized this feature](https://github.com/rust-lang/rust/pull/96709).

#### Higher-ranked types

Higher-ranked types are used to represent this requirement in a function
signature. They can be
[emulated using generic associated facets](https://smallcultfollowing.com/babysteps//blog/2016/11/03/associated-type-constructors-part-2-family-traits/).

### Field requirements

We might want to allow interfaces to express the requirement that any
implementing type has a particular field. This would be to match the
expressivity of inheritance, which can express "all subtypes start with this
list of fields."

### Bridge for C++ customization points

See details in [the goals document](goals.md#bridge-for-c-customization-points).

### Variadic arguments

Some facility for allowing a function to take a variable number of arguments,
with the [definition checked](terminology.md#complete-definition-checking)
independent of calls. Open
[proposal #2240](https://github.com/carbon-language/carbon-lang/pull/2240) is
adding this feature.

### Value constraints for template parameters

We have planned support for predicates that constrain the value of non-facet
template parameters. For example, we might support a predicate that constrains
an integer to live inside a specified range. See
[question-for-leads issue #2153: Checked generics calling templates](https://github.com/carbon-language/carbon-lang/issues/2153)
and
[future work in proposal #2200: Template generics](/proposals/p2200.md#predicates-constraints-on-values).

## References

-   [#553: Generics details part 1](https://github.com/carbon-language/carbon-lang/pull/553)
-   [#731: Generics details 2: adapters, associated types, parameterized interfaces](https://github.com/carbon-language/carbon-lang/pull/731)
-   [#818: Constraints for generics (generics details 3)](https://github.com/carbon-language/carbon-lang/pull/818)
-   [#931: Generic impls access (details 4)](https://github.com/carbon-language/carbon-lang/pull/931)
-   [#920: Generic parameterized impls (details 5)](https://github.com/carbon-language/carbon-lang/pull/920)
-   [#950: Generic details 6: remove facets](https://github.com/carbon-language/carbon-lang/pull/950)
-   [#983: Generic details 7: final impls](https://github.com/carbon-language/carbon-lang/pull/983)
-   [#989: Member access expressions](https://github.com/carbon-language/carbon-lang/pull/989)
-   [#990: Generics details 8: interface default and final members](https://github.com/carbon-language/carbon-lang/pull/990)
-   [#1013: Generics: Set associated constants using `where` constraints](https://github.com/carbon-language/carbon-lang/pull/1013)
-   [#1084: Generics details 9: forward declarations](https://github.com/carbon-language/carbon-lang/pull/1084)
-   [#1088: Generic details 10: interface-implemented requirements](https://github.com/carbon-language/carbon-lang/pull/1088)
-   [#1144: Generic details 11: operator overloading](https://github.com/carbon-language/carbon-lang/pull/1144)
-   [#1146: Generic details 12: parameterized types](https://github.com/carbon-language/carbon-lang/pull/1146)
-   [#1327: Generics: `impl forall`](https://github.com/carbon-language/carbon-lang/pull/1327)
-   [#2107: Clarify rules around `Self` and `.Self`](https://github.com/carbon-language/carbon-lang/pull/2107)
-   [#2138: Checked and template generic terminology](https://github.com/carbon-language/carbon-lang/pull/2138)
-   [Issue #2153: Checked generics calling templates](https://github.com/carbon-language/carbon-lang/issues/2153)
-   [#2173: Associated constant assignment versus equality](https://github.com/carbon-language/carbon-lang/pull/2173)
-   [#2200: Template generics](https://github.com/carbon-language/carbon-lang/pull/2200)
-   [#2347: What can be done with an incomplete interface](https://github.com/carbon-language/carbon-lang/pull/2347)
-   [#2360: Types are values of type `type`](https://github.com/carbon-language/carbon-lang/pull/2360)
-   [#2376: Constraints must use `Self`](https://github.com/carbon-language/carbon-lang/pull/2376)
-   [#2483: Replace keyword `is` with `impls`](https://github.com/carbon-language/carbon-lang/pull/2483)
-   [#2687: Termination algorithm for impl selection](https://github.com/carbon-language/carbon-lang/pull/2687)
-   [#2760: Consistent `class` and `interface` syntax](https://github.com/carbon-language/carbon-lang/pull/2760)
-   [#2964: Expression phase terminology](https://github.com/carbon-language/carbon-lang/pull/2964)
-   [#3162: Reduce ambiguity in terminology](https://github.com/carbon-language/carbon-lang/pull/3162)
