# Qualified names and member access

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Package and namespace members](#package-and-namespace-members)
-   [Lookup within values](#lookup-within-values)
    -   [Templates and generics](#templates-and-generics)
-   [Member access](#member-access)
-   [Precedence and associativity](#precedence-and-associativity)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

A _qualified name_ is a [word](../lexical_conventions/words.md) that is preceded
by a period. The name is found within a contextually-determined entity:

-   In a member access expression, this is the entity preceding the period.
-   For a designator in a struct literal, the name is introduced as a member of
    the struct type.

A _member access expression_ allows a member of a value, type, interface,
namespace, and so on to be accessed by specifying a qualified name for the
member. A member access expression is either a _direct_ member access expression
of the form:

-   _member-access-expression_ ::= _expression_ `.` _word_

or an _indirect_ member access of the form:

-   _member-access-expression_ ::= _expression_ `.` `(`
    _member-access-expression_ `)`

The meaning of a qualified name in a member access expression depends on the
first operand, which can be:

-   A [package or namespace name](#package-and-namespace-members).
-   A [value of some type](#lookup-within-values).

## Package and namespace members

If the first operand is a package or namespace name, the member access must be
direct. The _word_ must name a member of that package or namespace, and the
result is the package or namespace member with that name.

An expression that names a package or namespace can only be used as the first
operand of a member access or as the target of an `alias` declaration.

```
namespace MyNamespace;
fn MyNamespace.Fn() {}

// ✅ OK, can alias a namespace.
alias MyNS = MyNamespace;
fn CallFn() { MyNS.Fn(); }

// ❌ Error: a namespace is not a value.
let MyNS2:! auto = MyNamespace;

fn CallFn2() {
  // ❌ Error: cannot perform indirect member access into a namespace.
  MyNamespace.(MyNamespace.Fn)();
}
```

## Lookup within values

When the first operand is not a package or namespace name, there are three
remaining cases we wish to support:

-   The first operand is a value, and lookup should consider members of the
    value's type.
-   The first operand is a type, and lookup should consider members of that
    type. For example, `i32.Least` should find the member constant `Least` of
    the type `i32`.
-   The first operand is a type-of-type, and lookup should consider members of
    that type-of-type. For example, `Addable.Add` should find the member
    function `Add` of the interface `Addable`. Because a type-of-type is a type,
    this is a special case of the previous bullet.

Note that because a type is a value, and a type-of-type is a type, these cases
are overlapping and not entirely separable.

If any of the above lookups ever looks for members of a type parameter, it
should consider members of the type-of-type, treating the type parameter as an
archetype.

For a direct member access, the word is looked up in the following types:

-   If the first operand can be evaluated and evaluates to a type, that type.
-   If the type of the first operand can be evaluated, that type.
-   If the type of the first operand is a generic type parameter, and the type
    of that generic type parameter can be evaluated, that type-of-type.

The results of these lookups are combined. If more than one distinct entity is
found, the qualified name is invalid.

For an indirect member access, the second operand is evaluated to determine the
member being accessed.

For example:

```
interface Printable {
  fn Print[me: Self]();
}
external impl i32 as Printable;
class Point {
  var x: i32;
  var y: i32;
  // Internal impl injects the name `Print` into class `Point`.
  impl as Printable;
}

fn PrintPointTwice() {
  var p: Point = {.x = 0, .y = 0};

  // ✅ OK, `x` found in type of `p`, namely `Point`.
  p.x = 1;
  // ✅ OK, `y` found in the type `Point`.
  p.(Point.y) = 1;

  // ✅ OK, `Print` found in type of `p`, namely `Point`.
  p.Print();
  // ✅ OK, `Print` found in the type `Printable`.
  p.(Printable.Print)();
}
fn GenericPrint[T: Printable](a: T) {
  // ✅ OK, type of `a` is the type parameter `T`;
  // `Print` found in the type of `T`, namely `Printable`.
  a.Print();
}
fn CallGenericPrint(p: Point) {
  GenericPrint(p);
}
```

The resulting member is then [accessed](#member-access) within the value denoted
by the first operand.

### Templates and generics

If the value or type of the first operand depends on a template or generic
parameter, the lookup is performed from a context where the value of that
parameter is unknown. Evaluation of an expression involving the parameter may
still succeed, but will result in a symbolic value involving that parameter.

```
class GenericWrapper(T:! Type) {
  var field: T;
}
fn F[T:! Type](x: GenericWrapper(T)) -> T {
  // ✅ OK, finds `GenericWrapper(T).field`.
  return x.field;
}

class TemplateWrapper(template T:! Type) {
  var field: T;
}
fn G[template T:! Type](x: TemplateWrapper(T)) -> T {
  return x.field;
}
```

> **TODO:** The behavior of `G` above is not yet fully decided. If class
> templates can be specialized, then we cannot know the members of
> `TemplateWrapper(T)` without knowing `T`, so this first lookup will find
> nothing. In any case, as described below, the lookup will be performed again
> when `T` is known.

If the value or type depends on any template parameters, the lookup is redone
from a context where the values of those parameters are known, but where the
values of any generic parameters are still unknown. The lookup results from
these two contexts are combined, and if more than one distinct entity is found,
the qualified name is invalid.

The lookup for a member name never considers the values of any generic
parameters that are in scope at the point where the member name appears.

```carbon
class Cowboy { fn Draw[me: Self](); }
interface Renderable { fn Draw[me: Self](); }
external impl Cowboy as Renderable { fn Draw[me: Self](); }
fn DrawDirect(c: Cowboy) { c.Draw(); }
fn DrawGeneric[T:! Renderable](c: T) { c.Draw(); }
fn DrawTemplate[template T:! Renderable](c: T) { c.Draw(); }

fn Draw(c: Cowboy) {
  // ✅ Calls member of `Cowboy`.
  DrawDirect(c);
  // ✅ Calls member of `impl Cowboy as Renderable`.
  DrawGeneric(c);
  // ❌ Error: ambiguous.
  DrawTemplate(c);
}

class RoundWidget {
  external impl as Renderable {
    fn Draw[me: Self]();
  }
  alias Draw = Renderable.Draw;
}

class SquareWidget {
  fn Draw[me: Self]() {}
  external impl as Renderable {
    alias Draw = Self.Draw;
  }
}

fn DrawWidget(r: RoundWidget, s: SquareWidget) {
  // ✅ OK, lookup in type and in type-of-type find the same entity.
  DrawTemplate(r);

  // ✅ OK, lookup in type and in type-of-type find the same entity.
  DrawTemplate(s);

  // ✅ OK, found in type.
  r.Draw();
  s.Draw();

  // ✅ OK, inner member access resolves `RoundWidget.Draw` to
  // the member `Draw` of `impl RoundWidget as Renderable`,
  // outer member access forms a bound member function.
  r.(RoundWidget.Draw)();

  // ✅ OK, inner member access names `SquareWidget.Draw`,
  // outer member access forms a bound member function.
  s.(SquareWidget.Draw)();

  // ❌ Error, can't call `Draw[me: SquareWidget]()` on `RoundWidget` object.
  r.(SquareWidget.Draw)();

  // ❌ Error, inner member access resolves `RoundWidget.Draw` to
  // the member `Draw` of `impl RoundWidget as Renderable`;
  // can't call `Draw[me: RoundWidget]()` on `SquareWidget` object.
  s.(RoundWidget.Draw)();
}
```

## Member access

A member `M` is accessed within a value `V` as follows:

| Left of `.`  | Right of `.`                     | Result                                                                    |
| ------------ | -------------------------------- | ------------------------------------------------------------------------- |
| Non-type     | Member of type                   | Bound member of the object                                                |
| Non-type     | Non-instance member of interface | The member of the matching `impl`                                         |
| Non-type     | Instance member of interface     | Bound member of the object referring to the member of the matching `impl` |
| Type         | Member of type                   | The member of the type                                                    |
| Type         | Member of interface              | The member of the matching `impl`                                         |
| Type-of-type | Member of interface              | The member of the interface                                               |

Any other case is an error.

In more detail, member access consists of the following steps:

-   _`impl` lookup:_ If `M` is a member of an interface `I` and `V` does not
    evaluate to a type-of-type, then `M` is replaced by the corresponding member
    of an implementation of `I`.

    The type `T` that is expected to implement `I` is `V` if `V` can be
    evaluated and evaluates to a type, and `T` is the type of `V` otherwise. The
    appropriate `impl T as I` implementation is located. `M` is replaced by the
    member of that `impl` that corresponds to `M`.

    ```carbon
    interface I {
      // #1
      fn F[me: Self]() {}
    }
    class C {
      impl as I {
        // #2
        fn F[me: C]() {}
      }
    }

    // `M` is `I.F` and `V` is `I`. Because `V` is a type-of-type,
    // `impl` lookup is not performed, and the alias binds to #1.
    alias A1 = I.F;

    // `M` is `I.F` and `V` is `C`. Because `V` is a type, `T` is `C`.
    // `impl` lookup is performed, and the alias binds to #2.
    alias A2 = C.F;
    ```

    Instance binding may also apply if the member is an instance member.

    ```carbon
    var c: C;
    // `M` is `I.F` and `V` is `c`. Because `V` is not a type, `T` is the
    // type of `c`, which is `C`. `impl` lookup is performed, and `M` is
    // replaced with #2. Then instance binding is performed.
    c.F();
    ```

-   _Instance binding:_ If the member is an instance member -- a field or a
    method -- and `V` does not evaluate to a type, then:

    -   For a field member in class `C`, `V` is required to be of type `C` or of
        a type derived from `C`. The result is the corresponding subobject
        within `V`. The result is an lvalue if `V` is an lvalue.

        ```carbon
        var dims: auto = {.width = 1, .height = 2};
        // `dims.width` denotes the field `width` of the object `dims`.
        Print(dims.width);
        // `dims` is an lvalue, so `dims.height` is an lvalue.
        dims.height = 3;
        ```

    -   For a method, `V` is converted to the recipient type, as follows. First,
        if the method declares its recipient type with `addr`, then `V` is
        replaced by `&V`. Then, `V` is implicitly converted to the declared `me`
        type.

        The result is a _bound method_, which is a value `F` such that a
        function call `F(args)` behaves the same as a call to `M(args)` with the
        `me` parameter initialized by `V`.

        ```carbon
        class Blob {
          fn Mutate[addr me: Self*](n: i32);
        }
        fn F(p: Blob*) {
          // ✅ OK, forms bound method `((*p).M)` and calls it.
          // This calls `Blob.Mutate` with `me` initialized by `&(*p)`
          // and `n` initialized by `5`.
          (*p).Mutate(5);

          // ✅ OK, same as above.
          var bound_m: auto = (*p).Mutate;
          bound_m(5);
        }
        ```

-   If instance binding is not performed, the result is the member, but
    evaluating the member access expression still evaluates `V`. An expression
    that names an instance member can only be used as the second operand of a
    member access or as the target of an `alias` declaration.

    ```carbon
    class C {
      fn StaticMethod();
      var field: i32;
      class Nested {}
    }
    fn CallStaticMethod(c: C) {
      // ✅ OK, calls `C.StaticMethod`.
      C.StaticMethod();

      // ✅ OK, evaluates expression `c` then calls `C.StaticMethod`.
      c.StaticMethod();

      // ❌ Error: name of instance member `C.field` can only be used in a
      // member access or alias.
      C.field = 1;
      // ✅ OK, instance binding is performed by outer member access,
      // same as `c.field = 1;`
      c.(C.field) = 1;

      // ✅ OK
      let T:! Type = C.Nested;
      // ❌ Error: value of `:!` binding is not constant because it
      // refers to local variable `c`.
      let U:! Type = c.Nested;
    }
    ```

The first operand must be used in some way: a member access must either be
direct, so the first operand is used for lookup, or must result in `impl`
lookup, instance binding, or both.

```
interface Printable {
  fn Print[me: Self]();
}
external impl i32 as Printable {
  fn Print[me: Self]();
}
fn MemberAccess(n: i32) {
  // ✅ OK: `Printable.Print` is the interface member.
  // `i32.(Printable.Print)` is the corresponding member of the `impl`.
  // `n.(i32.(Printable.Print))` is a bound member function naming that member.
  n.(i32.(Printable.Print))();

  // ✅ Same as above, `n.(Printable.Print)` is interpreted as
  // `n.(i32.(Printable.Print))()`
  // because `a` does not evaluate to a type. Performs impl lookup
  // and then instance binding.
  n.(Printable.Print)();
}

// ✅ OK, member `Print` of interface `Printable`.
alias X1 = Printable.Print;
// ❌ Error, indirect access doesn't perform impl lookup or instance binding.
alias X2 = Printable.(Printable.Print);
// ✅ OK, member `Print` of `impl i32 as Printable`.
alias X3 = i32.(Printable.Print);
// ❌ Error, indirect access doesn't perform impl lookup or instance binding.
alias X4 = i32.(i32.(Printable.Print));
```

## Precedence and associativity

Member access expressions associate left-to-right:

```
class A {
  class B {
    fn F();
  }
}
interface B {
  fn F();
}
external impl A as B;

fn Use(a: A) {
  // Calls member `F` of class `A.B`.
  (a.B).F();
  // Calls member `F` of interface `B`, as implemented by type `A`.
  a.(B.F)();
  // Same as `(a.B).F()`.
  a.B.F();
}
```

Member access has lower precedence than primary expressions, and higher
precedence than all other expression forms.

```
// ✅ OK, `*` has lower precedence than `.`. Same as `(A.B)*`.
var p: A.B*;
// ✅ OK, `1 + (X.Y)` not `(1 + X).Y`.
var n: i32 = 1 + X.Y;
```

## Alternatives considered

-   [Separate syntax for static versus dynamic access, such as `::` versus `.`](/proposals/p0989.md#separate-syntax-for-static-versus-dynamic-access)
-   [Use a different lookup rule for names in templates](/proposals/p0989.md#use-a-different-lookup-rule-in-templates)

## References

-   Proposal
    [#989: member access expressions](https://github.com/carbon-language/carbon-lang/pull/989)
-   [Question for leads: constrained template name lookup](https://github.com/carbon-language/carbon-lang/issues/949)
