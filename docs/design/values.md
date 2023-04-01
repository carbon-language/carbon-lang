# Values, variables, and pointers

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Value categories](#value-categories)
-   [Binding patterns and local variables with `let` and `var`](#binding-patterns-and-local-variables-with-let-and-var)
    -   [Local variables](#local-variables)
-   [L-values or _located_ values](#l-values-or-located-values)
-   [R-values or _readonly_ values](#r-values-or-readonly-values)
    -   [R-value customization](#r-value-customization)
    -   [Interop with C++ `const &`](#interop-with-c-const-)
-   [Pointers](#pointers)
    -   [Dereferencing customization](#dereferencing-customization)
    -   [Indexing](#indexing)
-   [`const`-qualified types](#const-qualified-types)

<!-- tocstop -->

## Value categories

Every value in Carbon has a
[value category](<https://en.wikipedia.org/wiki/Value_(computer_science)#lrvalue>)
that is either an L-value or an R-value.

[_L-values_ are _located values_.](#l-values-or-located-values) They represent
_storage_ and have a stable address. They are in principle mutable, although
their type's API may limit the mutating operations available.

[_R-values_ are _readonly values_.](#r-values-or-readonly-values) They cannot be
mutated in any way and may not have storage or a stable address.

## Binding patterns and local variables with `let` and `var`

[_Binding patterns_](/docs/design/README.md#binding-patterns) produce named
R-values by default. This is the desired default for many pattern contexts,
especially function parameters. R-values are a good model for "input" function
parameters which are the dominant and default style of function parameters:

```carbon
fn Sum(x: i32, y: i32) -> i32 {
  // `x` and `y` are R-values here. We can read them, but not modify.
  return x + y;
}
```

A pattern can be introduced with the `var` keyword to create a _variable
pattern_. This creates an L-value including the necessary storage. Every binding
pattern name introduced within a variable pattern is also an L-value. The
initializer for a variable pattern is directly used to initialize the storage.

```carbon
fn Consume(var x: SomeData) {
  // We can mutate and use the local `x` L-value here.
}

fn MutateThing(ptr: i64*);

fn Example() {
  let (x: i64, var y: i64) = (1, 2);

  // Allowed to take the address and mutate `y`.
  MutateThing(&y);

  // ❌ This would be an error though due to trying to take the address of `x`.
  MutateThing(&x);
}
```

### Local variables

A local binding pattern can be introduced with either the `let` or `var`
keyword. The `let` introducer begins a readonly pattern just like the default
patterns in other contexts. The `var` introduce works exactly the same as
introducing the pattern inside of a `let` binding with `var` -- there's just no
need for the outer `let`.

-   `let` _identifier_`:` _< expression |_ `auto` _>_ `=` _value_`;`
-   `var` _identifier_`:` _< expression |_ `auto` _> [_ `=` _value ]_`;`

These are just simple examples of binding patterns used directly in local
declarations. Local `let` and `var` declarations build on Carbon's general
[pattern matching](/docs/design/pattern_matching.md) design, with `var`
declarations implicitly starting off within a `var` pattern while `let`
declarations introduce patterns that work the same as function parameters and
others with bindings that are R-values by default.

## L-values or _located_ values

Located values in Carbon with language-provided storage and a stable address for
that storage. This means that an L-value's address can be taken to produce a
[pointer](#pointers) to that value.

Taking an L-value's address can be done explicitly using an address-of
expression: `&x` requires `x` to be an L-value and produces the address of that
value's storage. It can also be done implicitly when calling a [method]() on an
L-value object where the method's `self` parameter has an [`addr` qualifier]().
The address of the L-value is passed as a pointer to the `self` parameter.

## R-values or _readonly_ values

An R-value cannot be mutated, cannot have its address taken, and may not have
storage at all or a stable address of storage. They model abstract values like
function input parameters and constants. They can be formed in two ways -- a
literal expression like `42`, or by converting an L-value to an R-value.

A core goal of R-values is to provide a single model that can get both the
efficiency of passing by value when working with small types such as those that
fit into a machine register, but also the efficiency of minimal copies when
working with types where a copy would require extra allocations or other costly
resources. This directly helps programmers by providing a simpler model to
select the mechanism of passing function inputs. But it is also important to
enable generic code that needs a single type model that will have generically
good performance.

To achieve this goal, a Carbon program must in general behave equivalently with
R-values that are implemented as a _reference_ to the original object or as
either a _copy_ or _move_ if that would be valid for the type. However, using a
copy or a move is purely optional and an optimization. R-values support
uncopyable and unmovable types.

**Experimental:** We currently make an additional requirement that helps ensure
this equivalence will be true and allows us to detect the most risky cases where
it would not be true: we require that once an R-value is formed, any original
object must not be mutated prior to the last read from that R-value. We consider
this restriction experimental as we may want to strengthen or weaken it based on
our experience with Carbon code using these constructs, and especially
interoperating with C++.

We expect even with these restrictions to make R-values in Carbon useful in
roughly the same places as `const &`s in C++, but with added efficiency in the
case where the values can usefully be kept in machine registers. We also
specifically encourage a mental model of a `const &` with extra efficiency.

### Comparison to C++ parameters

While these are called _R-values_ in Carbon and sometimes shortened to just
"values", they are not related to "by-value" parameters as they exist in C++.
C++ by-value parameters are semantically defined to create a new local copy of
the argument, although it may move into this copy.

Carbon's values are much closer to a `const &` in C++ with extra restrictions
such as allowing copies under "as-if" in limited cases and preventing taking the
address. Combined, these allow implementation strategies such as in-register
parameters.

### Representation and type-based modeling

The representation of an R-value binding is especially important because it
forms the calling convention used for the vast majority of function parameters
-- function inputs. Given this importance, it's important that it is predictable
and customizable by the value's type. Similarly, while Carbon code must be
correct with either a copy or a reference-based implementation, we want which
implementation strategy is used to be a predictable and customizable property of
the type of a value. To achieve both of these, Carbon models forming R-values as
a conversion to a representation type that is specified with an [interface]().
The default implementation of this interface works to choose a good default
based on the size and complexity of a given type, and it can be further
customized by types as needed. For example, types with dedicated and optimized
"view" representations can immediately use this to back any R-value, and it will
even be used in generic code.

### R-value customization

Carbon models both the conversion from L-value to R-value and the representation
of R-values in a way that permits customization. We can also explain the
expected default options using the same framework. Customization occurs by
implementing an interface:

```carbon
interface RValueRep {
  // ...
}

class CustomRValue {
  // I want a custom R-value!
  impl as RValueRep ... { ... }
}
```

The first aspect of customization is the representation of an R-value for a type
`T`. One option is to leave this uncustomized and let the language implement it
however is most effective. A second option is to provide a custom type that is
used to model the _representation_ of an R-value.

An uncustomized representation that leaves it up to the language to implement
can be requested by selecting the identity. This doesn't imply a _copy_ in any
way, it simply leaves it up to the language to pick an effective representation.
When this representation is used, fields of a type can be accessed from an
R-value of that type using normal field access syntax. The result of such an
access is an R-value of the field type. Example:

```carbon
class SomeDataType {
  var x: f64;
  var y: f64;

  // No custom `impl` for `RValueRep`.
}

fn F(data: SomeDataType) {
  // Can directly access `x` and `y` here as R-values.
  let sum: f64 = data.x + data.y;

  // Can't mutate them, that would require an L-value field.
  data.x = sum;
}
```

However, it is also possible to customize the representation of R-values for a
particular type. This involves specifying a custom type as the associated type
of the `RValueRep` impl. When forming an R-value of type `T`, the actual
representation will be an object of type `T.(RValueRep.RepType)`. This will be
used on function call boundaries and elsewhere we need to materialize a physical
manifestation of an R-value. It also dictates what operations are valid on an
R-value of type `T`. When `T` has a customized R-value representation type, the
_only_ valid operation on a `T` R-value is to convert it to that representation
type. No filed access or other operations are permitted. This is mostly useful
when there is a dedicated type that models an R-value semantic more effectively
to trigger conversions to that type and define the readonly API in terms of it.
This interface extends `ImplicitAs`, because it can only work when an implicit
conversion to the representation type is desirable. Example:

```carbon
interface RValueRep {
  let RepType:! type;

  // Called to for the representation object when binding an L-value as an
  // R-value.
  fn LValueTRValue[addr self: Self*]() -> RepType;

  // Extend the implicit conversions.
  extends ImplicitAs(RepType);
  // Provide the language-provided conversion given the above. Custom
  // conversions can still be provided here.
  default fn Convert[self: Self]() -> RepType { return self; }
}

class StringView {
  // A typical readonly view of a string API...
  fn ExampleMethod[self: Self]() { ... }
}

class String {
  var data_ptr: Char*;
  var size: i64;

  // Define the R-value method API by delegating to `StringView`.
  fn ExampleMethod[self: StringView]() { self.ExampleMethod(); }

  // Extends `ImplicitAs(StringView)` with a default implementation.
  impl as RValueRep where .RepType = StringView {
    fn LValueToRValue[addr self: Self*]() -> StringView {
      // Because this is called on the L-value being bound to an R-value, we
      // can get at an SSO buffer or other interior pointers of `self`.
      return StringView::Create(self->data_ptr, self->size);
    }
  }
}
```

When using a customized R-value representation, the `LValueToRValue` interface
method is called on an L-value in order to construct the representation used to
bind an R-value. Because this is done on the _L-value_, it has the opportunity
to capture the address of the underlying object as needed, for example to
provide an implementation similar to a C++ `const &` and the `StringView` above.
This also allows types with inline buffers or small-size-optimization buffers to
create effective R-value representations by capturing pointers into the original
L-value object's inline buffer.

However, it is important to note that the _representation_ type of an R-value is
just its representation and does not impact the type itself. Name lookup and
`impl` search occur for the same type regardless of R-value or L-value. But once
an particular method or function is selected, an implicit conversion can occur
from the original type to the representation type as part of the parameter or
receiver type. In fact, this conversion is the _only_ operation that can occur
for a customized representation type, wherever it is necessary as implemented.

### Polymorphic types

R-values can be used with
[polymorphic types](/docs/design/classes.md#inheritance), for example:

```
base class MyBase { ... }

fn UseBase(b: MyBase) { ... }

class Derived extends MyBase { ... }

fn PassDerived() {
  var d: Derived = ...;
  // Allowed to pass `d` here:
  UseBase(d);
}
```

This is still allowed to create a copy or to move, but it must not _slice_. Even
if a copy is created, it must be a `Derived` object, even though this may limit
the available implementation strategies.

### Interop with C++ `const &` and `const` methods.

While R-values cannot have their address taken in Carbon, they should be
interoperable with C++ `const &`s and C++ `const`-qualified methods. This will
in-effect "pin" the R-value (or a copy) into memory and allow C++ to take its
address. Without supporting this, R-values would likely create an untenable
interop ergonomic barrier. However, this does create some additional constraints
on R-values and a way that their addresses can escape unexpectedly.

Despite interop requiring an address to implement, the address isn't guaranteed
to be stable or useful or point back to some original L-value necessarily. The
ability of the implementation to introduce copies or a temporary specifically
for the purpose of the interop remains.

**Open question:** when a type customizes its R-value representation, as
currently specified this will break the use of `const &` C++ APIs with such an
R-value. We may need to further extend the R-value customization interface to
allow types to define how a `const &` is manifested when needed.

### Escape hatches for R-values in Carbon

**Open question:** It may be necessary to provide some amount of escape hatch
for taking the address of R-values. The [C++ interop](#interop-with-c-const)
above already takes their address. Currently, this is the extent of an escape
hatch to the restrictions on R-values.

If a further escape hatch is needed, this kind of fundamental weakening of the
semantic model would be a good case for some syntactic marker like Rust's
`unsafe`, although rather than a region, it would seem better to tie it directly
to the operation in question. For example:

```carbon
class S {
  fn ImmutableMemberFunction[me: Self]();
  fn MutableMemberFunction[addr me: Self*]();
}

fn F(immutable_s: S) {
  // This is fine.
  immutable_s.ImmutableMemberFunction();

  // This requires an unsafe marker in the syntax.
  immutable_s.unsafe MutableMemberFunction();
}
```

## Pointers

### Dereferencing customization

### Indexing

## `const`-qualified types
