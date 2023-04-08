# Values, variables, and pointers

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Values, variables, and expressions](#values-variables-and-expressions)
    -   [Expression categories](#expression-categories)
-   [Binding patterns and local variables with `let` and `var`](#binding-patterns-and-local-variables-with-let-and-var)
    -   [Local variables](#local-variables)
    -   [Consuming function parameters](#consuming-function-parameters)
-   [Reference expressions](#reference-expressions)
    -   [Durable reference expressions](#durable-reference-expressions)
    -   [Ephemeral reference expressions](#ephemeral-reference-expressions)
-   [Value expressions](#value-expressions)
    -   [Comparison to C++ parameters](#comparison-to-c-parameters)
    -   [Representation and type-based modeling](#representation-and-type-based-modeling)
    -   [Value representation customization](#value-representation-customization)
    -   [Polymorphic types](#polymorphic-types)
    -   [Interop with C++ `const &` and `const` methods.](#interop-with-c-const--and-const-methods)
    -   [Escape hatches for R-values in Carbon](#escape-hatches-for-r-values-in-carbon)
-   [Initializing expressions](#initializing-expressions)
-   [Pointers](#pointers)
    -   [Reference types](#reference-types)
    -   [Syntax](#syntax)
    -   [Syntax-free dereference and address-of](#syntax-free-dereference-and-address-of)
    -   [Dereferencing customization](#dereferencing-customization)
-   [`const`-qualified types](#const-qualified-types)
-   [Lifetime overloading](#lifetime-overloading)

<!-- tocstop -->

## Values, variables, and expressions

Carbon has both _values_ like `42`, `true`, and `i32` (a type value). It can
also have a _variable_ which has _storage_ where we can read and write values.
Storage also allows taking the address and working with the memory.

However, while these terms are convenient are often used to casually describe an
entity in Carbon, they're not the right way to categorize or formalize the
semantics of Carbon code. Instead, it uses a more precise model which
categorizes the _expressions_. This allows us to both be more precise and
articulate important differences not captured without looking at the expression
itself.

### Expression categories

There are three expression categories in Carbon:

-   [_Value expressions_](#value-expressions) produce an abstract, read-only
    value that cannot be modified or have its address taken.
-   [_Reference expressions_](#reference-expressions) refer to _storage_ with a
    particular _location_ where a value may be read, written, or its address
    taken.
    -   [_Durable reference expressions_](#durable-reference-expressions) are
        reference expressions which cannot refer to _temporary_ storage, but
        must refer to some storage with outlives the full expression.
    -   [_Ephemeral reference expressions_](#ephemeral-reference-expressions)
        are reference expressions which _can_ refer to temporary storage.
-   [_Initializing expressions_](#initializing-expressions) which require
    storage to be provided as an input and initialize it to some value.

The syntax and syntactic context of an expression fully determines both the
expressions initial category and which if any conversions from one category to
another must take place. Carbon specifically works to avoid involving the _type
system_ in this determination as a way to reduce the complexity required within
the type system.

The general conversions and the semantics implied between these categories are
below:

-   An _initializing expression_ can be formed from:
    -   A _value expression_ by using the value to initialize the storage,
        analogous to
        [direct initialization](https://en.cppreference.com/w/cpp/language/direct_initialization)
        in C++.
    -   A _reference expression_ by copying the referenced value into the
        storage, analogous to
        [copy initialization](https://en.cppreference.com/w/cpp/language/copy_initialization)
        in C++.
-   An _ephemeral reference expression_ can be formed from:
    -   A _durable reference expression_ trivially.
    -   An _initializing expression_ by materializing temporary storage to be
        initialized and then referencing that storage.
-   A _durable reference expression_ cannot be produced by converting from
    another expression category.
-   A _value expression_ can be formed from a _reference expression_ by reading
    the value from its storage.

Multiple steps of these conversions can be combined, for example to produce a
value by first initializing an ephemeral reference and then reading the value
from its storage.

## Binding patterns and local variables with `let` and `var`

[_Binding patterns_](/docs/design/README.md#binding-patterns) introduce names
that are [_value expressions_](#value-expressions) by default and are called
_value bindings_. This is the desired default for many pattern contexts,
especially function parameters. Values are a good model for "input" function
parameters which are the dominant and default style of function parameters:

```carbon
fn Sum(x: i32, y: i32) -> i32 {
  // `x` and `y` are value expressions here. We can use their value, but not
  // modify them or take their address.
  return x + y;
}
```

Value bindings require the matched expression to be a _value expression_,
converting it into one as necessary.

A pattern can be introduced with the `var` keyword to create a variable with
storage. Every binding pattern name introduced within a variable pattern is
called a _variable binding_ and forms a
[_reference_expression_](#reference-expressions) when used. Variable bindings
require their matched expression to be an _initializing expression_ and provide
the underlying variable storage to it to be initialized.

```carbon
fn MutateThing(ptr: i64*);

fn Example() {
  // Both `1` and `2` here start as value expressions. That is a match for `1`.
  // For `2`, the variable binding requires it to be converted to an
  // initializing expression by using the value `2` to initialize the provided
  // variable storage that `y` will refer to.
  let (x: i64, var y: i64) = (1, 2);

  // Allowed to take the address and mutate `y` as it is a durable reference
  // expression.
  MutateThing(&y);

  // ❌ This would be an error though due to trying to take the address of the
  // value expression `x`.
  MutateThing(&x);
}
```

### Local variables

A local binding pattern can be introduced with either the `let` or `var`
keyword. The `let` introducer begins a value pattern just like the default
patterns in other contexts. The `var` introducer works exactly the same as
introducing the pattern in some other context with `var` -- there's just no need
for the outer `let`.

-   `let` _identifier_`:` _< expression |_ `auto` _>_ `=` _value_`;`
-   `var` _identifier_`:` _< expression |_ `auto` _> [_ `=` _value ]_`;`

These are just simple examples of binding patterns used directly in local
declarations. Local `let` and `var` declarations build on Carbon's general
[pattern matching](/docs/design/pattern_matching.md) design, with `var`
declarations implicitly starting off within a `var` pattern while `let`
declarations introduce patterns that work the same as function parameters and
others with bindings that produce values by default.

### Consuming function parameters

Just as part of a `let` binding can use a `var` prefix to become a variable
pattern and bind a name to an L-value, so can function parameters:

```carbon
fn Consume(var x: SomeData) {
  // We can mutate and use the local `x` L-value here.
}
```

This allows us to model an important special case of function inputs -- those
that are _consumed_ by the function, either through local processing or being
moved into some persistent storage. Marking these in the pattern and thus
signature of the function changes the expression category of arguments in the
caller, causing them to become _initializing expressions_ that directly
initialize storage dedicated-to and owned-by the function parameter.

This pattern serves the same purpose as C++'s pass-by-value when used with types
that have non-trivial resources attached to pass ownership into the function and
consume the resource. But rather than that being the seeming _default_, Carbon
makes this a use case that requires a special marking.

## Reference expressions and variables

_Reference expressions_ refer to storage at some particular location where a
value may be read or written, and the address of the storage taken. There are
two sub-categories: _durable_ and _ephemeral_.

All reference expressions can have their storage's address taken implicitly when
calling a [method](TODO) on the reference expression where the method's `self`
parameter has an [`addr` specifier](TODO) -- the address of the reference
expression's storage is passed as a [pointer](#pointers) to the `self` parameter
for such methods.

### Durable reference expressions

_Durable reference expressions_ are those where the storage outlives the full
expression and the address could be meaningfully propagated out of it as well.
The address of a durable reference expression can be explicitly and directly
with an address-of expression: `&x`.

There is no way to convert another category of expression into a durable
reference expression, they always directly refer to some declared variable
binding.

### Ephemeral reference expressions

_Ephemeral reference expressions_ still refer to storage, but it may be storage
materialized for a temporary that will not outlive the full expression. Their
address can only be taken implicitly as part of a method call.

These expressions are typically formed by conversion from another expression
category. Either a trivial conversion from a durable reference expression, or a
conversion of an initializing expression by materializing temporary storage to
be initialized and then producing an ephemeral reference to it.

## Value expressions

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

### Value representation customization

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
  fn LValueToRValue[addr self: const Self*]() -> RepType;

  // Extend the implicit conversions.
  extends ImplicitAs(RepType);

  // Provide the language-provided conversion given the above. Custom
  // conversions can still be provided here.
  final fn Convert[self: Self]() -> RepType { ... }
}

class StringView {
  // A typical readonly view of a string API...
  fn ExampleMethod[self: Self]() { ... }
}

class String {
  var data_ptr: Char*;
  var size: i64;

  // Define the R-value method API by delegating to `StringView`. Note that
  // using `self: StringView` here instead of `self: Self` isn't necessary,
  // it merely allows immediately accessing that API rather than requiring
  // a conversion first.
  fn ExampleMethod[self: StringView]() { self.ExampleMethod(); }

  // Extends `ImplicitAs(StringView)` with a default implementation.
  impl as RValueRep where .RepType = StringView {
    fn LValueToRValue[addr self: const Self*]() -> StringView {
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
for taking the address of R-values. The
[C++ interop](#interop-with-c-const--and-const-methods) above already takes
their address. Currently, this is the extent of an escape hatch to the
restrictions on R-values.

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

## Initializing expressions

...

## Pointers

Pointers in Carbon are the primary mechanism for _indirect access_ to storage
containing some value. Dereferencing a pointer is one of the primary ways to
form a [_reference expression_](#reference-expressions).

Carbon pointers are heavily restricted compared to C++ pointers -- they cannot
be null and they cannot be indexed or have pointer arithmetic performed on them.
In some ways, this makes them more similar to references in C++, but they retain
the essential aspect of a pointer that they syntactically distinguish between
the point*er* and the point*ee*.

Carbon will still have mechanisms to achieve the equivalent behaviors as C++
pointers. Optional pointers are expected to serve nullable use cases. Slice or
view style types are expected to provide access to indexable regions. And even
raw pointer arithmetic is expected to be provided at some point, but through
specialized constructs given the specialized nature of these operations.

> TODO: Add explicit designs for these use cases and link to them here.

### Reference types

Unlike C++, Carbon does not currently have reference types. The only form of
indirect access are pointers. There are a few aspects to this decision that need
to be separated carefully from each other as the motivations and considerations
are different.

First, Carbon has only a single fundamental construct for indirection because
this gives it a single point that needs extension and configuration if and when
we want to add more powerful controls to the indirect type system such as
lifetime annotations or other safety or optimization mechanisms. The designs
attempts to identify a single, core indirection tool and then layer other
related use cases on top. This is motivated by keeping the language scalable as
it evolves and reducing the huge explosion of complexity that C++ sees due to
having a large space here. For example, when there are N > 1 ways to express
indirection equivalently and APIs want to accept any one of them across M
different parameters they can end up with N \* M combinations.

Second, with pointers, Carbon's indirection mechanism retains the ability to
refer distinctly to the point*er* and the point*ee* when needed. This ends up
critical for supporting rebinding and so without this property more permutations
of indirection would likely emerge.

Third, and most controversially, Carbon doesn't provide a straightforward way to
avoid the syntactic distinction between indirect access and direct access. This
aspect is covered by our design decision around
[syntax-free dereference](#syntax-free-dereference-and-address-of).

### Syntax

The type of a pointer to a type `T` is written with a postfix `*` as in `T*`.
Dereferencing a pointer is an expression that produces an _L-value_ and is
written with a prefix `*` as in `*pointer`:

```carbon
var i: i32 = 42;
var p: i32* = &i;

// Form a reference expression `*p` and assign `13` to the referenced storage.
*p = 13;
```

This syntax is chosen specifically to remain as similar as possible to C++
pointer types as they are commonly written in code and are expected to be
extremely common and a key anchor of syntactic similarity between the languages.
The different alternatives and tradeoffs for this syntax issue were discussed
extensively in
[#523](https://github.com/carbon-language/carbon-lang/issues/523).

Carbon also supports an infix `->` operation, much like C++. However, Carbon
directly defines this as an exact rewrite to `*` and `.` so that `p->member`
becomes `(*p).member` for example.

As also covered extensively in
[#523](https://github.com/carbon-language/carbon-lang/issues/523), one of the
primary challenges of the C++ syntax is the composition of a prefix dereference
operation and other postfix or infix operations, especially when chained
together such as a classic C++ frustrations of mixes of dereference and
indexing: `(*(*p)[42])[13]`. Where these compositions are sufficiently common to
create ergonomic problems, the current plan is to introduce custom syntax
analogous to `->` that rewrites down to the grouped dereference. However,
nothing beyond `->` itself is currently provided.

### Syntax-free dereference and address-of

Carbon does not provide a way to dereference with zero syntax, even on function
interface boundaries. The presence of a clear level of indirection can be an
important distinction for readability. It helps surface that an object that may
appear local to the caller is in fact escaped and referenced externally to some
degree. However, it can also harm readability by forcing code that doesn't
_need_ to look different to do so anyway. In the worst case, this can
potentially interfere with being generic. Currently, Carbon prioritizes making
the distinction here visible.

It may prove desirable to provide an ergonomic aid to reduce dereferencing
syntax within function bodies, but this proposal suggests deferring that at
least initially in order to better understand the extent and importance of that
use case. If and when it is considered, a direction based around a way to bind a
name to an L-value produced by dereferencing in a pattern appears to be a
promising technique. Alternatively, there are various languages with
implicit-dereference designs that might be considered.

A closely related concern to syntax-free dereference is syntax-free address-of.
Here, Carbon supports one very narrow form of this: implicitly taking the
address of the implicit object parameter of member functions. Currently that is
the only place with such an implicit affordance. It is designed to be
syntactically sound to extend to other parameters, but currently that is not
planned to avoid surprise.

### Dereferencing customization

Carbon should support user-defined pointer-like types such as _smart pointers_
using a similar pattern as operator overloading or other expression syntax. That
is, it should rewrite the expression into a member function call on an
interface. Types can then implement this interface to expose pointer-like
_user-defined dereference_ syntax.

The interface might look like:

```
interface Pointer {
  let ValueT:! Type;
  fn Dereference[me: Self]() -> ValueT*;
}
```

Here is an example using a hypothetical `TaggedPtr` that carries some extra
integer tag next to the pointer it emulates:

```
class TaggedPtr(T:! Type) {
  var tag: Int32;
  var ptr: T*;
}
external impl [T:! Type] TaggedPtr(T) as Pointer {
  let ValueT:$ T;
  fn Dereference[me: Self]() -> T* { return me.ptr; }
}

fn Test(arg: TaggedPtr(T), dest: TaggedPtr(TaggedPtr(T))) {
  **dest = *arg;
  *dest = arg;
}
```

There is one tricky aspect of this. The function in the interface which
implements a pointer-like dereference must return a raw pointer which the
language then actually dereferences to form an L-value similar to that formed by
`var` declarations. This interface is implemented for normal pointers as a
no-op:

```
impl [T:! Type] T* as Pointer {
  let ValueT:$ Type = T;
  fn Dereference[me: Self]() -> T* { return me; }
}
```

Dereference expressions such as `*x` are syntactically rewritten to use this
interface to get a raw pointer and then that raw pointer is dereferenced. If we
imagine this language level dereference to form an L-value as a unary `deref`
operator, then `(*x)` becomes `(deref (x.(Pointer.Dereference)()))`.

Carbon should also use a simple syntactic rewrite for implementing `x->Method()`
as `(*x).Method()` without separate or different customization.

## `const`-qualified types

Carbon provides the ability to qualify a type `T` with the keyword `const` to
get a `const`-qualified type: `const T`. This is exclusively an API-subsetting
feature in Carbon -- for more fundamentally "immutable" use cases, R-values
should be used instead. Pointers to `const`-qualified types in Carbon provide
access to a shared L-value with an API subset that can help model important
requirements like ensuring usage is exclusively by way of a _thread-safe_
interface subset of an otherwise _thread-compatible_ type.

The `const T` type has the same representation as `T` with the same field names,
but all of its field types are also `const`-qualified. Other than fields, all
other members `T` are also members of `const T`, and impl lookup ignores the
`const` qualification. There is an implicit conversion from `T` to `const T`,
but not the reverse. L-value-to-R-value conversion can be performed on
`const T`.

It is expected that `const T` will overwhelmingly occur as part of a
[pointer](#pointers), as the express purpose is to reference an L-value. Carbon
will support conversions between pointers to `const`-qualified types that follow
the same rules as used in C++ to avoid inadvertent loss of
`const`-qualification.

## Lifetime overloading

One use case not obviously or fully addressed by these designs in Carbon is
overloading function calls by observing the lifetime of arguments. The use case
here would be selecting different implementation strategies for the same
function or operation based on whether an argument lifetime happens to be ending
and viable to move-from.

Carbon currently intentionally leaves this use case unaddressed. There is a
fundamental scaling problem in this style of overloading: it creates a
combinatorial explosion of possible overloads similar to other permutations of
indirection models. Consider a function with N parameters that would benefit
from lifetime overloading. If each one benefits _independently_ from the others,
we would need 2<sup>N</sup> overloads to express all the possibilities.

Carbon should initially see if code can be designed without this facility. Some
of the tools needed to avoid it are suggested above such as the
[consuming](#consuming-function-parameters) input pattern. But it is possible
that more will be needed in practice. It would be good to identify the specific
and realistic Carbon code patterns that cannot be expressed with the tools in
this proposal in order to motivate a minimal extension. Some candidates based on
functionality already proposed here or for [classes](/docs/design/classes.md):

-   Allow overloading between `addr me` and `me` in methods. This is among the
    most appealing as it _doesn't_ have the combinatorial explosion. But it is
    also very limited as it only applies to the implicit object parameter.
-   Allow overloading between `var` and non-`var` parameters.
-   Expand the `addr` technique from object parameters to all parameters, and
    allow overloading based on it.

Perhaps more options will emerge as well. Again, the goal isn't to completely
preclude pursuing this direction, but instead to try to ensure it is only
pursued based on a real and concrete need, and the minimal extension is adopted.
