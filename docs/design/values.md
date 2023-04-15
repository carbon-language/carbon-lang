# Values, variables, and pointers

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Values, objects, and expressions](#values-objects-and-expressions)
    -   [Expression categories](#expression-categories)
-   [Binding patterns and local variables with `let` and `var`](#binding-patterns-and-local-variables-with-let-and-var)
    -   [Local variables](#local-variables)
    -   [Consuming function parameters](#consuming-function-parameters)
-   [Reference expressions](#reference-expressions)
    -   [Durable reference expressions](#durable-reference-expressions)
    -   [Ephemeral reference expressions](#ephemeral-reference-expressions)
-   [Value expressions](#value-expressions)
    -   [Comparison to C++ parameters](#comparison-to-c-parameters)
    -   [Value representation and customization](#value-representation-and-customization)
    -   [Polymorphic types](#polymorphic-types)
    -   [Interop with C++ `const &` and `const` methods.](#interop-with-c-const--and-const-methods)
    -   [Escape hatches for value addresses in Carbon](#escape-hatches-for-value-addresses-in-carbon)
-   [Initializing expressions](#initializing-expressions)
-   [Pointers](#pointers)
    -   [Reference types](#reference-types)
    -   [Syntax](#syntax)
    -   [Syntax-free dereference and address-of](#syntax-free-dereference-and-address-of)
    -   [Dereferencing customization](#dereferencing-customization)
-   [`const`-qualified types](#const-qualified-types)
-   [Lifetime overloading](#lifetime-overloading)

<!-- tocstop -->

## Values, objects, and expressions

Carbon has both abstract _values_ and concrete _objects_. Carbon _values_ are
things like `42`, `true`, and `i32` (a type value). Carbon _objects_ have
_storage_ where values can be read and written. Storage also allows taking the
address of an object in memory in Carbon.

Both objects and values can be nested within each other. For example
`(true, true)` is both a value and also contains two sub-values. When a
two-tuple is stored somewhere, it is both a tuple-typed object and contains two
subobjects.

These terms are important components in the describing the semantics of Carbon
code, but they aren't sufficient. We also need to explicitly and precisely talk
about the Carbon _expressions_ that produce or reference values and objects.
Categorizing the expressions themselves allows us to be more precise and
articulate important differences not captured without looking at the expression
itself.

### Expression categories

There are three expression categories in Carbon:

-   [_Value expressions_](#value-expressions) produce abstract, read-only
    _values_ that cannot be modified or have its address taken.
-   [_Reference expressions_](#reference-expressions) refer to _objects_ with
    _storage_ where a value may be read or written and the object's address can
    be taken.
    -   [_Durable reference expressions_](#durable-reference-expressions) are
        reference expressions which cannot refer to _temporary_ storage, but
        must refer to some storage that outlives the full expression.
    -   [_Ephemeral reference expressions_](#ephemeral-reference-expressions)
        are reference expressions which _can_ refer to temporary storage.
-   [_Initializing expressions_](#initializing-expressions) which require
    storage to be provided as an input and initializes an object in that
    storage.

The syntax and syntactic context of an expression fully determines both the
expressions initial category and which if any conversions from one category to
another must take place. Carbon specifically works to avoid involving the _type
system_ in this determination as a way to reduce the complexity required within
the type system.

The general conversions and the semantics implied between these categories are
below:

-   An _initializing expression_ can be formed from:
    -   A _value expression_ by using the value to initialize an object in the
        provided storage, analogous to
        [direct initialization](https://en.cppreference.com/w/cpp/language/direct_initialization)
        in C++.
    -   A _durable reference expression_ by copying the referenced object into
        the new storage, analogous to
        [copy initialization](https://en.cppreference.com/w/cpp/language/copy_initialization)
        in C++.
-   An _ephemeral reference expression_ can be formed from:
    -   A _durable reference expression_ trivially.
    -   An _initializing expression_ by materializing temporary storage for an
        object that is initialized and then referencing that object.
-   A _durable reference expression_ cannot be produced by converting from
    another expression category.
-   A _value expression_ can be formed from a _reference expression_ by reading
    the value from its storage.

Multiple steps of these conversions can be combined. For example, to produce a
value expression from an initializing expression, first the initializing
expression is converted to an ephemeral reference expression by materializing
temporary storage and initializing an object stored there, and that is then
turned into the desired value expression by reading the value from that stored
object.

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

A _variable pattern_ can be introduced with the `var` keyword to create an
object with storage when match.d Every binding pattern name introduced within a
variable pattern is called a _variable binding_ and forms a
[_reference_expression_](#reference-expressions) to an object within the
variable pattern's storage when used. Variable patterns require their matched
expression to be an _initializing expression_ and provide their storage to it to
be initialized.

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
pattern and bind names that will form reference expressions to the variable's
storage, so can function parameters:

```carbon
fn Consume(var x: SomeData) {
  // We can mutate and use variable that `x` refers to here.
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
makes this a use case that requires a special marking on the declaration.

## Reference expressions

_Reference expressions_ refer to _objects_ with _storage_ where a value may be
read or written and the object's address can be taken. There are two
sub-categories of reference expressions: _durable_ and _ephemeral_.

Calling a [method](TODO) on a reference expression where the method's `self`
parameter has an [`addr` specifier](TODO) can always implicitly take the address
of the referred-to object. This address is passed as a [pointer](#pointers) to
the `self` parameter for such methods.

### Durable reference expressions

_Durable reference expressions_ are those where the object's storage outlives
the full expression and the address could be meaningfully propagated out of it
as well.

There are two expressions that require one of their operands to be a durable
reference expression in Carbon:

-   [Assignment expressions](TODO) require the left-hand-side of the `=` to be a
    durable reference. This stronger requirement is enforced before the
    expression is rewritten to dispatch into the `Carbon.Assign.Op` interface
    method.
-   [Address-of expressions](TODO) require their operand to be a durable
    reference and compute the address of the referenced object.

There are several kinds of expressions that produce durable references in
Carbon:

-   Names of objects introduced with a
    [variable binding](#binding-patterns-and-local-variables-with-let-and-var):
    `x`
-   Dereferenced [pointers](#pointers): `*p`
-   Names of subobjects through member access: `x.member` or `p->member`
-   [Indexing](#indexing): `array[i]`

There is no way to convert another category of expression into a durable
reference expression, they always directly refer to some declared variable
binding.

### Ephemeral reference expressions

_Ephemeral reference expressions_ still refer to storage, but it may be storage
materialized for a temporary that will not outlive the full expression. Their
address can only be taken implicitly as part of a method call.

The only expressions in Carbon that directly require an ephemeral reference are
method calls where the method is marked with the `addr` specifier. However, they
are frequently indirectly required to enable converting an initializing
expression into a value expression.

Ephemeral reference expressions can only be formed by conversion from another
expression category. In the trivial case, a durable reference expression can be
used. They can also be formed by materializing temporary storage to provide to
an initializing expression, and referencing the initialized object in that
storage.

**Future work:** The current design allows directly requiring an ephemeral
reference for `addr`-methods because this replicates the flexibility in C++ --
very few C++ methods are L-value-ref-qualified which would have a similar effect
to `addr`-methods requiring a durable reference expression. This is leveraged
frequently in C++ for builder APIs and other patterns. However, Carbon provides
more tools in this space than C++ already, and so it may be worth evaluating
whether we can switch `addr`-methods to the same restrictions as assignment and
`&`. Temporaries would never have their address escaped (in a safe way) in that
world and there would be fewer different kinds of entities. But this is reserved
for future work as we should be very careful about the expressivity hit being
tolerable both for native-Carbon API design and for migrated C++ code.

## Value expressions

A value cannot be mutated, cannot have its address taken, and may not have
storage at all or a stable address of storage. Values are abstract constructs
like function input parameters and constants. They can be formed in two ways --
a literal expression like `42`, or by reading the value of some stored object.

A core goal of values in Carbon is to provide a single model that can get both
the efficiency of passing by value when working with small types such as those
that fit into a machine register, but also the efficiency of minimal copies when
working with types where a copy would require extra allocations or other costly
resources. This directly helps programmers by providing a simpler model to
select the mechanism of passing function inputs. But it is also important to
enable generic code that needs a single type model that will have generically
good performance.

To achieve this goal, a Carbon program must in general behave equivalently with
value expressions that are implemented as a _reference_ to the original object
or as either a _copy_ or _move_ if that would be valid for the type. However,
using a copy or a move is purely optional and an optimization. Value expressions
are valid with both uncopyable and unmovable types.

**Experimental:** We currently make an additional requirement that helps ensure
this equivalence will be true and allows us to detect the most risky cases where
it would not be true: we require that once a value is formed by reading a stored
object, that original object must not be mutated prior to the last use of the
value, or any transitive value. We consider this restriction experimental as we
may want to strengthen or weaken it based on our experience with Carbon code
using these constructs, and especially interoperating with C++.

Even with these restrictions, we expect to make values in Carbon useful in
roughly the same places as `const &`s in C++, but with added efficiency in the
case where the values can usefully be kept in machine registers. We also
specifically encourage a mental model of a `const &` with extra efficiency.

### Comparison to C++ parameters

While these are called "values" in Carbon, they are not related to "by-value"
parameters as they exist in C++. The semantics of C++'s by-value parameters are
defined to create a new local copy of the argument, although it may move into
this copy.

Carbon's values are much closer to a `const &` in C++ with extra restrictions
such as allowing copies under "as-if" and preventing taking the address.
Combined, these restrictions allow implementation strategies such as in-register
parameters.

### Value representation and customization

The representation of a value expression is especially important because it
forms the calling convention used for the vast majority of function parameters
-- function inputs. Given this importance, it's important that it is predictable
and customizable by the value's type. Similarly, while Carbon code must be
correct with either a copy or a reference-based implementation, we want which
implementation strategy is used to be a predictable and customizable property of
the type of a value.

A type can optionally control its value representation using a custom syntax
similar to customizing its [destructor](TODO). This syntax sets the
representation to some type uses a keyword `value_rep` and can appear where a
member declaration would be valid within the type:

```carbon
class SomeType {
  value_rep = RepresentationType;
}
```

**Open question:** The syntax for this is just placeholder, using a placeholder
keyword. It isn't final at all and likely will need to change to read well.

The provided representation type must be one of the following:

-   `const Self` -- this forces the use of a _copy_ of the object.
-   `const Self *` -- this forces the use of a [_pointer_](#pointers) to the
    original object.
-   A custom type that is not `Self`.

If the representation is `const Self` or `const Self *`, then the type fields
will be accessible as [_value expressions_](#value-expressions) using the normal
member access syntax for value expressions of a type. These will be implemented
by either accessing a copy of the object in the non-pointer case or a pointer to
the original object in the pointer case. A representation of `const Self`
requires copying to be valid for the type. This provides the builtin
functionality but allows explicitly controlling which representation should be
used. If no customization is provided, the implementation will select one based
on a set of heuristics.

When a custom type is provided, it must not be `Self` (or the other two cases).
The type provided will be used on function call boundaries and as the
implementation representation for `let` bindings and other value expressions
referencing an object of the type. A specifier of `value_rep = T;` will require
that the type `impls ReferenceImplicitAs(T)` using the following interface:

```carbon
interface ReferenceImplicitAs(T:! type) {
  fn Op[addr self: const Self*]() -> T;
}
```

Converting a reference expression into a value expression for such a type calls
this customization point to form a representation object from the original
reference expression.

When using a custom representation type in this way, no fields are accessible
through a value expression. Instead, only methods can be called using member
access, as they simply bind the value expression to the `self` parameter.
However, one important method can be called -- `.(ImplicitAs(T).Convert()`. This
implicitly converting a value expression for the type into its custom
representation type. The customization of the representation above causes the
class to have a builtin `impl as ImplicitAs(T)` which converts to the
representation type as a no-op, exposing the object created by the
`ReferenceImplicitAs` operation, and preserved as a representation of the value
expression.

Here is a more complete example of code using these features:

```carbon
class StringView {
  // A typical readonly view of a string API...
  fn ExampleMethod[self: Self]() { ... }
}

class String {
  // Customize the value representation to be `StringView`.
  value_rep = StringView;

  private var data_ptr: Char*;
  private var size: i64;

  // We can directly declare methods that take `self` as a `StringView` which
  // will cause the caller to implicitly convert value expressions to
  // `StringView` prior to calling.
  fn ExampleMethod[self: StringView]() { self.ExampleMethod(); }

  // Or we can use a value binding for `self` much like normal, but the
  // implementation will be constrained because of the custom value rep.
  fn ExampleMethod2[self: String]() {
    // Error due to custom value rep:
    self.data_ptr;

    // Fine, this uses the builtin `ImplicitAs(StringView)`.
    (self as StringView).ExampleMethod();
  }

  impl as ReferenceImplicitAs(StringView) {
    fn Op[addr self: const Self*]() -> StringView {
      // Because this is called on the String object prior to it becoming
      // a value, we can access an SSO buffer or other interior pointers
      // of `self`.
      return StringView::Create(self->data_ptr, self->size);
    }
  }
}
```

It is important to note that the _representation_ type of a value expression is
just its representation and does not impact the name lookup or type. Name lookup
and `impl` search occur for the same type regardless of the expression category.
But once a particular method or function is selected, an implicit conversion can
occur from the original type to the representation type as part of the parameter
or receiver type. In fact, this conversion is the _only_ operation that can
occur for a customized representation type, wherever it is necessary as
implemented.

### Polymorphic types

Value expressions and value bindings can be used with
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

While value expressions cannot have their address taken in Carbon, they should
be interoperable with C++ `const &`s and C++ `const`-qualified methods. This
will in-effect "pin" some object (potentially a copy or temporary) into memory
and allow C++ to take its address. Without supporting this, values would likely
create an untenable interop ergonomic barrier. However, this does create some
additional constraints on value expressions and a way that their addresses can
escape unexpectedly.

Despite interop requiring an address to implement, the address isn't guaranteed
to be stable or useful or point back to some original object necessarily. The
ability of the implementation to introduce copies or a temporary specifically
for the purpose of the interop remains.

**Future work:** when a type customizes its value representation, as currently
specified this will break the use of `const &` C++ APIs with such an value. We
should extend the rules around value representation customization to require
that either the representation type can be converted to (a copy) of the
customized type, or implements an interop-specific interface to compute a
`const` pointer to the original object used to form the representation object.
This will allow custom representations to either create copies for interop or
retain a pointer to the original object and expose that for interop as desired.

### Escape hatches for value addresses in Carbon

**Open question:** It may be necessary to provide some amount of escape hatch
for taking the address of values. The
[C++ interop](#interop-with-c-const--and-const-methods) above already takes
their address functionally. Currently, this is the extent of an escape hatch to
the restrictions on values.

If a further escape hatch is needed, this kind of fundamental weakening of the
semantic model would be a good case for some syntactic marker like Rust's
`unsafe`, although rather than a region, it would seem better to tie it directly
to the operation in question. For example:

```carbon
class S {
  fn ValueMemberFunction[me: Self]();
  fn AddrMemberFunction[addr me: const Self*]();
}

fn F(s_value: S) {
  // This is fine.
  s_value.ValueMemberFunction();

  // This requires an unsafe marker in the syntax.
  s_value.unsafe MutableMemberFunction();
}
```

## Initializing expressions

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

**Future work:** Add explicit designs for these use cases and link to them here.

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
Dereferencing a pointer is a [_reference expression_] and is written with a
prefix `*` as in `*p`:

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

**Future work:** As also covered extensively in
[#523](https://github.com/carbon-language/carbon-lang/issues/523), one of the
primary challenges of the C++ syntax is the composition of a prefix dereference
operation and other postfix or infix operations, especially when chained
together such as a classic C++ frustrations of mixes of dereference and
indexing: `(*(*p)[42])[13]`. Where these compositions are sufficiently common to
create ergonomic problems, the plan is to introduce custom syntax analogous to
`->` that rewrites down to the grouped dereference. However, nothing beyond `->`
itself is currently provided. Extending this, including the exact design and
scope of extension desired, is a future work area.

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
syntax within function bodies, but this proposal suggests deferring that in
order to better understand the extent and importance of that use case. If and
when it is considered, a direction based around a way to bind a name to a
reference expression in a pattern appears to be a promising technique.
Alternatively, there are various languages with implicit- or
automatic-dereference designs that might be considered.

A closely related concern to syntax-free dereference is syntax-free address-of.
Here, Carbon supports one very narrow form of this: implicitly taking the
address of the implicit object parameter of member functions. Currently that is
the only place with such an implicit affordance. It is designed to be
syntactically sound to extend to other parameters, but currently that is not
planned.

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
  fn Dereference[self: Self]() -> ValueT*;
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
  fn Dereference[self: Self]() -> T* { return self.ptr; }
}

fn Test(arg: TaggedPtr(T), dest: TaggedPtr(TaggedPtr(T))) {
  **dest = *arg;
  *dest = arg;
}
```

There is one tricky aspect of this. The function in the interface which
implements a pointer-like dereference must return a raw pointer which the
language then actually dereferences to form a reference expression similar to
that formed by `var` declarations. This interface is implemented for normal
pointers as a no-op:

```
impl [T:! Type] T* as Pointer {
  let ValueT:$ Type = T;
  fn Dereference[self: Self]() -> T* { return self; }
}
```

Dereference expressions such as `*x` are syntactically rewritten to use this
interface to get a raw pointer and then that raw pointer is dereferenced. If we
imagine this language level dereference to form a reference expression as a
unary `deref` operator, then `(*x)` becomes
`(deref (x.(Pointer.Dereference)()))`.

Carbon will also use a simple syntactic rewrite for implementing `x->Method()`
as `(*x).Method()` without separate or different customization.

## `const`-qualified types

Carbon provides the ability to qualify a type `T` with the keyword `const` to
get a `const`-qualified type: `const T`. This is exclusively an API-subsetting
feature in Carbon -- for more fundamentally "immutable" use cases, value
expressions and bindings should be used instead. Pointers to `const`-qualified
types in Carbon provide access to an object with an API subset that can help
model important requirements like ensuring usage is exclusively by way of a
_thread-safe_ interface subset of an otherwise _thread-compatible_ type.

The `const T` type has the same representation as `T` with the same field names,
but all of its field types are also `const`-qualified. Other than fields, all
other members `T` are also members of `const T`, and impl lookup ignores the
`const` qualification. There is an implicit conversion from `T` to `const T`,
but not the reverse. Conversion of reference expressions to value expressions is
defined in terms of `const T` reference expressions to `T` value expressions.

It is expected that `const T` will overwhelmingly occur as part of a
[pointer](#pointers), as the express purpose is to form reference expressions.
Carbon will support conversions between pointers to `const`-qualified types that
follow the same rules as used in C++ to avoid inadvertent loss of
`const`-qualification.

## Lifetime overloading

One use case that is not obviously or fully addressed by these designs in Carbon
is overloading function calls by observing the lifetime of arguments. The use
case here would be selecting different implementation strategies for the same
function or operation based on whether an argument lifetime happens to be ending
and viable to move-from.

Carbon currently intentionally leaves this use case unaddressed. There is a
fundamental scaling problem in this style of overloading: it creates a
combinatorial explosion of possible overloads similar to other permutations of
indirection models. Consider a function with N parameters that would benefit
from lifetime overloading. If each parameter benefits _independently_ from the
others, as is commonly the case, we would need 2<sup>N</sup> overloads to
express all the possibilities.

Carbon will initially see if code can be designed without this facility. Some of
the tools needed to avoid it are suggested above such as the
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
