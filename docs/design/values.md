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
        -   [Value acquisition](#value-acquisition)
        -   [Direct initialization](#direct-initialization)
        -   [Copy initialization](#copy-initialization)
        -   [Temporary materialization](#temporary-materialization)
-   [Binding patterns and local variables with `let` and `var`](#binding-patterns-and-local-variables-with-let-and-var)
    -   [Local variables](#local-variables)
    -   [Consuming function parameters](#consuming-function-parameters)
-   [Reference expressions](#reference-expressions)
    -   [Entire reference expressions](#entire-reference-expressions)
    -   [Durable reference expressions](#durable-reference-expressions)
    -   [Ephemeral reference expressions](#ephemeral-reference-expressions)
-   [Value expressions](#value-expressions)
    -   [Comparison to C++ parameters](#comparison-to-c-parameters)
    -   [Polymorphic types](#polymorphic-types)
    -   [Interop with C++ `const &` and `const` methods](#interop-with-c-const--and-const-methods)
    -   [Escape hatches for value addresses in Carbon](#escape-hatches-for-value-addresses-in-carbon)
-   [Initializing expressions](#initializing-expressions)
    -   [Function calls and returns](#function-calls-and-returns)
        -   [Deferred initialization from values and references](#deferred-initialization-from-values-and-references)
        -   [Declared `returned` variable](#declared-returned-variable)
-   [Extended types](#extended-types)
    -   [Initializing results](#initializing-results)
    -   [Extended type conversions](#extended-type-conversions)
        -   [Type conversions](#type-conversions)
        -   [Category conversions](#category-conversions)
-   [Pointers](#pointers)
    -   [Reference types](#reference-types)
    -   [Pointer syntax](#pointer-syntax)
    -   [Dereferencing customization](#dereferencing-customization)
-   [`const`-qualified types](#const-qualified-types)
-   [Lifetime overloading](#lifetime-overloading)
-   [Value representation and customization](#value-representation-and-customization)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

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

There are three primary expression categories in Carbon:

-   [_Value expressions_](#value-expressions) produce abstract, read-only
    _values_ that cannot be modified or have their address taken.
-   [_Reference expressions_](#reference-expressions) refer to _objects_ with
    _storage_ where a value may be read or written and the object's address can
    be taken.
-   [_Initializing expressions_](#initializing-expressions) require a result
    location to be provided implicitly when evaluating the expression. The
    expression then initializes an object in that location. These are used to
    model function returns, which can construct the returned value directly in
    the caller's storage.

Expressions in one category can be implicitly converted to any other primary
category when needed. The primitive conversion steps used are:

-   [_Value acquisition_](#value-acquisition) forms a value expression from the
    current value of the object referenced by a reference expression.
-   [_Direct initialization_](#direct-initialization) converts a value
    expression into an initializing expression.
-   [_Copy initialization_](#copy-initialization) converts a reference
    expression into an initializing expression.
-   [_Temporary materialization_](#temporary-materialization) converts an
    initializing expression into a reference expression.

These conversion steps combine to provide the transitive conversion table:

|               From: | value                     | reference | initializing          |
| ------------------: | ------------------------- | --------- | --------------------- |
|        to **value** | ==                        | acquire   | materialize + acquire |
|    to **reference** | direct init + materialize | ==        | materialize           |
| to **initializing** | direct init               | copy init | ==                    |

Reference expressions are divided into 2x2 sub-categories: they can be either
[_ephemeral_](#ephemeral-reference-expressions) or
[_durable_](#durable-reference-expressions), and either _entire_ or
_non-entire_.

Ephemeral reference expressions are formed through temporary materialization,
and have restrictions on how they are used. In contrast, durable reference
expressions refer to storage that outlives the expression, and typically has a
declared name. Entire reference expressions can only refer to complete objects,
whereas non-entire reference expressions can refer to both complete objects and
sub-objects (such as class fields and base class sub-objects). As a consequence,
only entire reference expressions can be destructively moved.

> **Future work:** This means that pointer-dereference expressions are
> non-entire, but we will presumably want to be able to destructively move from
> them. We need to figure out how to support that without violating the
> invariant that a live object has live fields.

Value acquisition and copy initialization can be applied to any reference
expression, but materialization only produces ephemeral entire reference
expressions. An entire reference expression can be implicitly converted to
non-entire; this has no run-time effect because it merely discards static
object-completeness information. Non-entire reference expressions can only be
converted to entire reference expressions by round-tripping through
copy-initialization and materialization. Non-durable-reference expressions
cannot be implicitly converted to durable reference expressions at all.

> **TODO:** Determine how these reference sub-categories relate to memory-safety
> properties like uniqueness, and make sure their names are aligned with
> memory-safety terminology.

#### Value acquisition

We call forming a value expression from a reference expression _value
acquisition_. This forms a value expression that will evaluate to the value of
the object in the referenced storage of the reference expression. It may do this
by eagerly reading that value into a machine register, lazily reading that value
on-demand into a machine register, or in some other way modeling that abstract
value.

See the [value expressions](#value-expressions) section for more details on the
semantics of value expressions.

#### Direct initialization

This is the first way we have of initializing storage of an object. There may
not be storage for the source of this initialization, as the value expression
used for the initialization may be in a machine register or simply be abstractly
modeled like a source literal. A canonical example here is zeroing an object.

#### Copy initialization

This initializes storage for an object based on some other object which already
has initialized storage. A classic example here are types which can be copied
trivially and where this is implemented as a `memcpy` of their underlying bytes.

#### Temporary materialization

We use temporary materialization when we need to initialize an object by way of
storage, but weren't provided dedicated storage and can simply acquire the
result as a value afterward.

> **Open question:** The lifetimes of temporaries is not yet specified.

## Binding patterns and local variables with `let` and `var`

A [_value binding pattern_](/docs/design/README.md#binding-patterns) introduces
a name that is a [_value expression_](#value-expressions) and is called a _value
binding_. This is the desired default for many pattern contexts, especially
function parameters. Values are a good model for "input" function parameters
which are the dominant and default style of function parameters:

```carbon
fn Sum(x: i32, y: i32) -> i32 {
  // `x` and `y` are value expressions here. We can use their value, but not
  // modify them or take their address.
  return x + y;
}
```

Value bindings require the matched expression to be a _value expression_,
converting it into one as necessary.

A _variable pattern_ is introduced with the `var` keyword. The matched
expression must be an ephemeral entire reference expression (which typically
requires the matched expression to be materialized); the `var` pattern takes
ownership of the newly-allocated temporary storage it refers to, which extends
its lifetime to the end of the enclosing scope. The subpattern is then matched
against a _durable_ entire reference expression to the object in that storage.

> **Open question:** This implies that `var field: T = F().field;` doesn't
> perform any copies or moves on `T`. This, in turn, implies that the storage
> for `field` must be laid out as part of a complete `typeof(F())` object
> layout, which is initialized by the call to `F()`. All other members of that
> layout are immediately destroyed, and their storage is theoretically reusable
> after that point, but it's unclear if this is the right default, or how to
> enable user code to override that default when it's the wrong tradeoff.

A _reference binding pattern_ is a binding pattern that is nested under a `var`
pattern. It introduces a name called a _reference binding_ that is a
[durable reference expression](#durable-reference-expressions) to an object
within the variable pattern's storage.

```carbon
fn MutateThing(ptr: i64*);

fn Example() {
  // `1` starts as a value expression, which is what a value binding expects.
  let x: i64 = 1;

  // `2` also starts as a value expression, but the variable pattern requires it
  // to be converted to an ephemeral entire reference expression by using the
  // value `2` to initialize temporary storage, which the variable pattern
  // takes ownership of. The reference binding pattern is then bound to a
  // durable reference to the newly-initialized object.
  var y: i64 = 2;

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
keyword. The `let` introducer begins a value pattern which works the same as the
default patterns in other contexts. The `var` introducer immediately begins a
variable pattern.

-   `let` _identifier_`:` _( expression |_ `auto` _)_ `=` _value_`;`
-   `var` _identifier_`:` _( expression |_ `auto` _) [_ `=` _value ]_`;`

These are just simple examples of binding patterns used directly in local
declarations. Local `let` and `var` declarations build on Carbon's general
[pattern matching](/docs/design/pattern_matching.md) design. `var` declarations
implicitly start off within a `var` pattern. `let` declarations introduce
patterns that bind values by default, the same as function parameters and most
other pattern contexts.

The general pattern matching model also allows nesting `var` sub-patterns within
a larger pattern that defaults to matching values. For example, we can combine
the two local declarations above into one destructuring declaration with an
inner `var` pattern here:

```carbon
fn DestructuringExample() {
  // Both `1` and `2` start as value expressions. The `x` binding directly
  // matches `1`. For `2`, the variable pattern requires it to be converted to
  // an ephemeral entire reference expression by using the value `2` to
  // initialize temporary storage, which the variable pattern takes ownership
  // of. The reference binding `y` is then bound to a durable reference to the
  // newly-initialized object.
  let (x: i64, var y: i64) = (1, 2);

  // Just like above, we can take the address and mutate `y`:
  MutateThing(&y);

  // ❌ And this remains an error:
  MutateThing(&x);
}
```

If `auto` is used in place of the type for a local binding pattern,
[type inference](type_inference.md) is used to automatically determine the
variable's type.

These local bindings introduce names scoped to the code block in which they
occur, which will typically be marked by an open brace (`{`) and close brace
(`}`).

### Consuming function parameters

Just as part of a `let` declaration can use a `var` prefix to become a variable
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
signature of the function changes the expression category required for arguments
in the caller. These arguments are required to be _ephemeral entire reference
expressions_, potentially being converted into such an expression if necessary,
whose storage will be dedicated-to and owned-by the function parameter.

This pattern serves the same purpose as C++'s pass-by-value when used with types
that have non-trivial resources attached to pass ownership into the function and
consume the resource. But rather than that being the seeming _default_, Carbon
makes this a use case that requires a special marking on the declaration.

## Reference expressions

_Reference expressions_ refer to _objects_ with _storage_ where a value may be
read or written and the object's address can be taken.

Reference expressions can be either _durable_ or _ephemeral_. These refine the
_lifetime_ of the underlying storage and provide safety restrictions reflecting
that lifetime. Reference expressions can also be either _entire_ or
_non-entire_, depending on whether the referenced object is known to be complete
(rather than a sub-object of another object).

### Entire reference expressions

An _entire reference expression_ is one that is statically known to refer to a
complete object. Other references are _non-entire_. Durable and ephemeral
reference expressions can both be either entire or non-entire (although
non-entire ephemeral references are rare). Unless otherwise specified, an
expression or operation that produces a reference produces a non-entire
reference.

Note that a non-entire reference expression still _might_ refer to a complete
object; the language rules just don't _guarantee_ that is does. As a result, an
entire reference can be implicitly converted to a non-entire reference (with the
same durability), because this merely discards the knowledge that the object is
complete. By the same token, there is no context that requires a non-entire
reference; only contexts that accept both, that accept only entire references,
or that don't accept references at all.

Currently, the only context that requires an entire reference is the scrutinee
of a `var` pattern, which is required to be an entire ephemeral reference (and
is [converted](#category-conversions) to that category if necessary).

> **Note:** This extends the lifetime of the reference, so it must be possible
> to determine _which_ temporary an ephemeral entire reference refers to, so
> that the implementation knows which lifetime to extend. Under the current
> language rules, this can be done statically.

> **Open question:** Should we extend the language in ways that would force that
> determination to be dynamic? For example, should we allow
> `if c then r1 else r2` to be an entire ephemeral reference expression if `r1`
> and `r2` are? As a more extreme example, should we support functions that take
> and return entire ephemeral references?

There are several kinds of expressions that produce entire references. For
example:

-   The name of an object introduced with a
    [variable binding pattern](pattern_matching.md#name-binding-patterns) (in
    other words, a name that was declared with `var <name> : <type>`) is a
    durable entire reference.
-   a member access expression `x.member` or `x.(member)` is an entire reference
    if `x` is an initializing or entire ephemeral reference expression with a
    struct or tuple type.
-   The result of materialization is an entire ephemeral reference.
-   When a [tuple pattern](pattern_matching.md#tuple-patterns) or
    [struct pattern](pattern_matching.md#struct-patterns) is matched with an
    ephemeral entire reference scrutinee, that scrutinee is destructured into
    ephemeral entire references to its elements, which are then matched with the
    corresponding subpatterns.

### Durable reference expressions

_Durable reference expressions_ are those where the object's storage outlives
the full expression and the address could be meaningfully propagated out of it
as well.

There are several contexts where durable reference expressions are required. For
example:

-   [Assignment statements](/docs/design/assignment.md) require the
    left-hand-side of the `=` to be a durable reference. This stronger
    requirement is enforced before the expression is rewritten to dispatch into
    the `Carbon.Assign.Op` interface method.
-   [Address-of expressions](#pointer-syntax) require their operand to be a
    durable reference and compute the address of the referenced object.
-   [`ref` binding patterns](pattern_matching.md#name-binding-patterns) require
    their scrutinee to be a durable reference.
-   If a function's [extended return type](#function-calls-and-returns) contains
    `ref` tags, `return` statements require the corresponding parts of the
    operand to be durable reference expressions.

There are also several kinds of expressions that produce durable references. For
example:

-   Names of objects introduced with a
    [reference binding](#binding-patterns-and-local-variables-with-let-and-var):
    `x`
-   Dereferenced [pointers](#pointers): `*p`
-   Names of subobjects through member access to some other durable reference
    expression: `x.member` or `p->member`
-   [Indexing](/docs/design/expressions/indexing.md) into a type similar to
    C++'s `std::span` that implements `IndirectIndexWith`, or indexing into any
    type with a durable reference expression such as `local_array[i]`.
-   Calls to functions whose
    [extended return types](#function-calls-and-returns) contain `ref`.

Durable reference expressions can only be produced _directly_ by one of these
expressions. They are never produced by converting one of the other expression
categories into a reference expression.

### Ephemeral reference expressions

We call the reference expressions formed through
[temporary materialization](#temporary-materialization) _ephemeral reference
expressions_. They still refer to an object with storage, but it may be storage
that will not outlive the full expression, and so it can't be used where a
durable reference is expected.

> **Future work:** The current design does not support mutating ephemeral
> references (or initializing expressions): assigning to an ephemeral reference
> is disallowed directly, and invoking mutating methods is disallowed because
> the `ref self` parameter can only bind to a durable reference. In C++ it's
> unusual but not rare to intentionally mutate a temporary, such as in a
> builder-style method chain (for example `MakeFoo().SetBar().AddBaz()`), so
> Carbon will need to provide some interop and migration target for that kind of
> code.

There is one context that requires an ephemeral reference expression in Carbon:
the scrutinee of a
[`var` pattern](#binding-patterns-and-local-variables-with-let-and-var) (which
also requires the reference to be entire).

There are only a few ways to produce an ephemeral reference expression. Most
notably:

-   The result of materialization is an entire ephemeral reference.
-   A member access expression `x.member` or `x.(member)` is an ephemeral
    reference if `x` is an initializing or ephemeral reference.
-   When a [tuple pattern](pattern_matching.md#tuple-patterns) or
    [struct pattern](pattern_matching.md#struct-patterns) is matched with an
    initializing or ephemeral reference scrutinee, that scrutinee is
    destructured into ephemeral references to its elements, which are then
    matched with the corresponding subpatterns.

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
enable generic code that needs a single type model that will have consistently
good performance.

When forming a value expression from a reference expression, Carbon
[acquires](#value-acquisition) the value of the referenced object. This allows
immediately reading from the object's storage into a machine register or a copy
if desired, but does not require that. The read of the underlying object can
also be deferred until the value expression itself is used. Once an object is
bound to a value expression in this way, any mutation to the object or its
storage ends the lifetime of the value binding, and makes any use of the value
expression an error.

> Note: this is _not_ intended to ever become "undefined behavior", but instead
> just "erroneous". We want to be able to detect and report such code as having
> a bug, but do not want unbounded UB and are not aware of important
> optimizations that this would inhibit.
>
> _Open issue:_ We need a common definition of erroneous behavior that we can
> use here (and elsewhere). Once we have that, we should cite it here.

> Note: this restriction is also **experimental** -- we may want to strengthen
> or weaken it based on experience, especially with C++ interop and a more
> complete memory safety story.

Even with these restrictions, we expect to make values in Carbon useful in
roughly the same places as `const &`s in C++, but with added efficiency in the
case where the values can usefully be kept in machine registers. We also
specifically encourage a mental model of a `const &` with extra efficiency.

The actual _representation_ of a value when bound, especially across function
boundaries, is [customizable](#value-representation-and-customization) by the
type. The defaults are based around preserving the baseline efficiency of C++'s
`const &`, but potentially reading the value when that would be both correct and
reliably more efficient, such as into a machine register.

### Comparison to C++ parameters

While these are called "values" in Carbon, they are not related to "by-value"
parameters as they exist in C++. The semantics of C++'s by-value parameters are
defined to create a new local copy of the argument, although it may move into
this copy.

Carbon's values are much closer to a `const &` in C++ with extra restrictions
such as allowing copies under "as-if" and preventing taking the address.
Combined, these restrictions allow implementation strategies such as in-register
parameters.

### Polymorphic types

Value expressions and value bindings can be used with
[polymorphic types](/docs/design/classes.md#inheritance), for example:

```carbon
base class MyBase { ... }

fn UseBase(b: MyBase) { ... }

class Derived {
  extend base: MyBase;
  ...
}

fn PassDerived() {
  var d: Derived = ...;
  // Allowed to pass `d` here:
  UseBase(d);
}
```

This is still allowed to create a copy or to move, but it must not _slice_. Even
if a copy is created, it must be a `Derived` object, even though this may limit
the available implementation strategies.

> **Future work:** The interaction between a
> [custom value representation](#value-representation-and-customization) and a
> value expression used with a polymorphic type needs to be fully captured.
> Either it needs to restrict to a `const ref` style representation (to prevent
> slicing) or it needs to have a model for the semantics when a different value
> representation is used.

### Interop with C++ `const &` and `const` methods

While value expressions cannot have their address taken in Carbon, they should
be interoperable with C++ `const &`s and C++ `const`-qualified methods. This
will in-effect "pin" some object (potentially a copy or temporary) into memory
and allow C++ to take its address. Without supporting this, values would likely
create an untenable interop ergonomic barrier. However, this does create some
additional constraints on value expressions and a way that their addresses can
escape unexpectedly.

Despite interop requiring an address to implement, C++ allows `const &`
parameters to bind to temporary objects where that address doesn't have much
meaning and might not be valid once the called function returns. As a
consequence, we don't expect C++ interfaces using a `const &` to misbehave in
practice.

> **Future work:** when a type customizes its
> [value representation](#value-representation-and-customization), as currently
> specified this will break the use of `const &` C++ APIs with such a value. We
> should extend the rules around value representation customization to require
> that either the representation type can be converted to (a copy) of the
> customized type, or implements an interop-specific interface to compute a
> `const` pointer to the original object used to form the representation object.
> This will allow custom representations to either create copies for interop or
> retain a pointer to the original object and expose that for interop as
> desired.

Another risk is exposing Carbon's value expressions to `const &` parameters in
this way, as C++ allows casting away `const`. However, in the absence of
`mutable` members, casting away `const` does not make it safe to _mutate_
through a `const &` parameter (or a `const`-qualified method). C++ allows
`const &` parameters and `const` member functions to access objects that are
_declared_ `const`. These objects cannot be mutated, even if `const` is removed,
exactly the same as Carbon value expressions. In fact, these kinds of mutations
[break in real implementations](https://cpp.compiler-explorer.com/z/KMhTondaK).
The result is that Carbon's value expressions will work similarly to
`const`-declared objects in C++, and will interop with C++ code similarly well.

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
  fn ValueMemberFunction(self);
  fn RefMemberFunction(ref self: const Self);
}

fn F(s_value: S) {
  // This is fine.
  s_value.ValueMemberFunction();

  // This requires an unsafe marker in the syntax.
  s_value.unsafe RefMemberFunction();
}
```

The specific tradeoff here is covered in a proposal
[alternative](/proposals/p002006-values-variables-pointers-and-references.md#value-expression-escape-hatches).

## Initializing expressions

Storage in Carbon is initialized using _initializing expressions_. Their
evaluation takes a _result location_ as an implicit input, and produces an
initialized object at that location, although that object may still be
_unformed_.

**Future work:** More details on initialization and unformed objects should be
added to the design from the proposal
[#257](https://github.com/carbon-language/carbon-lang/pull/257), see
[#1993](https://github.com/carbon-language/carbon-lang/issues/1993). When added,
it should be linked from here for the details on the initialization semantics
specifically.

The simplest form of initializing expressions are value or durable reference
expressions that are converted into an initializing expression. Value
expressions are written directly into the storage to form a new object.
Reference expressions have the object they refer to copied into a new object in
the provided storage.

**Future work:** The design should be expanded to fully cover how copying is
managed and linked to from here.

There are no syntactic contexts in Carbon that always require an initializing
expression, and no expression syntax that always produces an initializing
expression. By default, function call expressions are initializing expressions,
and correspondingly the operand of `return` is required to be an initializing
expression, but this default can be overridden by the
[function signature](#function-calls-and-returns).

Initializing expressions can also be created implicitly, when attempting to
convert an expression into an ephemeral entire reference expression
(particularly to match a `var` pattern): the expression is first converted to an
initializing expression if necessary, and then temporary storage is materialized
to act as its output, and as the referent of the resulting ephemeral reference
expression.

### Function calls and returns

The [result](#extended-types) of a function call can have an almost arbitrary
extended type. The return clause of a function signature consists of `->`
followed by an _extended return type_, an expression-like syntax that specifies
not only the type component but also the extended type of the function call's
result. `return` expressions in the function body are expected to have that
extended type, and are converted to it if necessary. When a function is declared
without a return clause, it behaves from the caller's point of view as if the
return clause were `-> ()`, but `return` statements in the function body don't
take operands (and can be omitted at the end of the function).

In the common case, the extended return type is a type expression, in which case
calls are modeled directly as initializing expressions -- they require storage
as an input and when evaluated cause that storage to be initialized with an
object. This means that when a function call is used to initialize some variable
pattern as here:

```carbon
fn CreateMyObject() -> MyType {
  return <return-expression>;
}

var x: MyType = CreateMyObject();
```

The `<return-expression>` in the `return` statement actually initializes the
storage provided for `x`. There is no "copy" or other step.

In the body of such a function, all `return` statement expressions are required
to be initializing expressions and in fact initialize the storage provided to
the function's call expression. This in turn causes the property to hold
_transitively_ across an arbitrary number of function calls and returns. The
storage is forwarded at each stage and initialized exactly once.

More generally, the syntax and semantics of an extended return type are as
follows:

-   _return-clause_ ::= `->` _extended-return-type_
-   _extended-return-type_ ::= _nesting-extended-return-type_ |
    _auto-extended-return-type_
-   _nesting-extended-return-type_ ::= _expression-extended-return-type_ |
    _proper-extended-return-type_

Extended return types can usually be nested, but syntaxes involving `auto` can
only occur at top level. We further divide nesting extended return types into
expressions and "proper" extended return types, but this is just a technical
means of avoiding formal ambiguity in the grammar; it has no greater
significance.

-   _category-tag_ ::= `val` | `ref` | `var`

These tags are used to specify "value", "non-entire durable reference", or
"initializing" expression category (respectively). Note that there is no way to
express an entire or ephemeral reference category in an extended return type.

-   _auto-extended-return-type_ ::= _category-tag_? `auto`

This denotes a primitive extended type with runtime phase and a deduced type
component. The category is determined by _category-tag_ if present, or
"initializing" otherwise.

-   _proper-extended-return-type_ ::= _category-tag_ _expression_

This denotes a primitive extended type with runtime phase, category
_category-tag_, and type "_expression_ `as type`".

-   _expression-extended-return-type_ ::= _expression_

An expression with no _category-tag_ is equivalent to "`var` _expression_".

-   _proper-extended-return-type_ ::= `(` [_expression-extended-return-type_
    `,`]\* _proper-extended-return-type_ [`,` _nesting-extended-return-type_]\*
    `,`? `)`

A tuple literal of extended return types denotes a tuple extended type whose
sub-extended-types are specified by the comma-separated elements. To avoid
formal ambiguity, this grammar rule requires at least one of the
sub-extended-types to be proper.

-   _expression-field-extended-type_ ::= _designator_ `:`
    _expression-extended-return-type_
-   _proper-field-extended-type_ ::= _designator_ `:`
    _proper-extended-return-type_
-   _field-extended-type_ ::= _field-decl_
-   _field-extended-type_ ::= _proper-field-extended-type_
-   _proper-extended-return-type_ ::= `{` [_expression-field-extended-type_
    `,`]\* _proper-extended-type_ [`,` _field-extended-type_]\* `}`

A struct literal of extended return types denotes a struct extended type whose
field names and their extended types are specified by the comma-separated field
extended types. To avoid formal ambiguity, this grammar rule requires at least
one of the field extended types to be proper.

> **Open question:** Should there be a way to specify symbolic or template phase
> in extended return types?

#### Deferred initialization from values and references

TODO: This section needs to be updated to reflect the addition of `-> val`
returns in
[proposal #5434](/proposals/p005434-ref-parameters-arguments-returns-and-val-returns.md).
This section could be replaced by a statement that initializing returns may be
replaced by value returns when that is safe and correct, moving much of this
content into a description of how value returns works.

Carbon also makes the evaluation of function calls and return statements tightly
linked in order to enable more efficiency improvements. It allows the actual
initialization performed by the `return` statement with its expression to be
deferred from within the body of the function to the caller initializer
expression if it can simply propagate a value or reference expression to the
caller that is guaranteed to be alive and available to the caller.

Consider the following code:

```carbon
fn SelectSecond(first: Point, second: Point, third: Point) -> Point {
  return second;
}

fn UsePoint(p: Point);

fn F(p1: Point, p2: Point) {
  UsePoint(SelectSecond(p2, p1, p2));
}
```

The call to `SelectSecond` must provide storage for a `Point` that can be
initialized. However, Carbon allows an implementation of the actual
`SelectSecond` function to not initialize this storage when it reaches
`return second`. The expression `second` is a name bound to the call's argument
value expression, and that value expression is necessarily valid in the caller.
Carbon in this case allows the implementation to merely communicate that the
returned expression is a name bound to a specific value expression argument to
the call, and the caller _if necessary_ should initialize the temporary storage.
This in turn allows the caller `F` to recognize that the value expression
argument (`p1`) is already valid to pass as the argument to `UsePoint` without
initializing the temporary storage from it and reading it back out of that
storage.

None of this impacts the type system and so an implementation can freely select
specific strategies here based on concrete types without harming generic code.
Even if generics were to be implemented without monomorphization, for example
dynamic dispatch of object-safe interfaces, there is a conservatively correct
strategy that will work for any type.

This freedom mirrors that of [input values](#value-expressions) where might be
implemented as either a reference or a copy without breaking genericity. Here
too, many small types will not need to be lazy and simply eagerly initialize the
temporary which is implemented as an actual machine register. But for large
types or ones with associated allocated storage, this can reliably avoid
extraneous memory allocations and other costs.

Note that this flexibility doesn't avoid the call expression materializing
temporary storage and providing it to the function. Whether the function needs
this storage is an implementation detail. It simply allows deferring an
important case of initializing that storage from a value or reference expression
already available in the caller to the caller so that it can identify cases
where that initialization is not necessary.

**References:** This addresses an issue-for-leads about
[reducing the potential copies incurred by returns](https://github.com/carbon-language/carbon-lang/issues/828).

#### Declared `returned` variable

The model of initialization of returns also facilitates the use of
[`returned var` declarations](control_flow/return.md#returned-var). These
directly observe the storage provided for initialization of a function's return.

## Extended types

We typically treat the category and type of an expression as independent
properties. However, in some cases we need to deal with them as an integrated
whole. The _extended type_ of an expression captures all of the information
about it that is visible to the type system, while abstracting away all other
information about it. Thus, extended types are a generalization of types: what
we conventionally call "types" are really the types of objects and values,
whereas extended types are the types of expressions and patterns. The type
`Core.ExtType` represents the type of an extended type constant, just as `type`
represents the type of an object type constant.

A _primitive extended type_ currently consists of a type, an expression
category, an expression phase, and optionally a constant value (which is present
if and only if the expression phase is not "runtime"). When dealing with
primitive extended types, which is the common case, we can treat each of those
properties as independent. For convenience, in this section we will use the
notation `<T, C, P, V>` to represent a primitive extended type with type `T`,
category `C`, phase `P` and value `V`, but this is not Carbon syntax.

Other extended types are called _composite extended types_, and there are two
kinds:

A _tuple extended type_ can be thought of as a tuple of extended types, just as
a tuple type can be thought of as a tuple of types. The extended type of a tuple
literal is a tuple extended type, whose elements are the extended types of the
literal elements.

> **TODO:** Extend this to support variadic extended types.

A _struct extended type_ can be thought of as a struct whose fields are extended
types, just as a struct type can be thought of as a struct whose fields are
types. The extended type of a struct literal is a struct extended type with the
same field names, whose values are the extended types of the corresponding
fields of the struct literal.

The _type component_ of an extended type is defined as follows:

-   The type component of a primitive extended type `<T, C, P, V>` is `T`.
-   The type component of a tuple extended type is a tuple of the type
    components of its elements.
-   The type component of a struct extended type is a struct whose field names
    are the field names of the struct extended type and whose field types are
    the type components of the corresponding elements.

The _category component_ and _phase component_ of an extended type are defined
likewise. The category component of a struct extended type is called a _struct
category_, and the category component of a tuple extended type is called a
_tuple category_.

The type of an expression is the type component of the expression's extended
type.

Evaluating an expression produces a _result_. It can be defined recursively in
terms of the expression's extended type:

-   The result of an initializing expression is an
    [initializing result](#initializing-results).
-   The result of a value expression is a value.
-   The result of a reference expression is a reference of the same kind.
-   The result of an expression with tuple extended type is a tuple of results.
-   The result of an expression with struct extended type is a struct of
    results.

An expression and its result always have the same extended type.

The code that accesses the result of an expression is said to _consume_ that
result, and every primitive-extended-type result is consumed exactly once
(except in certain narrow contexts where the result is known not to be
initializing). If a result isn't explicitly accessed, such as when the
expression is used as a statement, it is said to be _discarded_, which consumes
it in the absence of an explicit consumer. Discarding an initializing result
materializes and then immediately destroys it. Discarding an entire ephemeral
reference destroys the object it refers to. Discarding a value or any other kind
of reference is a no-op.

### Initializing results

As discussed earlier, evaluation of an initializing expression takes as an input
the result location that it initializes, which is implicitly provided by the
context in which the evaluation takes place. In some cases, the context may
obtain the location from its own context, and so on. For example:

```carbon
class C {
  private var i: i32;

  fn Make() -> C {
    return {.i = 0};
  }
}

fn F() -> C {
  return C.Make();
}

fn G() {
  var c: C = F();
}
```

By default, a function call is an initializing expression, and a `return`
statement initializes the call's result location (which is passed as a hidden
output parameter). So when the declaration of `c` is evaluated, its storage is
implicitly passed into `F()` as an output parameter, which is initialized by the
`return` statement inside `F`. When that `return` statement is evaluated to
initialize the result location, it likewise implicitly passes the storage into
`C.Make()` as an output parameter, which is initialized by the `return`
statement inside `C.Make`. Finally, that `return` statement initializes the
result location (which is still the location of `c`'s storage) by
[direct initialization](#direct-initialization) from the value expression
`{.i = 0}`.

Notice that the implicit storage parameter propagates "backwards", into an
expression from the code that uses its result. In order to simplify the
description of the language, we usually won't explicitly discuss the result
locations of initializing expressions, or how they're propagated. Instead, this
propagation is encapsulated inside the _initializing result_, which is the
notional result of an initializing expression.

Whenever an initializing result is consumed, that implicitly means that the
consumer passes a result location into the evaluation of the initializing
expression. The source of that location depends on the consumer:

-   If the consumer is a temporary materialization conversion, the result
    location is newly-allocated temporary storage (which the consumer may
    subsequently lifetime-extend to durable storage).
-   If the consumer is a `return` statement, and the initializing result
    corresponds to an initializing sub-extended-type of the function's return
    extended type, the result location is the implicit output parameter
    corresponding to that initializing sub-extended-type.

### Extended type conversions

A conversion between extended types can be broken down into up to three steps:
type conversion, category conversion, and phase conversion. These convert the
extended type to a particular target type, category, and phase component
(respectively). These steps aren't fully orthogonal: type conversions can change
the category and phase components as a byproduct, and category conversions can
change the phase component. However, category conversions can't change the type
component, and phase conversions can't change either of the other two, so
converting the type, then category, then phase, ensures that we converge on the
desired result.

Any of these steps may be omitted, depending on whether the context imposes
requirements on the corresponding component. Most commonly, an operand position
requires its operand to have a primitive extended type with a particular
category, usually with a particular type, and sometimes with a particular phase.

Phase conversions cannot change the extended type structure; they can only apply
primitive phase conversions to primitive sub-extended-types. Type and category
conversions are more complex, and are covered in the next two sections.

Note that these rules will implicitly convert between primitive and composite
extended types in both directions (except that a composite containing references
cannot be converted to a primitive extended type). As a result, although the
difference between primitive and composite extended types is observable by way
of overloading, it can't reliably carry any higher-level meaning, and should be
used only as an optimization tool.

Note that this section describes the _logical structure_ of extended type
conversions. As such, it primarily describes them "breadth-first", as a sequence
of operations that each applies to the whole expression by recursively operating
on its parts. However, the _physical execution_ of these conversions is actually
depth-first, applying as many operations as possible to a minimal subexpression
before moving on to the next one. The details of that process are described
[here](pattern_matching.md#evaluation-order).

#### Type conversions

See [here](expressions/implicit_conversions.md) for overall information about
type conversions. Conversions involving struct, tuple, and array types are
described here because of their unique interactions with extended types of
expressions.

> **TODO:** A forthcoming proposal is expected to update the type conversion
> interfaces to permit user-defined conversions to depend on the extended type
> of the input, and customize the extended type of the output. Once that is
> done, these "built in" conversions should be presented as implementations of
> those interfaces, possibly with some "magic" for things like introspecting on
> struct field names.

Each of the conversions described in this section is explicit if and only if it
invokes another explicit type conversion. Otherwise, it is implicit.

A type conversion of an expression with primitive extended type to a
[compatible type](generics/terminology.md#compatible-types) just re-interprets
the expression's result with a new type, so it requires no run-time work, and
has the same category as the input expression.

A result `source` that has a struct type can be converted to a struct type
`Dest` if they have the same set of field names:

-   If the type of `source` is `Dest`, return `source`.
-   If `source` is a struct result, for each field name `F` in `Dest`,
    type-convert `source.F` to `Dest.F`. Return a struct result where each field
    `F` is set to the result of the corresponding conversion.
-   If `source` is a primitive result, convert it to a struct result by
    [extended type decomposition](#category-conversions), and then type-convert
    the result to `Dest` and return the result.

Note that the sub-conversions invoked here are not necessarily defined; if so,
the conversion itself is not defined.

There is a conversion to a class type `Dest` from a result `source` that has a
struct type, if there is a conversion from `source` to a struct type that has
the same field names as `Dest` (including a `.base` field if `Dest` is a derived
class), with the same types, in the same order. The conversion type-converts
`source` to that struct type, category-converts that to an initializing
expression of the struct type, and then reinterprets it as an initializing
expression of `Dest` (which is layout-compatible with the struct type by
construction).

Note that some fields of an object may be initialized directly by the evaluation
of the source expression, while others may be initialized by the conversions
described here. The conversions initialize fields in their declaration order,
but the evaluation of the source expression always happens before any of the
conversions, and happens in the source expression's lexical order, so the fields
of an object are not necessarily initialized in declaration order.

Conversions between tuple types are defined in the same way, treating tuples as
structs that have fields named `.0`, `.1`, etc, in numerical order.

There is a conversion to `array(T, N)` from any expression with a tuple extended
type of exactly `N` elements, whose type components are convertible to `T`. The
conversion is an initializing expression, which type-converts each source
element to `T`, and initializes the corresponding array element from the result
of that conversion.

#### Category conversions

_Extended type composition_ converts an expression of composite extended type
with consistent category to a primitive extended type as follows (where `min` as
applied to phases uses the ordering "runtime" < "symbolic" < "template"):

-   An expression of tuple extended type
    `(<T1, C, P1, V1>, <T2, C, P2, V2>, ... <TN, C, PN, VN>)` can be converted
    to a primitive extended type
    `<(T1, T2, ..., TN), C, min(P1, P2, ..., PN), (V1, V2, ... VN)>`.
-   An expression of struct extended type
    `{.a = <Ta, C, Pa, Va>, .b = <Tb, C, Pb, Vb>, ... .z = <Tz, C, Pz, Vz>}` can
    be converted to a primitive extended type
    `<{.a = Ta, .b = Tb, ... .z = Tz}, C, min(Pa, Pb, ... Pz), {.a = Va, .b = Vb, ... .z = Vz}>`.

When `C` is "value", composition forms a value representation of the aggregate
from value representations of the elements. When `C` is "initializing", it
transforms initializing expressions for each element into a single initializing
expression that initializes the whole aggregate. `C` cannot be a reference
category, because an aggregate of references to independent objects can't be
replaced by a reference to a single aggregate object in a single step.

_Extended type decomposition_ is the inverse of extended type composition. It
converts an expression with primitive extended type to a composite extended type
as follows:

-   An expression with primitive extended type `<(T0, T1, ..., TN), C, P, V>`
    can be converted to a tuple extended type
    `(<T0, CC, P, V.0>, <T1, CC, P, V.1>, ... <TN, CC, P, V.N>)`.
-   An expression with primitive extended type
    `<{.a = Ta, .b = Tb, ... .z = Tz}, C, P, V>` can be converted to a struct
    extended type
    `{.a = <Ta, CC, P, V.a>, .b = <Tb, CC, P, V.b>, ... .z = <Tz, CC, P, V.z>}`.

The category `CC` of the resulting sub-extended-types is the same as `C`, with
two exceptions:

-   If `C` is "durable entire reference", `CC` will be "durable non-entire
    reference", because the sub-extended-types don't refer to complete objects.
    This doesn't apply to ephemeral entire references, because in that case
    extended type decomposition implicitly ends the lifetime of the original
    aggregate, promoting its elements to complete objects with independent
    lifetimes.
-   If `C` is "initializing", the original expression is materialized before it
    is decomposed, so `CC` will be "ephemeral entire reference".

By convention, extended type decomposition is a no-op when applied to an
expression with struct or tuple extended type.

_Category conversion_ converts an expression to have a given category component
without changing its type component. The conversion works by combining extended
type composition and decomposition with primitive category conversions, and is
defined recursively:

-   If the target category component is a tuple, the source extended type must
    have a type component that is a tuple type with the same arity. Convert the
    source to a tuple extended type by extended type decomposition, and then
    category-convert each source sub-extended-type to the corresponding target
    sub-category.
-   If the target category component is a struct, the source extended type must
    have a type component that is a struct type with the same set of field names
    in the same order. Convert the source to a struct extended type by extended
    type decomposition, and then category-convert each source sub-extended-type
    to the corresponding target sub-category.
-   If the target category is a primitive category `C`:
    -   If the source extended type is primitive, convert to `C` by applying
        primitive category conversions.
    -   If the source extended type is composite and `C` is a reference
        category, category-convert the source extended type to "initializing",
        and then convert the result to `C` by applying primitive category
        conversions.
    -   If the source extended type is composite and `C` is not a reference
        category, category-convert each source sub-extended-type to `C`, and
        then convert the aggregate result of these conversions to `C` by
        extended type composition.

## Pointers

Pointers in Carbon are the primary mechanism for _indirect access_ to storage
containing some value. Dereferencing a pointer is one of the primary ways to
form a [_durable reference expression_](#durable-reference-expressions).

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

TODO: This section needs to be updated to reflect
[proposal #5434](/proposals/p005434-ref-parameters-arguments-returns-and-val-returns.md).

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

Third, Carbon doesn't provide a straightforward way to avoid the syntactic
distinction between indirect access and direct access.

For a full discussion of the tradeoffs of these design decisions, see the
alternatives considered section of [P2006]:

-   [References in addition to pointers](/proposals/p002006-values-variables-pointers-and-references.md#references-in-addition-to-pointers)
-   [Syntax-free or automatic dereferencing](/proposals/p002006-values-variables-pointers-and-references.md#syntax-free-or-automatic-dereferencing)
-   [Exclusively using references](/proposals/p002006-values-variables-pointers-and-references.md#exclusively-using-references)

### Pointer syntax

The type of a pointer to a type `T` is written with a postfix `*` as in `T*`.
Dereferencing a pointer is a [_reference expression_] and is written with a prefix
`*` as in `*p`:

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
extensively in [#523] and are summarized in the
[proposal](/proposals/p002006-values-variables-pointers-and-references.md#alternative-pointer-syntaxes).

[#523]: https://github.com/carbon-language/carbon-lang/issues/523

Carbon also supports an infix `->` operation, much like C++. However, Carbon
directly defines this as an exact rewrite to `*` and `.` so that `p->member`
becomes `(*p).member` for example. This means there is no overloaded or
customizable `->` operator in Carbon the way there is in C++. Instead,
customizing the behavior of `*p` in turn customizes the behavior of `p->`.

**Future work:** As [#523] discusses, one of the primary challenges of the C++
syntax is the composition of a prefix dereference operation and other postfix or
infix operations, especially when chained together such as a classic C++
frustrations of mixes of dereference and indexing: `(*(*p)[42])[13]`. Where
these compositions are sufficiently common to create ergonomic problems, the
plan is to introduce custom syntax analogous to `->` that rewrites down to the
grouped dereference. However, nothing beyond `->` itself is currently provided.
Extending this, including the exact design and scope of extension desired, is a
future work area.

### Dereferencing customization

Carbon should support user-defined pointer-like types such as _smart pointers_
using a similar pattern as operator overloading or other expression syntax. That
is, it should rewrite the expression into a member function call on an
interface. Types can then implement this interface to expose pointer-like
_user-defined dereference_ syntax.

The interface might look like:

```carbon
interface Pointer {
  let ValueT:! Type;
  fn Dereference(self) -> ValueT*;
}
```

Here is an example using a hypothetical `TaggedPtr` that carries some extra
integer tag next to the pointer it emulates:

```carbon
class TaggedPtr(T:! Type) {
  var tag: Int32;
  var ptr: T*;
}
external impl [T:! Type] TaggedPtr(T) as Pointer {
  let ValueT:! T;
  fn Dereference(self) -> T* { return self.ptr; }
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

```carbon
impl [T:! Type] T* as Pointer {
  let ValueT:! Type = T;
  fn Dereference(self) -> T* { return self; }
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

Note that `const T` is a type qualification and is generally orthogonal to
expression categories or what form of pattern is used, including for object
parameters. Notionally, it can occur both with `ref` and value object
parameters. However, on value patterns, it is redundant as there is no
meaningful distinction between a value expression of type `T` and type
`const T`. For example, given a type and methods:

```carbon
class X {
  fn Method(self);
  fn ConstMethod(self: const Self);
  fn RefMethod(ref self);
  fn RefConstMethod(ref self: const Self);
}
```

The methods can be called on different kinds of expressions according to the
following table:

| Expression category: | `let x: X` <br/> (value) | `let x: const X` <br/> (const value) | `var x: X` <br/> (reference) | `var x: const X` <br/> (const reference) |
| -------------------: | ------------------------ | ------------------------------------ | ---------------------------- | ---------------------------------------- |
|        `x.Method();` | ✅                       | ✅                                   | ✅                           | ✅                                       |
|   `x.ConstMethod();` | ✅                       | ✅                                   | ✅                           | ✅                                       |
|     `x.RefMethod();` | ❌                       | ❌                                   | ✅                           | ❌                                       |
| `x.RefConstMethod()` | ❌                       | ❌                                   | ✅                           | ✅                                       |

The `const T` type has the same representation as `T` with the same field names,
but all of its field types are also `const`-qualified. Other than fields, all
other members `T` are also members of `const T`, and impl lookup ignores the
`const` qualification. There is an implicit conversion from `T` to `const T`,
but not the reverse. Conversion of reference expressions to value expressions is
defined in terms of `const T` reference expressions to `T` value expressions.

It is expected that `const T` will largely occur as part of a
[pointer](#pointers), as the express purpose is to form reference expressions.
The precedence rules are even designed for this common case, `const T*` means
`(const T)*`, or a pointer-to-const. Carbon will support conversions between
pointers to `const`-qualified types that follow the same rules as used in C++ to
avoid inadvertent loss of `const`-qualification.

The syntax details of `const` are also covered in the
[type operators](/docs/design/expressions/type_operators.md) documentation.

## Lifetime overloading

One potential use case that is not obviously or fully addressed by these designs
in Carbon is overloading function calls by observing the lifetime of arguments.
The use case here would be selecting different implementation strategies for the
same function or operation based on whether an argument lifetime happens to be
ending and viable to move-from.

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

-   Allow overloading between `ref self` and `self` in methods. This is among
    the most appealing as it _doesn't_ have the combinatorial explosion. But it
    is also very limited as it only applies to the implicit object parameter.
-   Allow overloading between `var` and non-`var` parameters.
-   Allow overloading between `ref` and non-`ref` parameters in general.

Perhaps more options will emerge as well. Again, the goal isn't to completely
preclude pursuing this direction, but instead to try to ensure it is only
pursued based on a real and concrete need, and the minimal extension is adopted.

## Value representation and customization

The representation of a value expression is especially important because it
forms the calling convention used for the vast majority of function parameters
-- function inputs. Given this importance, it's important that it is predictable
and customizable by the value's type. Similarly, while Carbon code must be
correct with either a copy or a reference-based implementation, we want which
implementation strategy is used to be a predictable and customizable property of
the type of a value.

A type can optionally control its value representation using a custom syntax
similar to customizing its [destructor](/docs/design/classes.md#destructors).
This syntax sets the representation to some type uses a keyword `value_rep` and
can appear where a member declaration would be valid within the type:

```carbon
class SomeType {
  value_rep = RepresentationType;
}
```

**Open question:** The syntax for this is just placeholder, using a placeholder
keyword. It isn't final at all and likely will need to change to read well.

The provided representation type must be one of the following:

-   `const Self` -- this forces the use of a _copy_ of the object.
-   `const ref` -- this forces the use of a [_pointer_](#pointers) to the
    original object, but with the `const` API subset.
-   A custom type that is not `Self`, `const Self`, or a pointer to either.

If the representation is `const Self` or `const ref`, then the type fields will
be accessible as [_value expressions_](#value-expressions) using the normal
member access syntax for value expressions of a type. These will be implemented
by either accessing a copy of the object in the non-pointer case or a pointer to
the original object in the pointer case. A representation of `const Self`
requires copying to be valid for the type. This provides the builtin
functionality but allows explicitly controlling which representation should be
used.

If no customization is provided, the implementation will select one based on a
set of heuristics. Some examples:

-   Non-copyable types and polymorphic types would use a `const ref`.
-   Small objects that are trivially copied in a machine register would use
    `const Self`.

When a custom type is provided, it must not be `Self`, `const Self`, or a
pointer to either. The type provided will be used on function call boundaries
and as the implementation representation for value bindings and other value
expressions referencing an object of the type. A specifier of `value_rep = T;`
will require that the type containing that specifier satisfies the constraint
`impls ReferenceImplicitAs where .T = T` using the following interface:

```carbon
interface ReferenceImplicitAs {
  let T:! type;
  fn Convert(ref self: const Self) -> T;
}
```

Converting a reference expression into a value expression for such a type calls
this customization point to form a representation object from the original
reference expression.

When using a custom representation type in this way, no fields are accessible
through a value expression. Instead, only methods can be called using member
access, as they simply bind the value expression to the `self` parameter.
However, one important method can be called -- `.(ImplicitAs(T).Convert)()`.
This implicitly converting a value expression for the type into its custom
representation type. The customization of the representation above and
`impls ReferenceImplicitAs where .T = T` causes the class to have a builtin
`impl as ImplicitAs(T)` which converts to the representation type as a no-op,
exposing the object created by calling `ReferenceImplicitAs.Convert` on the
original reference expression, and preserved as a representation of the value
expression.

Here is a more complete example of code using these features:

```carbon
class StringView {
  private var data_ptr: Char*;
  private var size: i64;

  fn Make(data_ptr: Char*, size: i64) -> StringView {
    return {.data_ptr = data_ptr, .size = size};
  }

  // A typical readonly view of a string API...
  fn ExampleMethod(self) { ... }
}

class String {
  // Customize the value representation to be `StringView`.
  value_rep = StringView;

  private var data_ptr: Char*;
  private var size: i64;

  private var capacity: i64;

  impl as ReferenceImplicitAs where .T = StringView {
    fn Op(ref self: const Self) -> StringView {
      // Because this is called on the String object prior to it becoming
      // a value, we can access an SSO buffer or other interior pointers
      // of `self`.
      return StringView.Make(self.data_ptr, self.size);
    }
  }

  // We can directly declare methods that take `self` as a `StringView` which
  // will cause the caller to implicitly convert value expressions to
  // `StringView` prior to calling.
  fn ExampleMethod(self: StringView) { self.ExampleMethod(); }

  // Or we can use a value binding for `self` much like normal, but the
  // implementation will be constrained because of the custom value rep.
  fn ExampleMethod2(self: String) {
    // Error due to custom value rep:
    self.data_ptr;

    // Fine, this uses the builtin `ImplicitAs(StringView)`.
    (self as StringView).ExampleMethod();
  }

  // Note that even though the `Self` type is `const` qualified here, this
  // cannot be called on a `String` value! That would require us to convert to a
  // `StringView` that does not track the extra data member.
  fn Capacity(ref self: const Self) -> i64 {
    return self.capacity;
  }
}
```

It is important to note that the _representation_ type of a value expression is
just its representation and does not impact the name lookup or type. Name lookup
and `impl` search occur for the same type regardless of the expression category.
But once a particular method or function is selected, an implicit conversion can
occur from the original type to the representation type as part of the parameter
or receiver type. In fact, this conversion is the _only_ operation that can
occur for a value whose type has a customized value representation.

The example above also demonstrates the fundamental tradeoff made by customizing
the value representation of a type in this way. While it provides a great deal
of control, it may result in some surprising limitations. Above, a method that
is classically available on a C++ `const std::string&` like querying the
capacity cannot be implemented with the customized value representation because
it loses access to this additional state. Carbon allows type authors to make an
explicit choice about whether they want to work with a restricted API and
leverage a custom value representation or not.

**Open question:** Beyond the specific syntax used where we currently have a
placeholder `value_rep = T;`, we need to explore exactly what the best
relationship is with the customization point. For example, should this syntax
immediately forward declare `impl as ReferenceImplicitAs where .T = T`, thereby
allowing an out-of-line definition of the `Convert` method and `... where _` to
pick up the associated constant from the syntax. Alternatively, the syntactic
marker might be integrated into the `impl` declaration for `ReferenceImplicitAs`
itself.

## Alternatives considered

-   [No `var` introducer keyword](/proposals/p000339-var-statement.md#no-var-introducer-keyword)
-   [Name of the `var` statement introducer](/proposals/p000339-var-statement.md#name-of-the-var-statement-introducer)
-   [Colon between type and identifier](/proposals/p000339-var-statement.md#colon-between-type-and-identifier)
-   [Type elision](/proposals/p000339-var-statement.md#type-elision)
-   [Type ordering](/proposals/p000618-var-ordering.md#type-ordering)
-   [Elide the type instead of using `auto`](/proposals/p000851-variable-type-inference.md#elide-the-type-instead-of-using-auto)
-   [Value expression escape hatches](/proposals/p002006-values-variables-pointers-and-references.md#value-expression-escape-hatches)
-   [References in addition to pointers](/proposals/p002006-values-variables-pointers-and-references.md#references-in-addition-to-pointers)
-   [Syntax-free or automatic dereferencing](/proposals/p002006-values-variables-pointers-and-references.md#syntax-free-or-automatic-dereferencing)
-   [Exclusively using references](/proposals/p002006-values-variables-pointers-and-references.md#exclusively-using-references)
-   [Alternative pointer syntaxes](/proposals/p002006-values-variables-pointers-and-references.md#alternative-pointer-syntaxes)
-   [Alternative syntaxes for locals](/proposals/p002006-values-variables-pointers-and-references.md#alternative-syntaxes-for-locals)
-   [Mixed expression categories](/proposals/p005545-expression-form-basics.md#mixed-expression-categories)
-   [Don't implicitly convert to less-primitive forms](/proposals/p005545-expression-form-basics.md#dont-implicitly-convert-to-less-primitive-forms)
-   [Use `exprtype` and `expr` keywords](/proposals/p007254-replace-and-with-keywords-and-contextual-defaults.md#use-exprtype-and-expr-keywords)

## References

-   [Proposal #257: Initialization of memory and values][#257]
-   [Proposal #339: `var` statement][#339]
-   [Proposal #618: `var` ordering][#618]
-   [Proposal #851: auto keyword for vars][#851]
-   [Proposal #2006: Values, variables, and pointers][#2006]
-   [Proposal #5545: Expression form basics][#5545]
-   [Proposal #7254: Replace `:!` and `:?` with keywords and contextual
    defaults][#7254]

[#257]: /proposals/p000257-initialization-of-memory-and-variables.md
[#339]: /proposals/p000339-var-statement.md
[#618]: /proposals/p000618-var-ordering.md
[#851]: /proposals/p000851-variable-type-inference.md
[#2006]: /proposals/p002006-values-variables-pointers-and-references.md
[#5545]: /proposals/p005545-expression-form-basics.md
[#7254]: /proposals/p007254-replace-and-with-keywords-and-contextual-defaults.md
