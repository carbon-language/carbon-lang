# Idioms

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [C++ dialect](#c-dialect)
-   [Abbreviations used in the code (AKA Carbon abbreviation decoder ring)](#abbreviations-used-in-the-code-aka-carbon-abbreviation-decoder-ring)
-   [`.def` files](#def-files)
    -   [EnumBase types](#enumbase-types)
-   [Index types](#index-types)
-   [ValueStore](#valuestore)
-   [Template metaprogramming](#template-metaprogramming)
    -   [Struct reflection](#struct-reflection)
    -   [Field detection](#field-detection)
-   [Local lambdas to reduce duplicate code](#local-lambdas-to-reduce-duplicate-code)
-   [Immediately invoked function expressions (IIFE)](#immediately-invoked-function-expressions-iife)
-   [Declarations in conditions](#declarations-in-conditions)
-   [CRTP or "Curiously recurring template pattern"](#crtp-or-curiously-recurring-template-pattern)
-   [Multiple inheritance](#multiple-inheritance)
-   [Defining constants usable in constexpr contexts](#defining-constants-usable-in-constexpr-contexts)

<!-- tocstop -->

## Overview

The toolchain implementation uses some implementation techniques that may not be
commonly found in typical C++ code.

## C++ dialect

The toolchain implementation does not use some C++ features, following
[Google's C++ style guide](https://google.github.io/styleguide/cppguide.html):

-   [Exceptions](https://google.github.io/styleguide/cppguide.html#Exceptions)
-   [Virtual base classes](https://google.github.io/styleguide/cppguide.html#Inheritance)
-   [RTTI](https://google.github.io/styleguide/cppguide.html#Run-Time_Type_Information__RTTI_)

## Abbreviations used in the code (AKA Carbon abbreviation decoder ring)

Note that abbreviations are typically only used in code, not comments (except
when referring to an entity from the code).

-   **Addr**: "address"
-   **Arg**: "argument"
-   **Decl**: "declaration"
-   **Expr**: "expression"
    -   **SubExpr**: "subexpression"
-   **Float**: "floating point"
-   **Init**: "initialization"
-   **Inst**: "instruction"
-   **Int**: "integer"
-   **Loc**: "location"
-   **Param**: "parameter"
-   **Paren**: "parenthesis"
-   **Ref**: "reference"
    -   **Deref**: "dereference"
-   **Subst**: "substitute"

Phrase abbreviations (where we have an abbreviation for a phrase, where we
wouldn't perform all of the abbreviations of those words individually):

-   **InitRepr**: "initializing representation"
-   **ObjectRepr**: "object representation"
-   **SemIR**: "semantics intermediate representation"
-   **ValueRepr**: "value representation"

## `.def` files

The Carbon toolchain uses a technique related to
[X-macros](https://en.wikipedia.org/wiki/X_macro) to generate code that operates
over a collection of types, enumerators, or another similar list of names. This
works as follows:

-   A `.def` file is provided, that is intended to be repeatedly included by way
    of `#include`.
-   The user of the `.def` defines a macro, with a name and a form specified by
    the `.def` file, for example
    `#define CARBON_EACH_WIDGET(Name) Scope::Name,`.
-   A `#include` of the `.def` file expands to `CARBON_EACH_WIDGET(Name1)`,
    `CARBON_EACH_WIDGET(Name2)`, ... for each widget name, and then `#undef`s
    the `CARBON_EACH_WIDGET` macro.

For example:

```cpp
enum Widgets {
#define CARBON_EACH_WIDGET(Name) Name,
#include "widgets.def"
}
```

... would expand to an enumeration definition with one enumerator per widget
name.

### EnumBase types

Most `.def` files will have a corresponding [EnumBase](/common/enum_base.h)
child class (if `widgets.def` has X-macros, `widgets.h` and `widgets.cpp` has
the `EnumBase` child class). These work similarly to an `enum class`, with the
addition of a `name()` function and `<<` stream operator support. Many also have
further utility functions for information related to the enum value.

In code, these types and values can be used directly in a `switch`. They will
convert to an internal _actual_ `enum class` for the `switch`, and receive
corresponding compiler safety checks that all enum values are handled.

## Index types

Carbon makes frequent use of
[IndexBase and IdBase](/toolchain/base/index_base.h). The `IndexBase` and
`IdBase` types are small wrappers around `int32_t` to provide a measure of
type-checking when passing around indices to vector-like storage types. The only
difference is that `IndexBase` supports all comparison operators, whereas
`IdBase` only supports equality comparison.

Variable naming will often have `_id` at the end to indicate that it corresponds
to an `IdBase`. This may include the full type, as in `operand_inst_id` being an
`InstId` for an operand.

A block is an array of ids. These will be indicated with either a `_block`
suffix or pluralization (for example, `param_refs` pluralizing `refs`).

The `ref` concept in a name means that there is an underlying instruction block,
but only a subset of instructions are present in the `refs` block. For example,
function parameters have a sequence, and also have a `refs` block with one entry
per parameter. The `refs` block allows parameters to be counted and accessed
directly, rather than through vector iteration.

## ValueStore

Many of Carbon's data types are stored in a
[ValueStore](/toolchain/base/value_store.h) or related type with similar
semantics (`sem_ir` has [several such classes](/toolchain/base/value_store.h)).
`ValueStore` links an indexing type to a value type with vector-like storage.
The indices typically use `IdBase`.

`ValueStore`s APIs follow the shape of simple array access and mutation:

-   `Add` which takes a value and returns the index.
-   `Get` takes an index and returns a reference to the value (possibly a
    constant reference).
-   Other vector-like functionality, including `size` or `Reserve`

ValueStores should be named after the type they contain. The index type used on
the value store should have a `using ValueType...` which indicates the stored
type. When taking a return of one of these functions, it's common to use `auto`
and rely on the name of the storage type to imply the returned type.

Some name mirroring examples are:

-   `ints` is a `ValueStore<IntId>`, which has an index type of `IntId` and a
    value type of `llvm::APInt`.

-   `functions` is a `ValueStore<SemIR::FunctionId>`, which has an index type of
    `SemIR::FunctionId` and a value type of `SemIR::` `Function`.

-   `strings` is a `ValueStore<StringId>`, which has an index type of
    `StringId`, but for copy-related reasons, uses `llvm::StringRef` for values.

There are also a number of wrappers around `ValueStore` that provide some
additional functionality and which are named with the `Store` suffix, such as
`InstStore` or `CanonicalValueStore`.

A fairly complete list of `ValueStore` (and `ValueStore` wrapper) uses should be
available on [checking's Context class].

<!-- google-doc-style-ignore -->

[checking's Context class]:
    https://github.com/search?q=repo%3Acarbon-language%2Fcarbon-lang+path%3Atoolchain%2Fcheck%2Fcontext.h+%2F%5Cw%2BStore%2F&type=code

<!-- google-doc-style-resume -->

## Template metaprogramming

TODO: show example patterns

-   InstLikeTypeInfo from toolchain/sem_ir/inst.h
-   templated using
-   std::declval
-   decltype
-   static_assert
-   if constexpr
-   template specialization, for example `Inst::FromRaw<T>` (maybe also type
    traits?)

### Struct reflection

The toolchain uses a primitive form of struct reflection to operate generically
over the fields in a typed `SemIR` instruction. This is implemented in
`common/struct_reflection.h`, and the interface to the functionality is
`StructReflection::AsTuple(your_struct)`, which converts the given struct into a
`std::tuple` containing the same fields in the same order.

### Field detection

The presence of specific fields in a struct with a specified type is detected
using the following idiom:

```cpp
// HasField<T> is true if T has a `U field` field of type FieldType.
template <typename T> concept HasField = requires (T x) {
  { &T::field } -> std::same_as<FieldType T::*>;
};
```

See `HasKindMemberAsField` in
[`toolchain/sem_ir/typed_insts.h`](/toolchain/sem_ir/typed_insts.h) for an
example.

## Local lambdas to reduce duplicate code

Sometimes code that would be repeated in a function is factored into a local
variable containing a
[lambda](https://en.cppreference.com/w/cpp/language/lambda):

```cpp
auto common_code = [&](AType param1, AnotherType param2) {
  // code that would otherwise be repeated
  ...
}
if (something) {
  common_code(...);
}
if (something_else) {
  common_code(...)
}
```

Compared to defining a new function, this has the advantage of being able to be
declared in context and access the local variables of the enclosing function.

## Immediately invoked function expressions (IIFE)

Instead of creating a separate function with its own name that will be called
once to produce the initial value for a variable, the function can be declared
inline and then immediately called.

This can be used for complex initialization, as in:

```cpp
// variable declaration
static const llvm::ArrayRef<std::byte> entropy_bytes =
    // initializer starts with a lambda
    []() -> llvm::ArrayRef<std::byte> {
      static llvm::SmallVector<std::byte> bytes;

      // a bunch of code

      // return the value to initialize the variable with
      return bytes;

      // finish defining the lambda, and then immediately invoke it
    }();
```

It can also be used inside a `CARBON_DCHECK` to avoid computation that is only
needed in debug builds:

```cpp
CARBON_DCHECK([&] {
  // a bunch of code

  // condition that will be tested by CARBON_DCHECK
  return complicated && multiple_parts;

// finish defining the lambda, and then immediately invoke it
}(), "Complicated things went wrong");
```

See a description of this technique on
[wikipedia](https://en.wikipedia.org/wiki/Immediately_invoked_function_expression).

## Declarations in conditions

The condition part of an `if` statement may contain a declaration with an
initializer followed by a semicolon (`;`) and then the proper boolean condition
expression, as in:

```cpp
if (auto verify = tree.Verify(); !verify.ok()) {
```

The condition can be replaced by a declaration entirely, as in:

```cpp
if (auto equals = context.ConsumeIf(Lex::TokenKind::Equal)) {
// Equivalent to:
if (auto equals = context.ConsumeIf(Lex::TokenKind::Equal); equals) {
```

or

```cpp
if (auto literal = bound_inst.TryAs<SemIR::IntegerLiteral>()) {
// Equivalent to:
if (auto literal = bound_inst.TryAs<SemIR::IntegerLiteral>(); literal) {
```

This is a common way of handling a function that returns an optional value.

See
[https://en.cppreference.com/w/cpp/language/if](https://en.cppreference.com/w/cpp/language/if)

## CRTP or "Curiously recurring template pattern"

[Curiously Recurring Template Pattern - cppreference.com](https://en.cppreference.com/w/cpp/language/crtp)

[Curiously recurring template pattern - Wikipedia](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)

[Google search](https://www.google.com/search?q=crtp+c%2B%2B)

Examples:

-   `template <typename DerivedT, ...>` in [enum_base.h](/common/enum_base.h)
-   `template <typename DerivedT>` in [ostream.h](/common/ostream.h)

## Multiple inheritance

We use multiple inheritance to support uses of
[CRTP](#crtp-or-curiously-recurring-template-pattern).

Example:

```cpp
struct NameScopeId : public IndexBase, public Printable<NameScopeId> {
```

## Defining constants usable in constexpr contexts

To declare a constant usable at compile time in `constexpr` contexts as a static
class member, we use this pattern:

Declaration:

```cpp
class Foo {
  // ...
  static const std::array<ElementType, ElementCount> MyTable;
  static constexpr auto ComputeMyTable()
      -> std::array<ElementType, ElementCount> { ... }
};
```

Definition:

```cpp
constexpr std::array<ElementType, ElementCount>
    Foo::MyTable = Foo::ComputeMyTable();
```

Note the `const` on the declaration does not match the `constexpr` on
definition, and that the definition is outside of the class body. This allows
the initializer to depend on the definition of the class.

Further note that this only works with static members of classes, not static
variables in functions.

Due to [a Clang bug](https://github.com/llvm/llvm-project/issues/85461), this
technique does not work in a class template. The following pattern can be used
instead:

```cpp
template <typename T>
class Foo {
  // ...
  template <typename Self = Foo>
  static constexpr auto MyValueImpl = Self();
  static constexpr const Foo& MyValue = MyValueImpl<>;
  // ...
};
```

The parameters of the variable template can be chosen to allow reuse of the same
variable template for multiple static data members.

For example, see `NodeStack::IdKindTable` in
[check/node_stack.h](/toolchain/check/node_stack.h).

A global constant may use a single definition without a separate declaration:

```cpp
static constexpr std::array<bool, 256> IsIdStartByteTable = [] {
  std::array<bool, 256> table = {};
  // ...
  return table;
}();
```

Note this example is using an
[immediately invoked function expression](#immediately-invoked-function-expressions-iife)
to compute the initial value, which is common.

Examples:

-   [lex/lex.cpp](/toolchain/lex/lex.cpp)
