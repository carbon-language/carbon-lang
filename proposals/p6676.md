# Carbon/C++ Interop: Importing C/C++ object-like macros

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/6676)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
    -   [Macros in C++](#macros-in-c)
        -   [Object-like macros](#object-like-macros)
        -   [Function-like macros](#function-like-macros)
        -   [Predefined macros](#predefined-macros)
    -   [Swift / C interop [GitHub][documentation]](#swift--c-interop-githubdocumentation)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Namespace](#namespace)
    -   [Constant type](#constant-type)
    -   [Constant value](#constant-value)
    -   [Constant expressions](#constant-expressions)
    -   [Empty macros](#empty-macros)
    -   [Implementation](#implementation)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
-   [Open questions](#open-questions)
    -   [Object-like macros](#object-like-macros-1)
    -   [Function-like macros](#function-like-macros-1)
    -   [Predefined macros](#predefined-macros-1)

<!-- tocstop -->

## Abstract

This proposal addresses importing object-like C/C++ macros into Carbon.

## Problem

C/C++ object-like macros are used to define constants, conditional compilation,
header guards, etc. and are common for low-level, cross-platform libraries, with
C like APIs. Despite the recommendations for replacing them in C++ with safer
techniques (for example `constexpr`), they remain to be present in C++ code
bases. Of particular interest for the interoperability are cases such as macros
present in the APIs of the standard C++ library (for example error codes in
`<errno.h>`), which can't be easily changed, but remain to be widely used in
low-level C++ libraries. A mechanism for properly importing and handling these
macros is necessary for a seamless interoperability.

## Background

### Macros in C++

Macros are defined with the `#define` directive and processed by the
preprocessor, which replaces the occurrences of the defined identifier with a
replacement-list. There are two types of macros: _object-like_ and
_function-like_ macros. The directive `#undef` undefines a defined macro.

#### Object-like macros

```
#define identifier replacement-list (optional)
```

Replacement-lists can have elements that are:

-   Literals: integer, floating-point, string, char, bool etc.
-   Non-literal types: structs, classes etc.
-   Identifiers: pointing to other macros, variables, types, keywords etc.
-   Operators: `+, -, <<, >>, |` etc. with one or more operands.

#### Function-like macros

```
#define identifier(parameters) replacement-list (optional)
#define identifier(parameters, ...) replacement-list (optional)
#define identifier(...) replacement-list (optional)
```

The syntax of function-like macros is similar to the syntax of a function call.
They accept a list of arguments and replace their occurrence in the
replacement-list.

As this is only a text replacement, it can lead to unexpected behaviour if the
arguments are not properly separated with brackets.

In function-like macros, the operators `#` and `##` enable:

-   `#operator (stringification)` - the arguments of the macro are converted to
    a string literal without expanding the argument. When `#` is used in front
    of a parameter in a macro definition (`#param`), the preprocessor replaces
    `#param` with the argument tokens as a string literal.

-   `##operator (concatenation or token pasting)` - two tokens are merged in a
    single token during a macro expansion. For example, when `a##b` is used as
    part of the macro definition, the preprocessor merges `a` and `b`, removing
    any white spaces in between to form a single token. When some of the tokens
    on either side of the `##` operator are parameter names, they are replaced
    by the actual argument before the execution of `##`. This is used for
    example to create new identifiers like variables, function names etc. and in
    general to avoid creating repeated boilerplate code.

#### Predefined macros

There are also predefined macros available in every translation unit. Examples
include: `__cplusplus`, `__FILE__`, `__LINE__`, `__DATE__`, `__TIME__` etc.

### Swift / C interop [[GitHub](https://github.com/swiftlang/swift/blob/main/lib/ClangImporter/ImportMacro.cpp)][[documentation](https://developer.apple.com/documentation/swift/using-imported-c-macros-in-swift)]

Swift supports importing object-like C macros as global constants. Macros that
use integer, floating-point and string literals are supported. Also simple
operators like `+, -, <<, >>` etc. between literals or macros are supported.

Function-like macros are not supported. Instead, using Swift functions and
generics is recommended.

## Proposal

An object-like macro that evaluates to a constant expression is imported from
C++ as a constant in Carbon. For example:

C++:

```
#define BUFFER_SIZE 4096
```

Carbon:

```
let a: i32 = Cpp.BUFFER_SIZE; // Cpp.BUFFER_SIZE is imported as an int value of type i32 with a value of 4096

```

## Details

### Namespace

The macro is evaluated in the global Cpp namespace and accessible as
Cpp.BUFFER_SIZE.

### Constant type

The type of the imported constant is deduced by Clang by evaluating the constant
expression and then mapped to a Carbon type following the existing
[Carbon <-> C++ type mapping rules](https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p5448.md).

### Constant value

The value is deduced by evaluating the tokens of a replacement list as a
constant expression.

### Constant expressions

The replacement list in the object-like macro expanding to a constant expression
can contain:

-   **Operators**: arithmetic: `+, -, *, /`; bitwise: `|, &, ^, <<, >>` ;
    logical: `||, &&`; comparison: `<, >, <=, >=, ==`; casts etc, with arbitrary
    number of operands.

For example:

```
#define ADDITION 1+2+3
```

-   **Chained macros**: macros that expand to another macro which evaluates to a
    constant.

For example:

```
#define VALUE 123
#define MY_VALUE VALUE
```

-   **Enum constants and `constexpr` variables**: if a macro's replacement list
    refers to a named constant, such as an enum constant or a `constexpr`
    variable, it is imported as an alias rather than as a literal value. This
    allows Carbon to preserve the specific type of the constant (such as `Color`
    in the example below). In the case of `constexpr` variables, importing as an
    alias also preserves addressability (that the constant is an lvalue), which
    would be lost if only the value were imported.

For example:

C++:

```
enum class Color { Red = 1, Green = 2 };
#define GREEN_COLOR Color::Green

constexpr int kValue = 123;
#define VALUE kValue
```

Carbon:

```
// Cpp.GREEN_COLOR is an alias to Cpp.Color.Green which has a type Cpp.Color.
let b: Cpp.Color = Cpp.GREEN_COLOR;

// Cpp.VALUE is an alias to kValue.
let a: i32 = Cpp.VALUE;
```

The macro will be evaluated by default in the global namespace (for example
`Cpp.VALUE`). Evaluating it in a child namespace (`Cpp.SomeNamespace.VALUE`) may
also be possible, though details about that are outside of the scope of this
proposal.

### Empty macros

The macro should have at least one value in the replacement-list so that it’s
imported. For example, the following macro won’t have a Carbon equivalent:

```
#define EMPTY
```

### Implementation

1. _Name lookup_: When a C++ macro name is encountered in Carbon it is looked-up
   before any other name. Following the C++ rules, this allows the macro to be
   found in case there is a non-macro with the same name (for example named
   variable).

2. _Macro import_: If a macro is found, it is imported as a constant to Carbon,
   by parsing the tokens of the replacement list to a constant expression and
   evaluating the result.

For example, given a macro:

```
#define MY_MACRO <some tokens>
```

The following constexpr will be (effectively) generated and imported:

```
constexpr inline decltype(auto) __carbon_import_MY_MACRO = (MY_MACRO);
```

That is, we would try to generate such a `constexpr` and import the macro if
that succeeds.

## Rationale

This work contributes to the Carbon’s goal for seamless interoperability with
C++
([Interoperability with and migration from existing C++ code](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)),
by enabling reusing and migrating existing macros, while adhering to the best
practices to use constant variables.

## Alternatives considered

-   Implementation

    a) Reuse Swift/C implementation

    -   _Reason to do this_: use existing implementation.
    -   _Reason not to do this_: this implementation is much more limited,
        longer and harder to maintain than the proposed one, as it manually
        scans the replacement tokens and tries to evaluate them, instead of
        using Clang for that. Limitations include for example type of supported
        operators, number and type of operands etc. Also, due to the licensing,
        it is not directly available.

    b) Manually scanning tokens (own implementation)

    -   _Reason not to do this_: this is a much longer implementation compared
        to the proposed one, with much more limitations compared to using Clang
        to do that.

-   Importing macros that refer to enum constants and `constexpr` variables as
    constant values, instead of aliases in Carbon
    -   _Reason to do this_: This would be a simpler implementation and
        consistent with how other object-like macros are imported.
    -   _Reason not to do this_: This would not preserve the enum's type (for
        example, `Color::Green` would be imported as an integer literal `2`
        instead of as a constant of type `Color`) or the addressability of
        `constexpr` variables (which are lvalues).

## Open questions

### Object-like macros

The support for the following cases still needs to be clarified:

-   non-constant variable name

```
int x;
#define VAL x
```

-   types:

```
#define SHORT_TYPE short
```

-   Statements or keywords:

```
#define RET return
#define FOREVER for(;;)
```

-   Expanding macros in child namespaces (`Cpp.SubNamespace.MACRO`)

### Function-like macros

The support for function-like macros will still need to be discussed in detail.
The current proposal could be extended for function-like macros in the following
direction:

Given a function-like macro:

```
#define MY_MACRO(a, b) ((a) + (b))
```

A function template whose body invokes the macro could be (eventually) generated
and imported:

```
constexpr decltype(auto) __carbon_import_MY_MACRO(auto a, auto b) {
  return (MY_MACRO(a, b));
}
```

### Predefined macros

Details on the support for predefined macros remains open as well.
