# C++ Interop: Toolchain Implementation for Function Calls

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/6254)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Importing C++ functions](#importing-c-functions)
    -   [Overload resolution](#overload-resolution)
    -   [Direct calls versus thunks](#direct-calls-versus-thunks)
    -   [Thunk generation](#thunk-generation)
    -   [Parameter and return value handling](#parameter-and-return-value-handling)
    -   [Member function calls](#member-function-calls)
    -   [Operator calls](#operator-calls)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Require manual C++ wrappers](#require-manual-c-wrappers)
    -   [Mandate Carbon ABI compatibility with C++](#mandate-carbon-abi-compatibility-with-c)

<!-- tocstop -->

## Abstract

This proposal details the toolchain implementation for calling imported C++
functions from Carbon. It covers how C++ overload sets are handled, the process
of overload resolution leveraging Clang, and the generation of "thunks"
(intermediate functions) when necessary to bridge Application Binary Interface
(ABI) differences between Carbon and C++.

## Problem

Seamless, high-performance interoperability with C++
[is a fundamental goal of Carbon](https://github.com/carbon-language/carbon-lang/blob/f9bd01536b97961039257cc10fb20b495f7a9b33/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code).
The Carbon language design for C++ interoperability, particularly for function
calls, is described in the
[Carbon calling convention design](https://docs.google.com/document/d/1KUxumZtNe3mY3TsjW2s_ZADOlAaFlrtsLKHVILtqIaM).
To implement that design, the Carbon toolchain must be able to translate
Carbon-side calls to C++ functions into instructions the C++ side can
understand. Several challenges arise at the toolchain level:

-   C++ supports function overloading, requiring the toolchain to resolve calls
    to the correct C++ function within an overload set.
-   C++ types do not always have identical representations or ABIs to their
    Carbon counterparts (see
    [Carbon <-> C++ Interop: Primitive Types](https://github.com/carbon-language/carbon-lang/blob/44b2f60c90df5c1b0ce86f97bb0ece2a94eb50ea/proposals/p005448-carbon-c-interop-primitive-types.md)).
    For example, parameter passing conventions (by value, by pointer) or return
    value handling (direct return versus return slot) might differ. This may
    require the toolchain to synthesize adapter code.
-   C++ member functions require special handling of the `this` pointer.
-   C++ supports features like default arguments which need a defined mapping.

A clear, robust implementation strategy is needed to handle these complexities,
ensuring both correctness and performance.

## Background

[Carbon's C++ interoperability philosophy](https://github.com/carbon-language/carbon-lang/blob/01e12111a8a685694ccd2c9deb2779f907917543/docs/design/interoperability/philosophy_and_goals.md)
aims to minimize bridge code and provide unsurprising mappings. When Carbon code
imports a C++ header, the functions declared within become potentially callable
entities. C++ overload resolution rules are complex, and replicating them
perfectly within Carbon would be difficult and likely divergent over time.
Furthermore, direct calls are only possible when the ABI conventions of the
Carbon call site precisely match the expectations of the C++ callee.

## Proposal

1.  **Import:** C++ functions and methods, including overload sets, are imported
    into Carbon and represented internally (conceptually, as specific overload
    set instructions in SemIR).
2.  **Overload Resolution:** When a call to an imported C++ function or overload
    set occurs in Carbon, Carbon leverages Clang's overload resolution
    mechanism. Carbon argument types are mapped to hypothetical C++ types /
    expressions, and Clang's `Sema` determines the best viable function.
3.  **ABI Bridging (Thunks):**
    -   If the selected C++ function's ABI (parameter types, return type
        handling, calling convention) matches the Carbon call site's ABI based
        on defined type mappings, a direct call is generated.
    -   If the ABIs mismatch, Carbon generates an intermediate function, called
        a **C++ thunk**. This thunk has a "simple" ABI callable directly from
        Carbon (typically using only pointers and basic integer types like
        `i32`/`i64`). The thunk internally calls the actual C++ function,
        performing necessary argument conversions (for example, loading a value
        from a pointer) and handling return value conventions (for example,
        managing a return slot).
4.  **Call Execution:** The Carbon code either calls the C++ function directly
    or calls the generated C++ thunk.

## Details

### Importing C++ functions

When a C++ header is imported using `import Cpp`, declarations within that
header are made available. Function declarations, including member functions and
overloaded functions, are represented internally within Carbon's SemIR. An
overload set from C++ is represented as a single callable entity in Carbon,
associated with the set of C++ candidate functions.

### Overload resolution

To resolve a call like `Cpp.MyNamespace.MyFunc(arg1, arg2)` where `MyFunc` might
be an overload set imported from C++:

1.  **Map Arguments:** Carbon argument instructions (`arg1`, `arg2`) are mapped
    to placeholder C++ expressions (conceptually similar to
    [`clang::OpaqueValueExpr`](https://github.com/llvm/llvm-project/blob/1e99026b45b048a52f8372399ab83d488132842e/clang/include/clang/AST/Expr.h#L1178)).
    The types of these expressions are determined by mapping the Carbon argument
    types to corresponding C++ types
    ([Carbon <-> C++ Interop: Primitive Types](https://github.com/carbon-language/carbon-lang/blob/44b2f60c90df5c1b0ce86f97bb0ece2a94eb50ea/proposals/p005448-carbon-c-interop-primitive-types.md)).
2.  **Invoke Clang Sema:** Carbon invokes Clang's overload resolution logic
    ([`clang::OverloadCandidateSet::BestViableFunction()`](https://github.com/llvm/llvm-project/blob/1e99026b45b048a52f8372399ab83d488132842e/clang/include/clang/Sema/Overload.h#L1456))
    with the mapped C++ name, the candidate functions from the imported overload
    set, and the placeholder argument expressions.
3.  **Select Candidate:** Clang determines the best viable C++ function based on
    C++ rules (implicit conversions, template argument deduction if applicable
    later, etc.). If resolution fails (no viable function, ambiguity), Clang's
    diagnostics are surfaced as Carbon diagnostics.
4.  **Access Check:** After selecting a function, Carbon checks if the function
    is accessible based on C++ access specifiers (`public`, `protected`,
    `private`) in the context of the call.

### Direct calls versus thunks

A direct call from Carbon to C++ is possible only if the ABI matches exactly. A
**C++ thunk** is required if:

-   **Type Representation Mismatch:** A parameter or the return type has a
    different representation in Carbon than expected by the C++ ABI, requiring
    conversion. For example, a Carbon `bool` (`i1`) passed to a C++ `bool`
    (often `i8`), or complex struct types.
-   **Return Convention Mismatch:** The C++ function returns a non-trivial type
    by value, which typically requires a hidden return slot parameter in the
    ABI, whereas Carbon might expect a direct return value.
-   **Parameter Convention Mismatch:** C++ expects a parameter by way of
    pointer/reference where Carbon provides a value, or vice-versa.
-   **Default Arguments:** The Carbon call omits arguments that have default
    values in C++. The thunk provides the default values.
-   **Variadic arguments:** (Future work) Calling
    [C++ variadic arguments](https://en.cppreference.com/w/cpp/language/variadic_arguments.html)
    functions.

If a thunk is _not_ required, Carbon emits a direct call instruction targeting
the mangled name of the C++ function.

### Thunk generation

If a thunk is required for a C++ function `CppOriginalFunc()`, Carbon generates
a new internal function, conceptually `CppOriginalFunc__carbon_thunk()`:

1.  **Signature:** The thunk has an ABI that is simple and directly callable
    from Carbon.
    -   Parameters corresponding to C++ parameters with complex ABIs are passed
        by pointer (`T*`).
    -   Parameters with simple ABIs (like `i32`, `i64`, raw pointers) are passed
        directly.
    -   If `CppOriginalFunc` uses a return slot, the thunk takes a pointer
        parameter for the return slot. Its LLVM return type becomes `void`.
    -   If `CppOriginalFunc` returns a simple type directly, the thunk returns
        the same simple type directly.
2.  **Body:** The thunk body performs the following:
    -   Loads values from pointer arguments passed by Carbon where necessary.
    -   Performs necessary type conversions between Carbon simple ABI types and
        C++ expected types (for example, `i1` to `i8` for `bool`).
    -   Calls `CppOriginalFunc` with the converted arguments, potentially
        passing the return slot address.
    -   If `CppOriginalFunc` returned directly, the thunk returns that value. If
        it used a return slot, the thunk returns `void`.
3.  **Attributes:** The thunk is typically marked `always_inline` to encourage
    the optimizer to remove the indirection. It is given a predictable mangled
    name based on the original function's mangled name plus a suffix.

The Carbon call site then calls the thunk instead of the original C++ function.

### Parameter and return value handling

-   **Arguments:** When calling a C++ function (directly or by way of a thunk),
    Carbon arguments undergo implicit conversions as needed to match the
    parameter types determined by overload resolution. For calls requiring a
    thunk, additional conversions might occur at the call site (for example,
    taking the address of an object to pass by pointer to the thunk) and within
    the thunk (for example, loading the object from the pointer).
-   **Return Values:** If the C++ function returns `void`, the Carbon call
    expression has type `()`. If it returns a simple type directly, the Carbon
    call has the corresponding mapped Carbon type. If the C++ function uses a
    return slot, the Carbon call is modeled as initializing the storage
    designated by the return slot argument (often a temporary created at the
    call site), and the overall call expression typically results in the
    initialized value.

### Member function calls

-   **Instance Methods:** When `object.CppMethod()` is called, `object` becomes
    the implicit `this` argument. Clang's overload resolution handles the
    qualification (for example, `const`). The `this` pointer is passed as the
    first argument, either directly or to the thunk.
-   **Static Methods:** Calls like `CppClass::StaticMethod()` are treated like
    free function calls; no `this` pointer is involved.

### Operator calls

Calls to overloaded C++ operators are handled similarly to function calls.
Carbon identifies the operator call, looks up potential C++ operator functions
(both member and non-member), and uses Clang's overload resolution to select the
best candidate. Thunks may be generated if required by the selected operator
function's ABI.

## Rationale

-   **Leverages Clang:** Reusing Clang's overload resolution avoids
    reimplementing complex C++ rules and ensures consistency.
-   **Performance:** Direct calls are used when possible. Thunks are designed to
    be minimal and aggressively inlined, minimizing overhead.
-   **Correctness:** Thunks handle ABI mismatches systematically, ensuring
    correct data marshalling between Carbon and C++.
-   **Developer Experience:** Aims for C++ calls to feel natural in Carbon,
    hiding much of the complexity of ABI bridging.
-   **Interop Goal:** Directly supports the core goal of seamless C++
    interoperability.

## Alternatives considered

### Require manual C++ wrappers

Instead of generating thunks automatically, Carbon could require developers to
write C++ wrapper functions with simple C-like ABIs for any C++ function whose
ABI doesn't directly match Carbon's expectations.

-   **Rejected because:** This places a significant burden on the developer,
    increases boilerplate, hinders rapid iteration, and makes C++ libraries feel
    less integrated. It violates the goal of minimizing bridge code.

### Mandate Carbon ABI compatibility with C++

Carbon could define its types and calling conventions to always match a specific
C++ ABI (for example, Itanium).

-   **Rejected because:** This would heavily constrain Carbon's own evolution
    and design choices. It wouldn't solve the problem entirely, as C++ ABIs
    themselves vary (for example, between platforms, compilers, or even
    libraries like libc++ vs libstdc++ for `string_view`). It conflicts with the
    goal of software and language evolution.
