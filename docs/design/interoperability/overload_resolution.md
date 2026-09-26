# C++ overload resolution

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Background](#background)
    -   [Overload sets](#overload-sets)
    -   [Implicit conversion sequences in C++ overload resolution](#implicit-conversion-sequences-in-c-overload-resolution)
-   [Detailed design](#detailed-design)
    -   [Example](#example)
    -   [Viability versus validity of making a call](#viability-versus-validity-of-making-a-call)
    -   [Non-overloaded functions](#non-overloaded-functions)
-   [Future work](#future-work)
    -   [Implementation of the Call interface](#implementation-of-the-call-interface)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

When calling a C++ overload set from Carbon, the rules that decide exactly which
overload to call and how to initialize the parameters are shared between Carbon
and C++, and the work to implement those rules are shared between the Carbon
toolchain and Clang. The C++ side determines _what_ is called, and the Carbon
side determines _how_ it is called.

C++ has complex rules for overload resolution, involving function templates,
partial ordering, and implicit conversion sequences. Carbon aims to interoperate
seamlessly with C++, which includes the ability to call overloaded C++
functions. However, Carbon's own rules for overload resolution and implicit
conversions differ from C++'s. We need to determine how to resolve calls to C++
overload sets from Carbon in a way that is predictable, preserves the semantics
of the C++ API, and respects Carbon's safety guarantees (such as avoiding lossy
implicit conversions).

## Background

### Overload sets

The basic unit of callable API in C++ is an _overload set_. This is a collection
of functions and function templates that were found by name lookup as potential
candidates for a call. An overload set should be treated as a unit and not
divided into individual declarations, because overloads may misbehave if called
with arguments that would have been a better match for another overload.

In modern C++, even a single non-templated function is considered to be an
overload set, and uses the same rules as an overload set that can produce
multiple different candidates.

### Implicit conversion sequences in C++ overload resolution

When overload resolution in C++ considers call candidates, it builds _implicit
conversion sequences_ to determine whether each argument can be converted to the
corresponding parameter, and approximately how that would be done. The implicit
conversion sequences for each argument are ranked and compared across overload
candidates to determine which overload should be called – the selected overload
must have the best (or tied for best) implicit conversion sequence for every
argument.

Implicit conversion sequences are a virtual representation of what an implicit
conversion might do, and the rules for forming them mostly mirror the C++
implicit conversion rules. However, they are not identical to the implicit
conversion rules:

-   There are cases where an implicit conversion sequence exists but no implicit
    conversion exists, where the C++ rules believe they have identified the
    function that the developer "meant to" call, even though it is not actually
    callable. For example:

    -   Given an argument that is a glvalue of type T, an implicit conversion
        sequence to a parameter of type T always exists and always has
        "identity" rank, even if T is not copyable or moveable.
    -   Implicit conversion sequence formation typically does not take access
        into account. An implicit conversion sequence can be formed when the
        argument is of type `Derived*` and the parameter is of type
        `PrivateBase*`.

    If an overload candidate is selected for which implicit conversion sequences
    can be formed but implicit conversions cannot, the later process of building
    a call to the selected candidate will fail with an error.

-   There are rare cases where an implicit conversion sequence does not exist
    despite an implicit conversion being possible. Initially this was assumed to
    not happen, and the C++ rules historically did not perform overload
    resolution when the only candidate was a single non-templated function, but
    [the rules were changed](https://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#2241)
    to perform overload resolution even for that case so that, among other
    concerns, calls with no viable implicit conversion sequence would be
    rejected even if an implicit conversion is possible.

## Detailed design

When an overloaded C++ function is called from Carbon, the overload is resolved
by Clang, using the C++ rules. This includes:

-   Performing template argument deduction and template instantiation.
-   Forming implicit conversion sequences.
-   Ranking the overload candidates and picking the best viable function.

Then, the selected function is converted into a Carbon function signature, and
called using the Carbon rules for performing a function call, including argument
conversions.

### Example

```carbon
import Cpp inline '''
#include "stdint.h"
void F(...);
template<typename T> void F(T);
template<typename T> void F(T*);
void F(int32_t);
''';

fn CallF() {
  var n: i64;

  // C++ overload resolution rules pick F<int64_t>(int64_t*).
  // This is converted to a synthesized Carbon function:
  //   fn F(p: i64*) -> () {
  //     // call C++ function template specialization
  //   }
  // This is then called using the Carbon rules for function calls.
  F(&n);

  // C++ overload resolution rules pick F(int32_t).
  // This is converted to a synthesized Carbon function:
  //   fn F(n: i32) -> () {
  //     // call C++ function
  //   }
  // The call is then rejected in Carbon because i64 can't be converted to i32.
  F(n);
}
```

### Viability versus validity of making a call

These rules mean that a distinction is made between determining whether a
candidate from C++ is _viable_ – determined using the C++ rules for implicit
conversion sequences – and whether that candidate is actually _callable_ –
determined using the Carbon rules for implicit conversion. This is the same
process that happens in C++ already, except that the callability check is
performed using the Carbon rules for implicit conversion instead of the C++
rules.

For this to work well, implicit conversions in Carbon need to line up reasonably
well with implicit conversion sequences in C++. There are two ways a divergence
can occur:

-   **C++ implicit conversion sequence can be formed but Carbon implicit
    conversion cannot be performed.** For example, for a parameter of type
    `int32_t`, C++ logic may accept an argument of type `int64_t` but Carbon
    implicit conversion rules would not permit the conversion. In this case, the
    C++ candidate is selected, and the call is rejected in Carbon. This is
    analogous to cases in C++ where overload resolution selects a function
    believed to be the intended callee but rejects the call because the
    conversion is not actually possible.
-   **C++ implicit conversion sequence cannot be formed but a Carbon implicit
    conversion can be.** This should be rare, as Carbon is generally more
    restrictive about which implicit conversions it permits than C++ is.
    However, if implicit conversions are defined in Carbon code but cannot be
    represented in C++, the C++ search may not find them. In this situation, the
    call will either be rejected because there are no viable candidates, or a
    different candidate will be selected. Conversions that cannot be lifted from
    Carbon to C++ should be rare, minimizing this concern.

### Non-overloaded functions

When a call is made from Carbon to a C++ function that is _not_ overloaded,
these rules reduce to two steps:

1.  Ensure that the C++ rules believe that the function is viable for the call.
2.  Call the function using the Carbon rules, as if it were a normal Carbon
    function.

Although it is tempting to remove the first step to align the behavior with a
Carbon -> Carbon call, doing so would make the interop behavior less consistent.
The extra check does not harm the migration story, as migrating the callee from
C++ to Carbon removes the check without affecting previously valid callers.

## Future work

### Implementation of the Call interface

A C++ overload set should eventually be modeled as providing a templated `impl`
of the `Call` interface, with a
[predicate](https://github.com/carbon-language/carbon-lang/issues/2153)
constraint that checks whether the function is callable from C++. In approximate
Carbon code, this might look like the following:

```carbon
// Built-in mechanism to call into C++ overload resolution, provided by C++
// interop.
fn! SelectCallee(F:! Cpp.OverloadSet, ...template each T:! type)
    -> Optional(Cpp.Candidate);
predicate IsCallable(F:! Cpp.OverloadSet, ...template each T:! type)
    = SelectCallee(F, ...each T).HasValue();

impl forall [...template each T:! type] Cpp.F as Call(... each T)
    if IsCallable(Cpp.F, ... each T) {
  fn Call(...each t: each T) -> auto {
    SelectCallee(Cpp.F, ...each T)(...each t);
  }
}
```

Any other information made available to the `Call` interface, such as the
expression category and constant value of the arguments, and whether the
arguments are tuple / struct literals, should also be made available to C++
overload resolution.

## Alternatives considered

-   [Use the Carbon rules alone](/proposals/p6825.md#use-the-carbon-rules-alone)
    -   For template argument deduction
    -   For viability of overload candidates
    -   For ranking overload candidates
-   [Use the C++ rules alone](/proposals/p6825.md#use-the-c-rules-alone)

## References

-   [Proposal #6825: C++ interop for overloaded functions and function templates](https://github.com/carbon-language/carbon-lang/pull/6825)
-   [Carbon: Interop - Using C++](https://docs.google.com/document/d/1mJk92JUPzPNr4LSDUUvfhL9-9i0Dh707uKCCuw3LncU/edit?tab=t.0#heading=h.6dyyhz5krl9v)
-   [Carbon: C++ Interop for constructors](https://docs.google.com/document/d/1_bD_WWkWaikPCxpZnzp8PAf-BFWkcTH-GJpw1GZt17E/edit?tab=t.0)
