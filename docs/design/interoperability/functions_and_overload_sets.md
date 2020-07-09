# Functions and overload sets

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [Exposing Carbon function patterns as a C++ overload set](#exposing-carbon-function-patterns-as-a-c-overload-set)
  - [Carbon template functions and C++ overload sets](#carbon-template-functions-and-c-overload-sets)
- [Exposing C++ overload sets as Carbon function patterns](#exposing-c-overload-sets-as-carbon-function-patterns)
  - [C++ template functions and Carbon function patterns](#c-template-functions-and-carbon-function-patterns)
  - [Overloaded C++ operators](#overloaded-c-operators)
  - [Open extension points (ADL-based overload sets)](#open-extension-points-adl-based-overload-sets)
- [Open questions](#open-questions)
  - [Overrides for specifying an overload](#overrides-for-specifying-an-overload)
  - [Exporting generic functions](#exporting-generic-functions)

<!-- tocstop -->

## Exposing Carbon function patterns as a C++ overload set

For a collection of Carbon function patterns to be exposed to C++ code, all the
types involved must also be exposed to C++ code. These patterns will be
expressed by synthesizing an overload set in C++ code that as accurately as
possible reflects the expected pattern match that would occur with native Carbon
code.

There is no need to rely on the complexity of C++ conversion sequences to
precisely match any conversions triggered by the Carbon pattern match. Instead,
this logic can be produced by explicitly generating all the necessary C++
overloads and managing conversion within them.

### Carbon template functions and C++ overload sets

Carbon template functions cannot contribute patterns to a collection exposed as
a C++ overload set. This would require translating arbitrarily complex Carbon
template code into C++ template code that could be instantiated during C++
compilation in a way that would result in identical behavior to a direct Carbon
instantiation. This seems too great of a complexity burden and constraint to
realistically support.

Bridge code will be required when C++ code needs to call into Carbon template
functions. That bridge code can still factor out and dispatch back into Carbon
code wherever it can do so in a non-dependent way.

## Exposing C++ overload sets as Carbon function patterns

Accessing C++ overload sets from Carbon is somewhat more complex than the
reverse because it will often be necessary to access overload sets that were not
precisely curated for this purpose. While not every pattern needs to be
supported, Carbon needs strong support for most common and idiomatic overload
sets it encounters.

Overload sets will be translated in a series of steps:

1. Carbon will consider how the overload set is formed in the absence of ADL
   (argument dependent lookup). The simple name mapping rules apply to find all
   the candidate overloads.

2. Carbon will then discard any overload involving types which cannot be
   _reached_ from Carbon. The types don't need to necessarily have any sensible
   formulation in Carbon, but our type mapping rules have to allow us to go from
   Carbon types to those C++ types in some fashion.

3. A canonical overload must be selected among cases where the mapping from
   Carbon type to C++ type could reach multiple different types. A representative
   example is mapping a non-null Carbon pointer, which could either match a C++
   pointer or a C++ reference. Here, Carbon will use heuristics to try to find
   the best candidate from any such cluster—if there is a reference overload,
   Carbon will select it and try to map the type in that fashion.

4. The remaining overloads are then translated into Carbon function patterns.
   Additional patterns and functions are synthesized to model the C++ conversion
   sequences. Any ranking or other C++ overload resolution semantics are modeled
   by filtering the pattern set produced, such that there in a specific Carbon
   function pattern set that fully describes the possible resolved call
   behavior.

### C++ template functions and Carbon function patterns

C++ template functions can be translated into Carbon template functions to
participate in the pattern set. This works because the Carbon compiler will
instantiate any necessary C++ templates, contrasting with our inability to have
C++ compilers instantiate Carbon templates. The template instantiation takes
place in C++ code on C++ types. All the types selected by the pattern match must
map into C++ types for which this instantiation can succeed.

### Overloaded C++ operators

Operators and certain other calls, such as `std::swap`, represent a small, fixed
set of possible extension points. Conversely, these also represent the largest
body of overloaded C++ functions. These intrinsically use ADL, and so won't work
with the above rules. However, we can work around this limitation in specific
cases.

Carbon will provide specialized operator template functions for C++ types which
are implemented as-if calling a C++ function template in bridge code which in
turn did the exact operator call, including ADL-based name lookup and overload
resolution. Carbon code can then override this behavior by providing specialized
patterns for operators when interacting with Carbon types.

### Open extension points (ADL-based overload sets)

The technique used for operators doesn't generalize to arbitrary ADL-based
overload sets and extension points cleanly. If this becomes an untenable
restriction, we can investigate ways to more fully surface extension points into
Carbon code.

## Open questions

### Overrides for specifying an overload

Depending on how well the heuristic for exposing C++ overload sets works, we may
want to support overrides to specify which call to use. Alternately, bridge code
could also be required.

Whether an override is needed may only become clear once there's an
implementation to test and determine the frequency of issues.

### Exporting generic functions

We could likely export generic functions. More investigation should be done for
the design.
