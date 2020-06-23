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
  - [Overloaded C++ operators](#overloaded-c-operators)
  - [Open extension points (ADL-based overload sets)](#open-extension-points-adl-based-overload-sets)
- [Open questions](#open-questions)
  - [Exporting eneric functions](#exporting-eneric-functions)

<!-- tocstop -->

## Exposing Carbon function patterns as a C++ overload set

For a collection of Carbon function patterns to be exposed to C++ code, all the
types involved must also be exposed to C++ code. These patterns will be
expressed by synthesizing an overload set in C++ code that as accurately as
possible reflects the expected pattern match that would occur with native Carbon
code. Note that there is no need to rely on the complexity of C++ conversion
sequences to precisely match any conversions triggered by the Carbon pattern
match -- instead this logic can be produced by explicitly generating all the
necessary C++ overloads and managing conversion within them.

There is a special complexity in the case of an open "overload" set. These are
functions which allow other packages and modules to add new patterns in as an
open extension point. This pattern poses two challenges. First, we will discover
more functions extending the pattern in other modules in the future. Second, C++
uses a wildly different model to accomplish similar functionality: argument
dependent lookup of unqualified names. To support exposing these kinds of open
extension points from Carbon to C++ we will need to carefully build C++ code to
emulate the Carbon facility. However, because the C++ interfaces needed for this
can be generated from the Carbon code, we can simply use a dedicated namespace
which is re-opened as needed to extend an overload set, and hide this as an
implementation detail.

### Carbon template functions and C++ overload sets

Carbon template function cannot contribute patterns to a collection exposed as a
C++ overload set. This would require translating arbitrarily complex Carbon
template code into C++ template code that could be instantiated during C++
compilation and result in identical behavior to a direct Carbon instantiation.
This seems too great of a complexity burden and constraint to realistically
support. Instead, when this kind of C++ interface needs to be exported from a
Carbon library, it must be written directly in C++ as bridge code. That bridge
code can still factor out and dispatch back into Carbon code wherever it can do
so in a non-dependent way.

## Exposing C++ overload sets as Carbon function patterns

Accessing C++ overload sets from Carbon is somewhat more complex than the
reverse because it will often be necessary to access overload sets that were not
precisely curated for this purpose. While not every pattern needs to be
supported, Carbon needs strong support for most common and idiomatic overload
sets it encounters.

The first step is how the overload set is formed. First we will consider
overload sets in the absence of argument dependent lookup. The simple name
mapping rules apply to find all the candidate overloads. Carbon will then
discard any overload involving types which cannot be _reached_ from Carbon. The
types don't need to necessarily have any sensible formulation in Carbon, but our
type mapping rules have to allow us to go from Carbon types to those C++ types
in some fashion. Next, a canonical overload must be selected among cases where
the mapping from Carbon type to C++ type could reach multiple different types. A
canonical example is mapping a non-null Carbon pointer, which could either match
a C++ pointer or a C++ reference. Here, Carbon will use heuristics to try to
find the best candidate from any such cluster -- if there is a reference
overload, Carbon will select it and try to map the type in that fashion. The
remaining overloads are then translated into Carbon function patterns.
Additional patterns and functions are synthesized to model the C++ conversion
sequences. Any ranking or other C++ overload resolution semantics are modeled by
filtering the pattern set produced, such that there in a specific Carbon
function pattern set that fully describes the possible resolved call behavior.

Note that C++ template functions are perfectly fine and can be translated into
Carbon template functions to participate in the pattern set. The reason is that
Carbon is able to instantiate any necessary C++ templates, whereas an existing
C++ compiler is not in the same position. However, the template instantiation
takes place in C++ code on C++ types. All the types selected by the pattern
match must map into C++ types for which this instantiation can succeed.

### Overloaded C++ operators

The largest body of C++ overloaded functions are operator overloads. However, in
C++ these intrinsically use argument dependent lookup and so won't work with the
above rules. Plus, Carbon uses a different set of operators. Carbon addresses
this specific use case by providing specialized operator template functions for
C++ types which are implemented as-if calling a C++ function template in bridge
code which in turn did the exact operation including ADL-based name lookup and
overload resolution. Carbon code can then override this behavior by providing
specialized patterns for operators when interacting with Carbon types. This
technique largely works because the set of possible operators is small and known
up-front.

### Open extension points (ADL-based overload sets)

The technique used for operators doesn't generalize to arbitrary ADL-based
overload sets and extension points cleanly. Currently, Carbon will only support
a specific, fixed set of such extension points because of the need to identify
their names up front; this includes operators and swap.

If this becomes an untenable restriction, we can investigate ways to more fully
surface extension points into Carbon code.

## Open questions

### Exporting eneric functions

We could likely export generic functions. More investigation should be done for
the design.
