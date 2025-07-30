# Coalescing generic functions emitted when lowering to LLVM IR

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Problem details](#problem-details)
    -   [SemIR representation and when to do coalescing](#semir-representation-and-when-to-do-coalescing)
    -   [Recursion and strongly connected components](#recursion-and-strongly-connected-components)
    -   [Complexity](#complexity)
    -   [Canonical specific to use](#canonical-specific-to-use)
-   [Algorithm details](#algorithm-details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Coalescing in the front-end vs back-end?](#coalescing-in-the-front-end-vs-back-end)
    -   [When to do coalescing in the front-end?](#when-to-do-coalescing-in-the-front-end)
    -   [Compile-time trade-offs](#compile-time-trade-offs)
-   [Opportunities for further improvement](#opportunities-for-further-improvement)

<!-- tocstop -->

## Overview

When lowering Carbon generics to LLVM, it is possible we emit duplicate LLVM IR
functions. This document describes the algorithm implemented in
[lowering](lower.md) for determining when and which generated specifics, while
different at the Carbon language level, can be coalesced into a single one when
lowering Carbon’s intermediate representation (_SemIR_), to
[LLVM IR](https://llvm.org/docs/LangRef.html).

The overall goal of this optimization is to avoid generating duplicate LLVM IR
code where it is easy to determine this from the front-end. Such an optimization
needs to be done after specialization, but there is some flexibility in when to
do it afterwards: before lowering, through analysis of SemIR or during/after
lowering.

The goal of this doc is to describe the algorithm implemented in
[https://github.com/carbon-language/carbon-lang/pull/5314](https://github.com/carbon-language/carbon-lang/pull/5314),
from putting it into context, to the overall goal, the challenges and where
there is still room for improvement in subsequent iterations.

Determining the impact on compile-time is beyond the scope of this document, but
an important problem to follow up on.

## Problem details

In order to determine if two specific functions are equivalent, and a single one
of them can be used instead of the other, the following need to be considered as
part of the algorithm and its implementation.

### SemIR representation and when to do coalescing

In SemIR, a specific function is defined by an unique tuple: <`function_id`,
`specific_id`>. There is a single in-memory representation of a generic
function’s body (not one for each specific), where the instructions that are
different between specifics can be determined, on-demand, based on a given
specific_id. Hence determining if two specifics are equivalent needs to analyze
if these specific-dependent instructions are equivalent at the LLVM-IR level.
This can only be determined after the eval phase is complete and using
information on how Carbon types map to LLVM-IR `Type`s.

The algorithm described below does coalescing of specifics during lowering. For
alternatives considered, see [the section below](#alternatives-considered).

### Recursion and strongly connected components

Comparing if two different specific functions contain (access, invoke, etc) the
same specific-dependent instruction is not straight forward when recursion is
involved. The simplest example is when A and B each are recursive functions, and
are equivalent. The check “are A and B equivalent” needs to use that as the
starting assumption, and when a call to A and B respectively are found in each,
then the conclusion is that they are equivalent. In practice this requires
comparison of `specific_id`s, which in SemIR are distinct.

In the general case, this analysis needs to analyze the call-graph for all
functions and build strongly connected components (SCCs). Either the call graph
is created before lowering, or it is built while lowering, and in a
post-processing phase we can conclude equivalence and simplify the (read: delete
unnecessary) emitted LLVM IR. The current implementation does the latter.

A non-viable option is building the call graph based on the information “what
are all call sites of myself, where I am a specific function”, because this
information is not available until processing the function bodies of all
specific functions. This is an optimization done so that the definition of a
specific isn’t emitted until a use of it is found. Building that information
would duplicate all the lowering logic, minus the LLVM IR creation.

### Complexity

Even with limiting the comparison of specific functions to those defined from
the same generic, a comparison algorithm would still end up with quadratic
complexity in the number of specifics for that generic.

We define two fingerprints for each specific: the first fingerprint
`common_fingerprint` includes specific-dependent information that does not
include `specific_id` information (this would only be discoverable as equivalent
as part of an equivalence SCC), while the second fingerprint,
`specific_fingerprint`, includes all specific-dependent information. As such,
two specific functions are not equivalent if their `common_fingerprint` differs.
Two specific functions are equivalent if their `specific_fingerprint`s are
equal. If the `common_fingerprint`s are equal but the `specific_fingerprint`s is
not, the two functions may still be equivalent.

Ideally, the `specific_fingerprint` can be used as a unique hash and used to
first coalesce all specific functions with this same fingerprint, with no
additional checks. Then, all remaining functions may use the
`common_fingerprint` as a another unique hash to group remaining potential
candidates for coalescing. Then, only those with this same second hash are
processed in a quadratic pass walking all calls instructions and comparing if
the `specific_id` information is equivalent. This optimization is not currently
implemented.

Note that, if we were to extend the algorithm to do compile-time ICF (Identical
Code Folding), that is define a fingerprint for all functions (not only
specifics of the same generic), based on all their instructions (not only their
specific-dependent ones), the complexity would be much higher, and so would the
compile-time cost.

### Canonical specific to use

For determining the canonical specific to use, we use a
[disjoint set](https://en.wikipedia.org/wiki/Disjoint-set_data_structure).

## Algorithm details

Below is a pseudocode of the existing algorithm in
`toolchain/lower/specific_coalescer.*`.

The implementation has been merged in
[https://github.com/carbon-language/carbon-lang/pull/5314](https://github.com/carbon-language/carbon-lang/pull/5314)

At the top level, the current algorithm first generates all function
definitions, and once this is complete, it performs the logic to coalesce
specifics and delete the redundant LLVM function definitions.

```none
LowerToLLVM () {
  for all non_generic_functions
    CreateLLVMFunctionDefinition (function, no_specific_id);
  PerformCoalescingPostProcessing ();
}
```

The lowering starts with all non-generic functions. While lowering these, when
calls to specifics are encountered, it also generates definitions for those
specific functions.

For each lowered specific function definition, we create the
`SpecificFunctionFingerprint`, which includes the two fingerprints (hashes)
defined [above](#complexity), and a list of calls to other specific functions.

```none
CreateLLVMFunctionDefinition (function, specific_id) {
   Step1: Build LLVM::Function*: emit LLVM IR for each SemIR instruction.

   Step2: If the instruction is specific-dependent, hash it and add to its `common_fingerprint`

   Step3: When finding a call to a generic, with a defined type (that is a call to a specific),
    a) create a definition for this specific_id if it doesn't exist:
      CreateLLVMFunctionDefinition (function, specific_id);
    b) hash the specific_id to the current function's `specific_fingerprint`
    c) add the non-hashed specific_id to list of calls performed
}
```

The logic that performs the actual coalescing, first checks if the LLVM function
types match (using a third hash-like fingerprint: `function_type_fingerprint`
for storage optimization), then if these are equivalent based on the
`SpecificFunctionFingerprint`. For each pair of equivalent functions found (in
an SCC), the uses of a definition will be replaced with the canonical one, and
that definition will be deleted.

```none
PerformCoalescingPostProcessing () {
  for each two specifics of the same generic {
    if function_type_fingerprints differ {
      track as non-equivalent
      continue
    }

    add the two specifics to assumed equivalent specifics list
    if (CheckIfEquivalent(two specifics, assumed equivalent specifics list)) {
      for each two equivalent specifics found {
        find the canonical specific & mark the duplicates for replacement/deletion
    }
  }
  replace all duplicate specifics with the respective canonical specifics
  and delete all replaced LLVM function definitions.
}

```

The equivalence check for specifics based on the constructed
`SpecificFunctionFingerprint` can make an early non-equivalence determination
based on the `common_fingerprint`s, and an early equivalence determination based
on the `specific_fingerprint`s. Otherwise, it uses the call list and recurses to
make the determination for all functions in the SCC call graph (in practice the
implementation uses a worklist to avoid the recursion).

```none
CheckIfEquivalent(two specifics, mutable list of assumed equivalent specifics) -> bool {
  if common_fingerprints are non-equal or already tracked as non-equivalent {
    track as non-equivalent
    return false
  }
  if specific_fingerprints are equal {
    track as equivalent
    return true
  }
  if already tracked as equivalent or assumed equivalent {
    return true
  }

  for each of the calls in each of the specifics {
    if the functions called are the same or already equivalent or assumed equivalent {
      continue
    }
    if the functions called are already non-equivalent {
      return false
    }
    add <pair of calls> to assumed equivalent specifics
    if !CheckIfEquivalent(specifics in <pair of calls>, assumed equivalent specifics) {
      return false;
    }
  }
}
```

## Rationale

The Carbon language cares about low compile times in general. This optimization
focuses on that.

## Alternatives considered

### Coalescing in the front-end vs back-end?

An alternative considered was not doing any coalescing in the front-end and
relying on LLVM to make the analysis and optimization. The current choice was
made based on the expectation that such an
[LLVM pass](https://llvm.org/docs/MergeFunctions.html) would be more costly in
terms of compile-time. The relative cost has not yet been evaluated.

### When to do coalescing in the front-end?

The analysis and coalescing could be done prior to lowering, after
specialization. The advantage of that choice would be avoiding to lower
duplicate LLVM functions and then removing the duplicates. The disadvantage of
that choice would be duplicating much of the lowering logic, currently necessary
to make the equivalence determination.

### Compile-time trade-offs

Not doing any coalescing is also expected to increase compile time more than
performing the analysis and deduplication. This can be evaluated in practice and
the feature disabled if found to be too costly.

## Opportunities for further improvement

The current implemented algorithm can be improved with at least the following:

-   The `specific_fingerprint` can be used to already bucket specifics that can
    be coalesced right away.
-   The remaining ones can be pre-bucketed such that only the specifics with the
    same `common_fingerprint` have their list of calls further compared (linear
    in the number of specific calls inside the functions) to determine SCCs that
    may be equivalent.

This should reduce the complexity from the current N^2, with N=number of
specifics for a generic, to M^2, with M being the number of specifics for a
generic that have different `specific_fingerprint` and equal
`common_fingerprint` (expectation is that M << N).

An additional potential improvement is defining the function fingerprints in a
manner that is translation-unit independent, so this can be used in the mangled
name, and the same function name emitted. This does not currently occur, as the
two fingerprints use internal SemIR identifiers (`function_id` and `specific_id`
respectively).
