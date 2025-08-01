# Coalescing generic functions emitted when lowering to LLVM IR

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Design details](#design-details)
    -   [SemIR representation and why to coalesce during lowering](#semir-representation-and-why-to-coalesce-during-lowering)
    -   [Recursion and strongly connected components (SCCs)](#recursion-and-strongly-connected-components-sccs)
    -   [Function fingerprints](#function-fingerprints)
    -   [Canonical specific to use](#canonical-specific-to-use)
-   [Algorithm details](#algorithm-details)
-   [Alternatives considered](#alternatives-considered)
    -   [Coalescing in the front-end vs back-end?](#coalescing-in-the-front-end-vs-back-end)
    -   [When to do coalescing in the front-end?](#when-to-do-coalescing-in-the-front-end)
    -   [Compile-time trade-offs](#compile-time-trade-offs)
    -   [Coalescing duplicate non-specific functions](#coalescing-duplicate-non-specific-functions)
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
[specifics_coalescer](/toolchain/lower/specific_coalescer.h), from putting it
into context, to the overall goal, the challenges and where there is still room
for improvement in subsequent iterations.

Determining the impact on compile-time is beyond the scope of this document, but
an important problem to follow up on.

## Design details

In order to determine if two specific functions are equivalent, and a single one
of them can be used instead of the other, the following need to be considered as
part of the algorithm and its implementation.

### SemIR representation and why to coalesce during lowering

In SemIR, a specific function is defined by an unique tuple:
`(function_id, specific_id)`. There is a single in-memory representation of a
generic function’s body (not one for each specific), where the instructions that
are different between specifics can be determined, on-demand, based on a given
`specific_id`. Hence, determining if two specifics are equivalent needs to
analyze if these specific-dependent instructions are equivalent at the LLVM IR
level. This can only be determined after the eval phase is complete and using
information on how Carbon types map to `llvm::Type`s.

The algorithm described below does coalescing of specifics during lowering. Also
see [alternatives considered](#alternatives-considered).

### Recursion and strongly connected components (SCCs)

Comparing if two different specific functions contain (access, invoke, etc.) the
same specific-dependent instruction is not straightforward when recursion is
involved. The simplest example is when A and B each are recursive functions, and
are equivalent. The check "are A and B equivalent" needs to start by assuming
they are equivalent, and when a self-recursive call is found in each, that call
is still equivalent. In practice this requires comparison of `specific_id`s,
which in SemIR are distinct.

In the general case, this analysis needs to analyze the call graph for all
functions and build strongly connected components (SCCs). The call graph could
either be created before lowering or built while lowering. The current
implementation does the latter, and in a post-processing phase we can conclude
equivalence and simplify the emitted LLVM IR by deleting unnecessary parts.

A non-viable option is building the call graph based on the information "what
are all call sites of myself, where I am a specific function", because this
information is not available until processing the function bodies of all
specific functions. This is an optimization done so that the definition of a
specific isn’t emitted until a use of it is found. Building that information
would duplicate all the lowering logic, minus the LLVM IR creation.

### Function fingerprints

Even with limiting the comparison of specific functions to those defined from
the same generic, a comparison algorithm would still end up with quadratic
complexity in the number of specifics for that generic.

We define two fingerprints for each specific:

1. `specific_fingerprint`: Includes all specific-dependent information.
2. `common_fingerprint`: Includes the same except for `specific_id` information,
   as `specific_id`s can only be determined to be equivalent after building an
   equivalence SCC.

Two specific functions are equivalent if their `specific_fingerprint`s are equal
and are not equivalent if their `common_fingerprint`s differs. If the
`common_fingerprint`s are equal but the `specific_fingerprint`s are not, the two
functions may still be equivalent.

Ideally, the `specific_fingerprint` can be used as a unique hash to first
coalesce all specific functions with this same fingerprint, with no additional
checks. Then, all remaining functions may use the `common_fingerprint` as
another unique hash to group remaining potential candidates for coalescing.
Then, only those with this same `common_fingerprint` are processed in a
quadratic pass walking all calls instructions and comparing if the `specific_id`
information is equivalent. These optimizations are not currently implemented.

Note that this does not
[coalesce non-specifics](#coalescing-duplicate-non-specific-functions).

### Canonical specific to use

For determining the canonical specific to use, we use a
[disjoint set](https://en.wikipedia.org/wiki/Disjoint-set_data_structure).

## Algorithm details

Below is a pseudocode of the existing algorithm in
`toolchain/lower/specific_coalescer.*`.

The implementation can be found in
[specifics_coalescer.h](/toolchain/lower/specific_coalescer.h) and
[specifics_coalescer.cpp](/toolchain/lower/specific_coalescer.cpp).

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
`SpecificFunctionFingerprint`, which includes the
[two fingerprints](#function-fingerprints), and a list of calls to other
specific functions.

```none
CreateLLVMFunctionDefinition (function, specific_id) {
  For each SemIR instruction in the function:
    Step 1: Emit LLVM IR for the instruction
    Step 2: If the instruction is specific-dependent, hash it and add to its `common_fingerprint`
    Step 3: If the SemIR instruction is a call to a specific,
      a) Create a definition for this specific_id if it doesn't exist:
        CreateLLVMFunctionDefinition (function, specific_id);
      b) Hash the specific_id to the current function's `specific_fingerprint`
      c) Add the non-hashed specific_id to list of calls performed
}
```

The logic that performs the actual coalescing analyzes all specifics. For each
pair of two specifics, it first checks if the LLVM function types match (using a
third hash-like fingerprint: `function_type_fingerprint` for storage
optimization), then if these are equivalent based on the
`SpecificFunctionFingerprint`. For each pair of equivalent functions found (in a
callgraph SCC), one function will be marked non-canonical: its uses are replaced
with the canonical one and its definition will ultimately be deleted.

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
CheckIfEquivalent(two specifics, &assumed equivalent specifics) -> bool {
  if common_fingerprints are non-equal {
    track as non-equivalent specifics
    return false
  }
  if specific_fingerprints are equal {
    track as equivalent specifics
    return true
  }
  if already tracked as equivalent or assumed equivalent specifics {
    return true
  }

  for each of the calls in each of the specifics {
    if the functions called are the same or already equivalent or assumed equivalent specifics {
      continue
    }
    if the functions called are already non-equivalent specifics {
      return false
    }
    add <pair of calls> to assumed equivalent specifics
    if !CheckIfEquivalent(specifics in <pair of calls>, assumed equivalent specifics) {
      return false;
    }
  }
}
```

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

Not doing any coalescing is also expected to increase the back-end codegen time
more than performing the analysis and deduplication. This can be evaluated in
practice and the feature disabled if found to be too costly.

### Coalescing duplicate non-specific functions

We could coalesce duplicate functions in non-specific cases, similar to lld's
[Identical Code Folding](https://lld.llvm.org/NewLLD.html#glossary) or LLVM's
[MergeFunctions pass](https://llvm.org/docs/MergeFunctions.html). This would
require fingerprinting all instructions in all functions, whereas specific
coalescing can focus on cases that only Carbon's front-end knows about. Carbon
would also be restricted to coalescing functions in a single compilation unit,
which would require replacing function definitions that allow external calls
with a placeholder that calls the coalesced definition. We don't expect
sufficient advantages over existing support.

## Opportunities for further improvement

The current implemented algorithm can be improved with at least the following:

-   The `specific_fingerprint` can be used to already bucket specifics that can
    be coalesced right away.
-   The remaining ones can be pre-bucketed such that only the specifics with the
    same `common_fingerprint` have their list of calls further compared (linear
    in the number of specific calls inside the functions) to determine SCCs that
    may be equivalent.

This should reduce the complexity from the current O(N^2), with N=number of
specifics for a generic, to O(M^2), with M being the number of specifics for a
generic that have different `specific_fingerprint` and equal
`common_fingerprint` (expectation is that M << N).

An additional potential improvement is defining the function fingerprints in a
manner that is translation-unit independent, so this can be used in the mangled
name, and the same function name emitted. This does not currently occur, as the
two fingerprints use internal SemIR identifiers (`function_id` and `specific_id`
respectively).
