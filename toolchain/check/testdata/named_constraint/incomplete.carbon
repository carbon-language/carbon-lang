// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/named_constraint/incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/named_constraint/incomplete.carbon

// --- fail_incomplete_constraint_in_impl_lookup.carbon
library "[[@TEST_NAME]]";

interface Z {}

// An incomplete constraint is a valid type for a symbolic facet, and is just
// not able to provide any witness for an impl lookup on that facet.
constraint N;
fn F(generic T: N) {
  // CHECK:STDERR:        ╭─┤ fail_incomplete_constraint_in_impl_lookup.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   T as Z;
  // CHECK:STDERR:        ·   ┳━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `T` that implements `N` into type implementing
  // CHECK:STDERR:        ·      ┃        `Z`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureFacetToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  T as Z;
}

// --- fail_does_not_diagnose_identify_failure_in_deduce.carbon
library "[[@TEST_NAME]]";

constraint N;

interface Z {}
// Since N is an incomplete constraint, it can not be identified as a target for
// impl lookup to determine if C implements N. No diagnostic is attached to the
// impl when the lookup fails, we simply fail to convert.
impl forall [T: N] T as Z {}

class C;

fn F() {
  // CHECK:STDERR:        ╭─┤ fail_does_not_diagnose_identify_failure_in_deduce.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   C as Z;
  // CHECK:STDERR:        ·   ━━━┳━━
  // CHECK:STDERR:        ·      ┗━━┫ error: cannot convert type `C` into type implementing `Z`
  // CHECK:STDERR:        ·         ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  C as Z;
}

// --- constraint_completed_after_deducing_impl_refers_to_it.carbon
library "[[@TEST_NAME]]";

constraint N;

interface Z {}
// N is incomplete here, but that's okay. We don't need it to be identified
// until a lookup tries to use this impl.
impl forall [T: N] T as Z {}

class C;

// N is now identified, so it can be the target of a convertion, such as for
// deducing `T` in the impl above.
constraint N {}

fn F() {
  C as Z;
}

// --- fail_impl_lookup_target_not_identified.carbon
library "[[@TEST_NAME]]";

constraint Z;

fn AsZ(unused generic T: Z) {}

fn F() {
  // Converting to facet type `Z` requires `Z` to be identified.
  // CHECK:STDERR:        ╭─┤ fail_impl_lookup_target_not_identified.carbon:[[@LINE+17]]:3
  // CHECK:STDERR:        │ constraint Z;
  // CHECK:STDERR:        · ──────┬──────
  // CHECK:STDERR:        ·       ╰──┤ constraint was forward declared here
  // CHECK:STDERR:        ·          │ [NamedConstraintForwardDeclaredHere]
  // CHECK:STDERR:        │
  // CHECK:STDERR:        │ fn AsZ(unused generic T: Z) {}
  // CHECK:STDERR:        ·                       ──┬─
  // CHECK:STDERR:        ·                         ╰──┤ initializing generic parameter `T` declared here
  // CHECK:STDERR:        ·                            │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   AsZ(());
  // CHECK:STDERR:        ·   ━━━┳━━━
  // CHECK:STDERR:        ·      ┗━━┫ error: facet type `Z` can not be identified
  // CHECK:STDERR:        ·         ┃        [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  AsZ(());
}

// --- fail_incomplete_self.carbon
library "[[@TEST_NAME]]";

// Returning a facet of type `W` requires `W` to be complete, but it's currently
// being defined where this is called. This causes an error diagnostic at the
// call.
//
// TODO: Define the function to `return U` once we can write `T: Core.Copy`.
// Right now, due to #5750, facets are not seen to implement `Core.Copy`.
eval fn ReturnFacet[T: type](generic U: T) -> T;

interface Y { let YT: type; }

interface Z {}

interface X { let XZ: Z; }
constraint W {
  require impls Z;
  // CHECK:STDERR:        ╭─┤ fail_incomplete_self.carbon:[[@LINE+29]]:31
  // CHECK:STDERR:        │ eval fn ReturnFacet[T: type](generic U: T) -> T;
  // CHECK:STDERR:        ·                                            ──┬─
  // CHECK:STDERR:        ·                                              ╰──┤ return type declared here
  // CHECK:STDERR:        ·                                                 │ [IncompleteReturnTypeHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ constraint W {
  // CHECK:STDERR:        · ───────┬──────
  // CHECK:STDERR:        ·        ╰──┤ constraint is currently being defined
  // CHECK:STDERR:        ·           │ [NamedConstraintIncompleteWithinDefinition]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   require impls X where .XZ = ReturnFacet(Self);
  // CHECK:STDERR:        ·                               ━━┳━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·                                 ┗━━┫ error: function returns incomplete type `W`
  // CHECK:STDERR:        ·                                    ┃        [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_incomplete_self.carbon:[[@LINE+12]]:31
  // CHECK:STDERR:        │ eval fn ReturnFacet[T: type](generic U: T) -> T;
  // CHECK:STDERR:        · ────────────────────────┬───────────────────────
  // CHECK:STDERR:        ·                         ╰──┤ generic function declared here
  // CHECK:STDERR:        ·                            │ [MissingGenericFunctionDefinitionHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   require impls X where .XZ = ReturnFacet(Self);
  // CHECK:STDERR:        ·                               ━━━━━┳━━━━━
  // CHECK:STDERR:        ·                                    ┗━━┫ error: use of undefined generic function
  // CHECK:STDERR:        ·                                       ┃        [MissingGenericFunctionDefinition]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  require impls X where .XZ = ReturnFacet(Self);
}
