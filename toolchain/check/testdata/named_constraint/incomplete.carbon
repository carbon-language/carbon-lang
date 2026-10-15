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
  // CHECK:STDERR: fail_incomplete_constraint_in_impl_lookup.carbon:[[@LINE+2]]:3-8: error: cannot convert type `T` that implements `N` into type implementing `Z` [ConversionFailureFacetToFacet]
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
  // CHECK:STDERR: fail_does_not_diagnose_identify_failure_in_deduce.carbon:[[@LINE+2]]:3-8: error: cannot convert type `C` into type implementing `Z` [ConversionFailureTypeToFacet]
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
  // CHECK:STDERR: fail_impl_lookup_target_not_identified.carbon:[[@LINE+4]]:3-9: error: facet type `Z` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR: fail_impl_lookup_target_not_identified.carbon:[[@LINE-7]]:1-13: note: constraint was forward declared here [NamedConstraintForwardDeclaredHere]
  // CHECK:STDERR: fail_impl_lookup_target_not_identified.carbon:[[@LINE-6]]:23-26: note: initializing generic parameter `T` declared here [InitializingGenericParam]
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
  // CHECK:STDERR: fail_incomplete_self.carbon:[[@LINE+7]]:31-47: error: function returns incomplete type `W` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR: fail_incomplete_self.carbon:[[@LINE-3]]:1-14: note: constraint is currently being defined [NamedConstraintIncompleteWithinDefinition]
  // CHECK:STDERR: fail_incomplete_self.carbon:[[@LINE-11]]:44-47: note: return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_incomplete_self.carbon:[[@LINE+3]]:31-41: error: use of undefined generic function [MissingGenericFunctionDefinition]
  // CHECK:STDERR: fail_incomplete_self.carbon:[[@LINE-14]]:1-48: note: generic function declared here [MissingGenericFunctionDefinitionHere]
  // CHECK:STDERR:
  require impls X where .XZ = ReturnFacet(Self);
}
