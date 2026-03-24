// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/fail_incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/fail_incomplete.carbon

// --- fail_impl_lookup_target_not_identified.carbon
library "[[@TEST_NAME]]";

constraint Z;

fn AsZ(unused T:! Z) {}

fn F() {
  // Converting to facet type `Z` requires `Z` to be identified.
  // CHECK:STDERR: fail_impl_lookup_target_not_identified.carbon:[[@LINE+10]]:3: error: facet type `Z` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   AsZ(());
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_impl_lookup_target_not_identified.carbon:[[@LINE-9]]:1: note: constraint was forward declared here [NamedConstraintForwardDeclaredHere]
  // CHECK:STDERR: constraint Z;
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_lookup_target_not_identified.carbon:[[@LINE-10]]:15: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn AsZ(unused T:! Z) {}
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  AsZ(());
}

// --- fail_incomplete_self.carbon
library "[[@TEST_NAME]]";

// Returning a facet of type `W` requires `W` to be complete, but it's currently
// being defined where this is called. This causes an error diagnostic at the
// call.
//
// TODO: Define the function to `return U` once we can write `T:! Core.Copy`.
// Right now, due to #5750, facets are not seen to implement `Core.Copy`.
eval fn ReturnFacet[T:! type](U:! T) -> T;

interface Y { let YT:! type; }

interface Z {}

interface X { let XZ:! Z; }
constraint W {
  require impls Z;
  // CHECK:STDERR: fail_incomplete_self.carbon:[[@LINE+17]]:31: error: function returns incomplete type `W` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:   require impls X where .XZ = ReturnFacet(Self);
  // CHECK:STDERR:                               ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_self.carbon:[[@LINE-5]]:1: note: constraint is currently being defined [NamedConstraintIncompleteWithinDefinition]
  // CHECK:STDERR: constraint W {
  // CHECK:STDERR: ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_self.carbon:[[@LINE-15]]:41: note: return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR: eval fn ReturnFacet[T:! type](U:! T) -> T;
  // CHECK:STDERR:                                         ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_incomplete_self.carbon:[[@LINE+7]]:31: error: use of undefined generic function [MissingGenericFunctionDefinition]
  // CHECK:STDERR:   require impls X where .XZ = ReturnFacet(Self);
  // CHECK:STDERR:                               ^~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_self.carbon:[[@LINE-22]]:1: note: generic function declared here [MissingGenericFunctionDefinitionHere]
  // CHECK:STDERR: eval fn ReturnFacet[T:! type](U:! T) -> T;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  require impls X where .XZ = ReturnFacet(Self);
}
