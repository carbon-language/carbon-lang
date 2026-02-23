// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/named_constraint/incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/named_constraint/incomplete.carbon

// --- fail_incomplete_constraint_in_impl_lookup.carbon

interface Z {}

constraint N;
fn F(T:! N) {
  // TODO: The failed-to-identify error should be a note under the convert error
  // instead of being a separate error. This can be done by making a top-level
  // convert error as a Context for specific errors that happen during convert.
  //
  // CHECK:STDERR: fail_incomplete_constraint_in_impl_lookup.carbon:[[@LINE+11]]:3: error: facet type of value `T` can not be identified [ImplLookupInUnidentifiedFacetTypeOfQuerySelf]
  // CHECK:STDERR:   T as Z;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_incomplete_constraint_in_impl_lookup.carbon:[[@LINE-9]]:1: note: constraint was forward declared here [NamedConstraintForwardDeclaredHere]
  // CHECK:STDERR: constraint N;
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_incomplete_constraint_in_impl_lookup.carbon:[[@LINE+4]]:3: error: cannot convert type `T` that implements `N` into type implementing `Z` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   T as Z;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  T as Z;
}
