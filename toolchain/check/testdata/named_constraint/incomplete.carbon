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
library "[[@TEST_NAME]]";

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

// --- fail_todo_does_not_diagnose_identify_failure_in_deduce.carbon
library "[[@TEST_NAME]]";

constraint N;

interface Z {}
// Since N is an incomplete constraint, it can not be identified for impl lookup
// to determine if C implements N.
impl forall [T:! N] T as Z {}

class C;

fn F() {
  // TODO: The "can not be identified" diagnostic should not occur here. That
  // diagnostic happened during deduction, which means to suppress diagnostics.
  // Only the convert error should be emitted.
  //
  // CHECK:STDERR: fail_todo_does_not_diagnose_identify_failure_in_deduce.carbon:[[@LINE+11]]:3: error: facet type `N` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   C as Z;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_todo_does_not_diagnose_identify_failure_in_deduce.carbon:[[@LINE-17]]:1: note: constraint was forward declared here [NamedConstraintForwardDeclaredHere]
  // CHECK:STDERR: constraint N;
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_does_not_diagnose_identify_failure_in_deduce.carbon:[[@LINE+4]]:3: error: cannot convert type `C` into type implementing `Z` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   C as Z;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  C as Z;
}
