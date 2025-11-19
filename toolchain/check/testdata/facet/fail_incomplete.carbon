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

// --- fail_impl_lookup_incomplete.carbon
library "[[@TEST_NAME]]";

constraint Z;

fn AsZ(T:! Z) {}

fn F() {
  // Requires Z identified.
  // CHECK:STDERR: fail_impl_lookup_incomplete.carbon:[[@LINE+10]]:3: error: facet type `Z` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   AsZ(());
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_impl_lookup_incomplete.carbon:[[@LINE-9]]:1: note: constraint was forward declared here [NamedConstraintForwardDeclaredHere]
  // CHECK:STDERR: constraint Z;
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_lookup_incomplete.carbon:[[@LINE-10]]:8: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn AsZ(T:! Z) {}
  // CHECK:STDERR:        ^
  // CHECK:STDERR:
  AsZ(());
}
