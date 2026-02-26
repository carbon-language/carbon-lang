// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/require_constrains_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/require_constrains_self.carbon

// --- fail_self_doesnt_meet_constraint.carbon
library "[[@TEST_NAME]]";

interface Y2 {}
interface Y(T:! Y2) {}
// TODO: The error should be that Self does not implement Y2, not that it can't
// be identified. We should allow identifying `Self`.
//
// CHECK:STDERR: fail_self_doesnt_meet_constraint.carbon:[[@LINE+17]]:37: error: facet type of value `Self` can not be identified [ImplLookupInUnidentifiedFacetTypeOfQuerySelf]
// CHECK:STDERR: constraint W { extend require impls Y(Self); }
// CHECK:STDERR:                                     ^~~~~~~
// CHECK:STDERR: fail_self_doesnt_meet_constraint.carbon:[[@LINE+14]]:1: note: constraint is currently being defined [NamedConstraintIncompleteWithinDefinition]
// CHECK:STDERR: constraint W { extend require impls Y(Self); }
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_self_doesnt_meet_constraint.carbon:[[@LINE-10]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: interface Y(T:! Y2) {}
// CHECK:STDERR:             ^
// CHECK:STDERR:
// CHECK:STDERR: fail_self_doesnt_meet_constraint.carbon:[[@LINE+7]]:37: error: cannot convert type `Self` that implements `W` into type implementing `Y2` [ConversionFailureFacetToFacet]
// CHECK:STDERR: constraint W { extend require impls Y(Self); }
// CHECK:STDERR:                                     ^~~~~~~
// CHECK:STDERR: fail_self_doesnt_meet_constraint.carbon:[[@LINE-17]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: interface Y(T:! Y2) {}
// CHECK:STDERR:             ^
// CHECK:STDERR:
constraint W { extend require impls Y(Self); }

// --- fail_todo_self_meets_constraint.carbon
library "[[@TEST_NAME]]";

interface Y2 {}
interface Y(T:! Y2) {}
constraint W {
  // TODO: This should allow the next line to work, as Self should be known to impl Y2.
  require impls Y2;
  // CHECK:STDERR: fail_todo_self_meets_constraint.carbon:[[@LINE+17]]:24: error: facet type of value `Self` can not be identified [ImplLookupInUnidentifiedFacetTypeOfQuerySelf]
  // CHECK:STDERR:   extend require impls Y(Self);
  // CHECK:STDERR:                        ^~~~~~~
  // CHECK:STDERR: fail_todo_self_meets_constraint.carbon:[[@LINE-6]]:1: note: constraint is currently being defined [NamedConstraintIncompleteWithinDefinition]
  // CHECK:STDERR: constraint W {
  // CHECK:STDERR: ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_self_meets_constraint.carbon:[[@LINE-10]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: interface Y(T:! Y2) {}
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_self_meets_constraint.carbon:[[@LINE+7]]:24: error: cannot convert type `Self` that implements `W` into type implementing `Y2` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   extend require impls Y(Self);
  // CHECK:STDERR:                        ^~~~~~~
  // CHECK:STDERR: fail_todo_self_meets_constraint.carbon:[[@LINE-17]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: interface Y(T:! Y2) {}
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  extend require impls Y(Self);
}
