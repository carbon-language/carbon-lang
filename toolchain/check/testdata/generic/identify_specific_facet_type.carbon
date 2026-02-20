// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/identify_specific_facet_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/identify_specific_facet_type.carbon

// --- fail_error_in_identifying_require_facet_type.carbon
library "[[@TEST_NAME]]";

// Identification tries to form the specific interface `K(array(i32, -1))` which
// is invalid, so the impl as a whole is invalid.

interface K(T:! type) {}
constraint J(N:! i32) {
  require impls K(array(i32, N));
}

// CHECK:STDERR: fail_error_in_identifying_require_facet_type.carbon:[[@LINE+7]]:1: error: facet type `J(-1)` cannot be identified in `impl as` [ImplOfUnidentifiedFacetType]
// CHECK:STDERR: impl {} as J(-1) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_error_in_identifying_require_facet_type.carbon:[[@LINE-6]]:30: note: array bound of -1 is negative [ArrayBoundNegative]
// CHECK:STDERR:   require impls K(array(i32, N));
// CHECK:STDERR:                              ^
// CHECK:STDERR:
impl {} as J(-1) {}

// --- fail_error_in_identifying_extend_require_facet_type.carbon
library "[[@TEST_NAME]]";

// Identification tries to form the specific interface `K(array(i32, -1))` which
// is invalid, so the impl as a whole is invalid.

interface K(T:! type) {}
constraint J(N:! i32) {
  extend require impls K(array(i32, N));
}

// CHECK:STDERR: fail_error_in_identifying_extend_require_facet_type.carbon:[[@LINE+7]]:1: error: facet type `J(-1)` cannot be identified in `impl as` [ImplOfUnidentifiedFacetType]
// CHECK:STDERR: impl {} as J(-1) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_error_in_identifying_extend_require_facet_type.carbon:[[@LINE-6]]:24: note: array bound of -1 is negative [ArrayBoundNegative]
// CHECK:STDERR:   extend require impls K(array(i32, N));
// CHECK:STDERR:                        ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl {} as J(-1) {}
