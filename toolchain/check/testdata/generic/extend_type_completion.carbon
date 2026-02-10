// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/extend_type_completion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/extend_type_completion.carbon

// --- class_impl_doesnt_need_complete_interface.carbon
library "[[@TEST_NAME]]";

interface K(T:! type) {}

class C(N:! i32) {
  impl as K(array(i32, N)) {}
}

// C does not extend K so the type of K is not completed. No error.
var v: C(-1);

// --- fail_class_extend_impl_does_need_complete_interface.carbon
library "[[@TEST_NAME]]";

interface K(T:! type) {}

class C(N:! i32) {
  // CHECK:STDERR: fail_class_extend_impl_does_need_complete_interface.carbon:[[@LINE+3]]:18: error: array bound of -1 is negative [ArrayBoundNegative]
  // CHECK:STDERR:   extend impl as K(array(i32, N)) {}
  // CHECK:STDERR:                  ^~~~~~~~~~~~~~~~
  extend impl as K(array(i32, N)) {}
}

// C extends K so the type of K is completed, but is invalid.
// CHECK:STDERR: fail_class_extend_impl_does_need_complete_interface.carbon:[[@LINE+4]]:8: note: in `C(-1)` used here [ResolvingSpecificHere]
// CHECK:STDERR: var v: C(-1);
// CHECK:STDERR:        ^~~~~
// CHECK:STDERR:
var v: C(-1);

// --- interface_require_impls_doesnt_need_complete_interface.carbon
library "[[@TEST_NAME]]";

interface K(T:! type) {}
interface J(N:! i32) {
  require impls K(array(i32, N));
}

// J does not extend K so the type of K is not completed. No error.
var v: J(-1);

// --- fail_interface_extend_require_impls_does_need_complete_interface.carbon
library "[[@TEST_NAME]]";

interface K(T:! type) {}
interface J(N:! i32) {
  // CHECK:STDERR: fail_interface_extend_require_impls_does_need_complete_interface.carbon:[[@LINE+3]]:24: error: array bound of -1 is negative [ArrayBoundNegative]
  // CHECK:STDERR:   extend require impls K(array(i32, N));
  // CHECK:STDERR:                        ^~~~~~~~~~~~~~~~
  extend require impls K(array(i32, N));
}
interface I(N:! i32) {
  // CHECK:STDERR: fail_interface_extend_require_impls_does_need_complete_interface.carbon:[[@LINE+3]]:24: note: in `J(-1)` used here [ResolvingSpecificHere]
  // CHECK:STDERR:   extend require impls J(N);
  // CHECK:STDERR:                        ^~~~
  extend require impls J(N);
}

// I extends J extends K so the type of K is completed, but is invalid.
//
// TODO: The error location should be the type, like in the class case above. We
// need a location for the type in context.bind_name_map() to use as the
// location to Convert().
//
// CHECK:STDERR: fail_interface_extend_require_impls_does_need_complete_interface.carbon:[[@LINE+4]]:1: note: in `I(-1)` used here [ResolvingSpecificHere]
// CHECK:STDERR: var v: I(-1);
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
var v: I(-1);

// --- constraint_require_impls_doesnt_need_complete_interface.carbon
library "[[@TEST_NAME]]";

interface K(T:! type) {}
constraint J(N:! i32) {
  require impls K(array(i32, N));
}

// J does not extend K so the type of K is not completed. No error.
var v: J(-1);

// --- fail_constraint_extend_require_impls_does_need_complete_interface.carbon
library "[[@TEST_NAME]]";

interface K(T:! type) {}
constraint J(N:! i32) {
  // CHECK:STDERR: fail_constraint_extend_require_impls_does_need_complete_interface.carbon:[[@LINE+3]]:24: error: array bound of -1 is negative [ArrayBoundNegative]
  // CHECK:STDERR:   extend require impls K(array(i32, N));
  // CHECK:STDERR:                        ^~~~~~~~~~~~~~~~
  extend require impls K(array(i32, N));
}
constraint I(N:! i32) {
  // CHECK:STDERR: fail_constraint_extend_require_impls_does_need_complete_interface.carbon:[[@LINE+3]]:24: note: in `J(-1)` used here [ResolvingSpecificHere]
  // CHECK:STDERR:   extend require impls J(N);
  // CHECK:STDERR:                        ^~~~
  extend require impls J(N);
}

// I extends J extends K so the type of K is completed, but is invalid.
// CHECK:STDERR: fail_constraint_extend_require_impls_does_need_complete_interface.carbon:[[@LINE+4]]:1: note: in `I(-1)` used here [ResolvingSpecificHere]
// CHECK:STDERR: var v: I(-1);
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
var v: I(-1);

// --- interface_require_impls_doesnt_need_complete_self.carbon
library "[[@TEST_NAME]]";

interface K {}
interface J(N:! i32) {
  require array(Self, N) impls K;
}

// J does not extend K so the type of Self impling K is not completed. No error.
var v: J(-1);

// --- constraint_require_impls_doesnt_need_complete_self.carbon
library "[[@TEST_NAME]]";

interface K {}
constraint J(N:! i32) {
  require array(Self, N) impls K;
}

// J does not extend K so the type of Self impling K is not completed. No error.
var v: J(-1);
