// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/impl_as_named_constraint.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/impl_as_named_constraint.carbon

// --- fail_empty_constraint.carbon
library "[[@TEST_NAME]]";

constraint A {}

// CHECK:STDERR: fail_empty_constraint.carbon:[[@LINE+4]]:1: error: impl as 0 interfaces, expected 1 [ImplOfNotOneInterface]
// CHECK:STDERR: impl () as A {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as A {}

// --- fail_too_many_interfaces_in_constraint.carbon
library "[[@TEST_NAME]]";

interface A1;
interface A2;
constraint B {
  extend require impls A1;
  extend require impls A2;
}

// CHECK:STDERR: fail_too_many_interfaces_in_constraint.carbon:[[@LINE+4]]:1: error: impl as 2 interfaces, expected 1 [ImplOfNotOneInterface]
// CHECK:STDERR: impl () as B {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as B {}

// --- one_extend_impls_interface_in_constraint.carbon
library "[[@TEST_NAME]]";

interface A;
constraint B {
  extend require impls A;
}

// Requries B identified.
impl () as B;

interface A {}

// Requries B complete.
impl () as B {}

// --- fail_one_impls_interface_in_constraint.carbon
library "[[@TEST_NAME]]";

interface A;
constraint B {
  require impls A;
}

// CHECK:STDERR: fail_one_impls_interface_in_constraint.carbon:[[@LINE+4]]:1: error: impl as 0 interfaces, expected 1 [ImplOfNotOneInterface]
// CHECK:STDERR: impl () as B {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as B {}

// --- nested_constraints.carbon
library "[[@TEST_NAME]]";

interface A {}
constraint B {
  extend require impls A;
}
constraint C {
  extend require impls B;
}

impl () as C {}

// --- fail_nested_constraints_not_extend_outer.carbon
library "[[@TEST_NAME]]";

interface A {}

constraint B {
  extend require impls A;
}
constraint C {
  require impls B;
}

// CHECK:STDERR: fail_nested_constraints_not_extend_outer.carbon:[[@LINE+4]]:1: error: impl as 0 interfaces, expected 1 [ImplOfNotOneInterface]
// CHECK:STDERR: impl () as C {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as C {}

// --- fail_nested_constraints_not_extend_inner.carbon
library "[[@TEST_NAME]]";

interface A {}

constraint B {
  require impls A;
}
constraint C {
  extend require impls B;
}

// CHECK:STDERR: fail_nested_constraints_not_extend_inner.carbon:[[@LINE+4]]:1: error: impl as 0 interfaces, expected 1 [ImplOfNotOneInterface]
// CHECK:STDERR: impl () as C {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as C {}

// --- fail_nested_constraints_not_extend_both.carbon
library "[[@TEST_NAME]]";

interface A {}

constraint B {
  require impls A;
}
constraint C {
  require impls B;
}

// CHECK:STDERR: fail_nested_constraints_not_extend_both.carbon:[[@LINE+4]]:1: error: impl as 0 interfaces, expected 1 [ImplOfNotOneInterface]
// CHECK:STDERR: impl () as C {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as C {}

// --- todo_fail_duplicate_through_generic_constraint.carbon
library "[[@TEST_NAME]]";

interface A(T:! type) {}

constraint B(X:! type, Y:! type) {
  extend require impls A(Y);
}

// This should impl () as A({}).
impl () as B((), {}) {}

// This should be considered a duplicate.
impl () as A({}) {}
