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

// --- fail_inteface_require_incomplete_constraint.carbon
library "[[@TEST_NAME]]";

constraint A;
interface B {
  // CHECK:STDERR: fail_inteface_require_incomplete_constraint.carbon:[[@LINE+4]]:3: error: semantics TODO: `require` [SemanticsTodo]
  // CHECK:STDERR:   require impls A;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  require impls A;
}

class C {}

// TODO: This should fail since B can not be identified.
impl C as B {}

// --- fail_incomplete_constraint.carbon
library "[[@TEST_NAME]]";

constraint A;

class C {}

// CHECK:STDERR: fail_incomplete_constraint.carbon:[[@LINE+7]]:1: error: facet type `A` cannot be identified [ImplOfUnidentifiedFacetType]
// CHECK:STDERR: impl C as A {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_incomplete_constraint.carbon:[[@LINE-7]]:1: note: constraint was forward declared here [NamedConstraintForwardDeclaredHere]
// CHECK:STDERR: constraint A;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C as A {}

// --- fail_empty_constraint.carbon
library "[[@TEST_NAME]]";

constraint A {}

class C {}

// CHECK:STDERR: fail_empty_constraint.carbon:[[@LINE+4]]:1: error: impl as 0 interfaces, expected 1 [ImplOfNotOneInterface]
// CHECK:STDERR: impl C as A {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C as A {}

// --- fail_too_many_interfaces_in_constraint.carbon
library "[[@TEST_NAME]]";

interface A1;
interface A2;
constraint B {
  // CHECK:STDERR: fail_too_many_interfaces_in_constraint.carbon:[[@LINE+4]]:3: error: semantics TODO: `require` [SemanticsTodo]
  // CHECK:STDERR:   require impls A1;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  require impls A1;
  require impls A2;
}

class C {}

// TODO: This should fail since B does not name a single interface.
impl C as B {}

// --- fail_todo_one_declared_interface_in_constraint.carbon
library "[[@TEST_NAME]]";

// TODO: This should work since B can be identified to have one interface.

interface A;
constraint B {
  // CHECK:STDERR: fail_todo_one_declared_interface_in_constraint.carbon:[[@LINE+4]]:3: error: semantics TODO: `require` [SemanticsTodo]
  // CHECK:STDERR:   require impls A;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  require impls A;
}

class C {}
impl C as B {}

// --- fail_todo_one_defined_interface_in_constraint.carbon
library "[[@TEST_NAME]]";

// TODO: This should work since B can be identified to have one interface.

interface A {}
constraint B {
  // CHECK:STDERR: fail_todo_one_defined_interface_in_constraint.carbon:[[@LINE+4]]:3: error: semantics TODO: `require` [SemanticsTodo]
  // CHECK:STDERR:   require impls A;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  require impls A;
}

class C {}
impl C as B {}

// --- fail_one_declared_interface_in_constraint_with_rewrite.carbon
library "[[@TEST_NAME]]";

interface A;
constraint B {
  // TODO: This should fail because A is incomplete.
  // CHECK:STDERR: fail_one_declared_interface_in_constraint_with_rewrite.carbon:[[@LINE+4]]:3: error: semantics TODO: `require` [SemanticsTodo]
  // CHECK:STDERR:   require impls A where .X = {};
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  require impls A where .X = {};
}

class C {}

impl C as B {}
