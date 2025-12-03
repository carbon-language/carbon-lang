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

interface A1 {}
interface A2 {}
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

interface A {}
constraint B {
  extend require impls A;
}

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

// --- fail_duplicate_through_generic_constraint.carbon
library "[[@TEST_NAME]]";

interface A(T:! type) {}

constraint B(X:! type, Y:! type) {
  extend require impls A(Y);
}

// This should impl () as A({}).
impl () as B((), {}) {}

// CHECK:STDERR: fail_duplicate_through_generic_constraint.carbon:[[@LINE+7]]:1: error: found non-final `impl` with the same type structure as another non-final `impl` [ImplNonFinalSameTypeStructure]
// CHECK:STDERR: impl () as A({}) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_duplicate_through_generic_constraint.carbon:[[@LINE-5]]:1: note: other `impl` here [ImplNonFinalSameTypeStructureNote]
// CHECK:STDERR: impl () as B((), {}) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as A({}) {}

// --- fail_error_self_in_require.carbon
library "[[@TEST_NAME]]";

interface A(T:! type) {}

// This makes the type of `Self` into an ErrorInst. No `require` is added to
// the constraint.
//@dump-sem-ir-begin
//
// CHECK:STDERR: fail_error_self_in_require.carbon:[[@LINE+4]]:18: error: name `Undefined` not found [NameNotFound]
// CHECK:STDERR: constraint B(X:! Undefined) {
// CHECK:STDERR:                  ^~~~~~~~~
// CHECK:STDERR:
constraint B(X:! Undefined) {
  extend require impls A(X);
}
//@dump-sem-ir-end

impl () as B(1) {}

// CHECK:STDOUT: --- fail_error_self_in_require.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B.type: type = generic_named_constaint_type @B [concrete]
// CHECK:STDOUT:   %empty_struct: %B.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %B.decl: %B.type = constraint_decl @B [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %X.patt: <error> = symbolic_binding_pattern X, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %X: <error> = symbolic_binding X, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @B(%X: <error>) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self: <error> = symbolic_binding Self, 1 [concrete = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self
// CHECK:STDOUT:     .A = <poisoned>
// CHECK:STDOUT:     .X = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(<error>) {}
// CHECK:STDOUT:
