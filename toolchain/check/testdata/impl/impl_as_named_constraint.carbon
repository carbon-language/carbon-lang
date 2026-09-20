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

// CHECK:STDERR: fail_empty_constraint.carbon:[[@LINE+2]]:1-14: error: impl as 0 interfaces, expected 1 [ImplOfNotOneInterface]
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

// CHECK:STDERR: fail_too_many_interfaces_in_constraint.carbon:[[@LINE+2]]:1-14: error: impl as 2 interfaces, expected 1 [ImplOfNotOneInterface]
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

// CHECK:STDERR: fail_one_impls_interface_in_constraint.carbon:[[@LINE+2]]:1-14: error: impl as 0 interfaces, expected 1 [ImplOfNotOneInterface]
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

// CHECK:STDERR: fail_nested_constraints_not_extend_outer.carbon:[[@LINE+2]]:1-14: error: impl as 0 interfaces, expected 1 [ImplOfNotOneInterface]
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

// CHECK:STDERR: fail_nested_constraints_not_extend_inner.carbon:[[@LINE+2]]:1-14: error: impl as 0 interfaces, expected 1 [ImplOfNotOneInterface]
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

// CHECK:STDERR: fail_nested_constraints_not_extend_both.carbon:[[@LINE+2]]:1-14: error: impl as 0 interfaces, expected 1 [ImplOfNotOneInterface]
// CHECK:STDERR:
impl () as C {}

// --- fail_duplicate_through_generic_constraint.carbon
library "[[@TEST_NAME]]";

interface A(T: type) {}

constraint B(X: type, Y: type) {
  extend require impls A(Y);
}

// This should impl () as A({}).
impl () as B((), {}) {}

// CHECK:STDERR: fail_duplicate_through_generic_constraint.carbon:[[@LINE+3]]:1-18: error: found non-final `impl` with the same type structure as another non-final `impl` [ImplNonFinalSameTypeStructure]
// CHECK:STDERR: fail_duplicate_through_generic_constraint.carbon:[[@LINE-3]]:1-22: note: other `impl` here [ImplNonFinalSameTypeStructureNote]
// CHECK:STDERR:
impl () as A({}) {}

// --- fail_error_self_in_require.carbon
library "[[@TEST_NAME]]";

interface A(T: type) {}

// This makes the type of `Self` into an ErrorInst. No `require` is added to
// the constraint.
//@dump-sem-ir-begin
//
// CHECK:STDERR: fail_error_self_in_require.carbon:[[@LINE+2]]:17-25: error: name `Undefined` not found [NameNotFound]
// CHECK:STDERR:
constraint B(X: Undefined) {
  extend require impls A(X);
}
//@dump-sem-ir-end

impl () as B(1) {}

// --- impl_as_constraint_where_rewrite.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1: type;
}

constraint N {
  extend require impls I;
}

class C;
impl C as N where .I1 = C {}

fn F(generic _: I where .I1 = .Self) {}
fn G() {
  F(C);
}

// --- impl_as_concrete_generic_constraint_where_rewrite.carbon
library "[[@TEST_NAME]]";

interface I(T: type) {
  let I1: type;
}

class C;

constraint N {
  extend require impls I(C);
}

impl C as N where .I1 = C {}

fn F(generic _: I(.Self) where .I1 = .Self) {}
fn G() {
  F(C);
}

// --- impl_as_symbolic_generic_constraint_where_rewrite.carbon
library "[[@TEST_NAME]]";

interface I(T: type) {
  let I1: type;
}

constraint N {
  extend require impls I(Self);
}

class C;
impl C as N where .I1 = C {}

fn F(generic _: I(.Self) where .I1 = .Self) {}
fn G() {
  F(C);
}

// --- fail_todo_impl_as_constraint_containing_rewrite_for_concrete_interface.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1: type;
}

constraint N {
  extend require impls I where .I1 = Self;
}

class C;
// CHECK:STDERR: fail_todo_impl_as_constraint_containing_rewrite_for_concrete_interface.carbon:[[@LINE+3]]:1-13: error: associated constant I1 not given a value in impl of interface I [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: fail_todo_impl_as_constraint_containing_rewrite_for_concrete_interface.carbon:[[@LINE-9]]:7-14: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR:
impl C as N {}

fn F(generic _: I where .I1 = .Self) {}
fn G() {
  // CHECK:STDERR: fail_todo_impl_as_constraint_containing_rewrite_for_concrete_interface.carbon:[[@LINE+3]]:3-6: error: cannot convert type `C` into type implementing `I where .(I.I1) = .Self` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_todo_impl_as_constraint_containing_rewrite_for_concrete_interface.carbon:[[@LINE-3]]:14-35: note: initializing generic parameter `_` declared here [InitializingGenericParam]
  // CHECK:STDERR:
  F(C);
}

// --- fail_todo_impl_as_constraint_containing_rewrite_for_generic_interface.carbon
library "[[@TEST_NAME]]";

interface I(T: type) {
  let I1: type;
}

constraint N {
  extend require impls I(Self) where .I1 = Self;
}

class C;
// CHECK:STDERR: fail_todo_impl_as_constraint_containing_rewrite_for_generic_interface.carbon:[[@LINE+3]]:1-13: error: associated constant I1 not given a value in impl of interface I [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: fail_todo_impl_as_constraint_containing_rewrite_for_generic_interface.carbon:[[@LINE-9]]:7-14: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR:
impl C as N {}

fn F(generic _: I(.Self) where .I1 = .Self) {}
fn G() {
  // CHECK:STDERR: fail_todo_impl_as_constraint_containing_rewrite_for_generic_interface.carbon:[[@LINE+3]]:3-6: error: cannot convert type `C` into type implementing `I(.Self) where .(I(.Self).I1) = .Self` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_todo_impl_as_constraint_containing_rewrite_for_generic_interface.carbon:[[@LINE-3]]:1-45: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  F(C);
}

// --- fail_no_require_self.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) {}

class C;

constraint Y {
  // `Self` must appear somewhere in the require decl, so it's in the interface
  // specific because we want to test a different self-type.
  require C impls Z(Self);
}

// Y does not have any extend require on `Self`.
// CHECK:STDERR: fail_no_require_self.carbon:[[@LINE+2]]:1-14: error: impl as 0 interfaces, expected 1 [ImplOfNotOneInterface]
// CHECK:STDERR:
impl () as Y {}

// --- member_access_in_extend_require.carbon
library "[[@TEST_NAME]]";

interface I { let X: type; }
interface J {}

impl () as J {}

constraint GivesI {
  extend require impls I where .X impls J;
}

// When we check this we check that GivesI is complete, which:
// - Forms the self-specific for GivesI's extend require
// - Creates Self.(I.X) which has a LookupImplWitness for `Self as I`
// - Evaluates the witness which finds this impl
// - Deduces for Self which tries to convert Self to type
// - Requires the type of Self to be complete, and its type is GivesI
// - Forms the self-specfic for GivesI's extend require....we have a loop.
impl forall [T: type] T as GivesI where .X = () {}

// CHECK:STDOUT: --- fail_error_self_in_require.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B.type: type = generic_named_constaint_type @B [concrete]
// CHECK:STDOUT:   %empty_struct: %B.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %B.decl: %B.type = constraint_decl @B [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %X.patt: <error> = symbolic_binding_pattern X, 0 [concrete = <error>]
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
// CHECK:STDOUT:     %B.WithSelf.decl = constraint_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self
// CHECK:STDOUT:     .A = <poisoned>
// CHECK:STDOUT:     .X = <poisoned>
// CHECK:STDOUT:     .A = <poisoned>
// CHECK:STDOUT:     .X = <poisoned>
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(<error>) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(<error>, <error>) {}
// CHECK:STDOUT:
