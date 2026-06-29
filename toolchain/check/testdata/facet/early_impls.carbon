// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/early_impls.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/early_impls.carbon

// Tests that `.X impls Y` in a facet type is available in later constraints of
// the same facet type.

// --- member_access_uses_impls_constraint.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
  let Z2:! type;
}
interface Y {
  let Y1:! type;
}

fn F(_:! Z where .Z1 impls Y and .Z2 = .Z1.(Y.Y1)) {}

class C(T:! type);
constraint X {
  require C(Self) impls Y;
}

fn G(_:! Z where .Z1 impls X and .Z2 = C(.Z1).(Y.Y1)) {}

// --- early_self_impls.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
}
interface Y {
  let Y1:! type;
}

fn F(_:! Z where .Self impls Y and .Z1 = .Self.(Y.Y1)) {}

// --- early_generic_class_impls.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
}
interface Y {
  let Y1:! type;
}
class C(T:! type);

fn F(_:! Z where C(.Self) impls Y and .Z1 = C(.Self).(Y.Y1)) {}

// --- fail_todo_early_class_impls_generic_interface.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
}
interface Y(T:! type) {}
interface X(U:! Y(.Self)) {}
class C;

// Needs to identify `C as Y(.Self)` without replacing this `.Self` with `C`,
// since it refers to `T`.
// CHECK:STDERR: fail_todo_early_class_impls_generic_interface.carbon:[[@LINE+4]]:53: error: cannot convert type `C` into type implementing `Y(.Self)` [ConversionFailureTypeToFacet]
// CHECK:STDERR: fn F(unused T:! Z where C impls Y(.Self) and .Z1 = (C as Y(.Self))) {}
// CHECK:STDERR:                                                     ^~~~~~~~~~~~~
// CHECK:STDERR:
fn F(unused T:! Z where C impls Y(.Self) and .Z1 = (C as Y(.Self))) {}

class D(V:! type);

// Needs to identify `D(.Self) as Y(D(.Self))` without replacing the argument
// `.Self` to `D` since it refers to `T`. But we _do_ need to replace the
// `.Self` in the type of `U`, since it refers to `U` which is being replaced by
// `D(.Self)` in this specific.
// CHECK:STDERR: fail_todo_early_class_impls_generic_interface.carbon:[[@LINE+7]]:71: error: cannot convert type `D(.Self)` into type implementing `Y(.Self)` [ConversionFailureTypeToFacet]
// CHECK:STDERR: fn G(unused T:! Z where D(.Self) impls Y(D(.Self)) and D(.Self) impls X(D(.Self))) {}
// CHECK:STDERR:                                                                       ^~~~~~~~~~~
// CHECK:STDERR: fail_todo_early_class_impls_generic_interface.carbon:[[@LINE-20]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: interface X(U:! Y(.Self)) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G(unused T:! Z where D(.Self) impls Y(D(.Self)) and D(.Self) impls X(D(.Self))) {}

// --- fail_todo_early_class_impls_generic_interface_with_member_designator.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
  let Z2:! type;
}
interface Y(T:! type) {
  let Y1:! type;
}
class C;

// CHECK:STDERR: fail_todo_early_class_impls_generic_interface_with_member_designator.carbon:[[@LINE+4]]:43: error: cannot convert type `C` into type implementing `Y(.(Z.Z2))` [ConversionFailureTypeToFacet]
// CHECK:STDERR: fn F(_:! Z where C impls Y(.Z2) and .Z1 = C.(Y(.Z2).Y1)) {}
// CHECK:STDERR:                                           ^~~~~~~~~~~~~
// CHECK:STDERR:
fn F(_:! Z where C impls Y(.Z2) and .Z1 = C.(Y(.Z2).Y1)) {}

// --- fail_todo_early_tuple_impls_generic_interface.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
}
interface Y(T:! type) {
  let Y1:! type;
}
interface X {
  let X1:! type;
}
class C;

// CHECK:STDERR: fail_todo_early_tuple_impls_generic_interface.carbon:[[@LINE+4]]:46: error: cannot convert type `()` into type implementing `Y(.Self)` [ConversionFailureTypeToFacet]
// CHECK:STDERR: fn F(_:! Z where () impls Y(.Self) and .Z1 = ().(Y(.Self).Y1)) {}
// CHECK:STDERR:                                              ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(_:! Z where () impls Y(.Self) and .Z1 = ().(Y(.Self).Y1)) {}

// CHECK:STDERR: fail_todo_early_tuple_impls_generic_interface.carbon:[[@LINE+4]]:49: error: cannot convert type `(C,)` into type implementing `Y(.Self)` [ConversionFailureTypeToFacet]
// CHECK:STDERR: fn G(_:! Z where (C, ) impls Y(.Self) and .Z1 = (C, ).(Y(.Self).Y1)) {}
// CHECK:STDERR:                                                 ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G(_:! Z where (C, ) impls Y(.Self) and .Z1 = (C, ).(Y(.Self).Y1)) {}

// TODO: Can merge with the next test once they both pass.

// --- early_tuple_impls.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
}
interface X {
  let X1:! type;
}

fn H(_:! Z where (.Self, ) impls X and .Z1 = (.Self, ).(X.X1)) {}

// --- fail_todo_early_struct_impls.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
}
interface Y(T:! type) {
  let Y1:! type;
}
interface X {
  let X1:! type;
}
class C;

// CHECK:STDERR: fail_todo_early_struct_impls.carbon:[[@LINE+4]]:46: error: cannot convert type `{}` into type implementing `Y(.Self)` [ConversionFailureTypeToFacet]
// CHECK:STDERR: fn F(_:! Z where {} impls Y(.Self) and .Z1 = {}.(Y(.Self).Y1)) {}
// CHECK:STDERR:                                              ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(_:! Z where {} impls Y(.Self) and .Z1 = {}.(Y(.Self).Y1)) {}

// --- fail_todo_early_concrete_impls.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
}
interface Y(T:! type) {
  let Y1:! type;
}

// A concrete type that is not a class, tuple, or struct.
fn MakeIntLiteral() -> type = "int_literal.make_type";
alias IntLiteral = MakeIntLiteral();

// CHECK:STDERR: fail_todo_early_concrete_impls.carbon:[[@LINE+4]]:54: error: cannot convert type `IntLiteral` into type implementing `Y(.Self)` [ConversionFailureTypeToFacet]
// CHECK:STDERR: fn F(_:! Z where IntLiteral impls Y(.Self) and .Z1 = IntLiteral.(Y(.Self).Y1)) {}
// CHECK:STDERR:                                                      ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(_:! Z where IntLiteral impls Y(.Self) and .Z1 = IntLiteral.(Y(.Self).Y1)) {}

// --- early_impl_named_constraint.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {
  let Z1:! type;
  let Z2:! Y;
}
constraint N {
  require impls Y;
}

fn F(_:! Z where .Z1 impls N and .Z2 = .Z1) {}

// --- early_impl_named_constraint_gives_witness_for_different_self.carbon
library "[[@TEST_NAME]]";

class C;

interface Y(T:! type) {}
interface Z {
  let Z1:! type;
  let Z2:! type;
}

constraint N(V:! type) {
  require V impls Y(Self);
}

// `C impls N(.Z1)` gives `.Z1 impls Y(C)`.
fn F(_:! Z where C impls N(.Z1) and .Z2 = (.Z1 as Y(C))) {}

// --- fail_early_impl_named_constraint_gives_wrong_witness_for_different_self.carbon
library "[[@TEST_NAME]]";

class C;

interface Y(T:! type) {}
interface Z(U:! type) {
  let Z1:! type;
  let Z2:! type;
}

constraint N(V:! type) {
  require V impls Y(Self);
}

// `C impls N(.Z1)` gives `.Z1 impls Y(C)`.
//
// CHECK:STDERR: fail_early_impl_named_constraint_gives_wrong_witness_for_different_self.carbon:[[@LINE+4]]:51: error: cannot convert type `.(Z(.Self).Z1)` into type implementing `Y(.Self)` [ConversionFailureTypeToFacet]
// CHECK:STDERR: fn F(_:! Z(.Self) where C impls N(.Z1) and .Z2 = (.Z1 as Y(.Self))) {}
// CHECK:STDERR:                                                   ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(_:! Z(.Self) where C impls N(.Z1) and .Z2 = (.Z1 as Y(.Self))) {}

// --- fail_todo_early_type_impls_nested_self_impls.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
}
interface Y {
  let Y1:! type;
}
interface X {}

class C(T:! type);
class D(T:! X);

constraint NY {
  require impls Y where .Y1 impls X;
}

// A lookup of `C(V).(Y.Y1) as X` requires us to see that the `.Y1 impls X`
// constraint is visible through the `C(.Self) impls NY` constraint and that
// `C(.Self)` is used as the implied `.Self` in `.Y1 impls X`.
// CHECK:STDERR: fail_todo_early_type_impls_nested_self_impls.carbon:[[@LINE+7]]:53: error: cannot convert type `C(.Self).(Y.Y1)` into type implementing `X` [ConversionFailureTypeToFacet]
// CHECK:STDERR: fn F(unused V:! Z where C(.Self) impls NY and .Z1 = D(C(.Self).(Y.Y1))) {}
// CHECK:STDERR:                                                     ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_early_type_impls_nested_self_impls.carbon:[[@LINE-12]]:9: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: class D(T:! X);
// CHECK:STDERR:         ^~~~~
// CHECK:STDERR:
fn F(unused V:! Z where C(.Self) impls NY and .Z1 = D(C(.Self).(Y.Y1))) {}
