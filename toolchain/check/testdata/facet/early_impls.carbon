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

// --- early_class_impls_generic_interface.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
}
interface Y(T:! type) {}
interface X(U:! Y(.Self)) {}
class C;

// Needs to identify `C as Y(.Self)` without replacing this `.Self` with `C`,
// since it refers to `T`.
fn F(unused T:! Z where C impls Y(.Self) and .Z1 = (C as Y(.Self))) {}

class D(V:! type);

// Needs to identify `D(.Self) as Y(D(.Self))` without replacing the argument
// `.Self` to `D` since it refers to `T`. But we _do_ need to replace the
// `.Self` in the type of `U`, since it refers to `U` which is being replaced by
// `D(.Self)` in this specific.
fn G(unused T:! Z where D(.Self) impls Y(D(.Self)) and D(.Self) impls X(D(.Self))) {}

// --- early_class_impls_generic_interface_with_member_designator.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
  let Z2:! type;
}
interface Y(T:! type) {
  let Y1:! type;
}
class C;

fn F(_:! Z where C impls Y(.Z2) and .Z1 = C.(Y(.Z2).Y1)) {}

// --- early_tuple_impls_generic_interface.carbon
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

fn F(_:! Z where () impls Y(.Self) and .Z1 = ().(Y(.Self).Y1)) {}

fn G(_:! Z where (C, ) impls Y(.Self) and .Z1 = (C, ).(Y(.Self).Y1)) {}

fn H(_:! Z where (.Self, ) impls X and .Z1 = (.Self, ).(X.X1)) {}

// --- early_struct_impls.carbon
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

fn F(_:! Z where {} impls Y(.Self) and .Z1 = {}.(Y(.Self).Y1)) {}

// --- early_concrete_impls.carbon
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

// --- early_type_impls_nested_self_impls.carbon
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
fn F(unused V:! Z where C(.Self) impls NY and .Z1 = D(C(.Self).(Y.Y1))) {}

// --- period_self_impls_named_constraint_used_in_later_constraint.carbon
library "[[@TEST_NAME]]";

interface X {}
interface Y {
  let Y1:! type;
}

constraint ConstraintForY {
  require impls Y where .Y1 impls X;
}

interface Z {
  let Z1:! type;
}

private constraint ConstraintForZ {
  extend require impls Z where .Z1 impls ConstraintForY;
}

// If we replace `.Self` inside the named constraint `ConstraintForY`, we end up
// replacing `.Self` in the `.Z1` from the specific args for the named
// constraint, which replaces `Self` in the named constraint. Doing this is
// incorrect and was leading to an infinite loop of `.Self` substitution.
fn F(_:! ConstraintForZ where
           .Z1 impls ConstraintForY and
           .Z1.(Y.Y1) == ()) {}
