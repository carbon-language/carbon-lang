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
  let Z1: type;
  let Z2: type;
}
interface Y {
  let Y1: type;
}

fn F(generic _: Z where .Z1 impls Y and .Z2 = .Z1.(Y.Y1)) {}

class C(T: type);
constraint X {
  require C(Self) impls Y;
}

fn G(generic _: Z where .Z1 impls X and .Z2 = C(.Z1).(Y.Y1)) {}

// --- early_self_impls.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}
interface Y {
  let Y1: type;
}

fn F(generic _: Z where .Self impls Y and .Z1 = .Self.(Y.Y1)) {}

// --- early_generic_class_impls.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}
interface Y {
  let Y1: type;
}
class C(T: type);

fn F(generic _: Z where C(.Self) impls Y and .Z1 = C(.Self).(Y.Y1)) {}

// --- early_class_impls_generic_interface.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}
interface Y(T: type) {}
interface X(U: Y(.Self)) {}
class C;

// Needs to identify `C as Y(.Self)` without replacing this `.Self` with `C`,
// since it refers to `T`.
fn F(unused generic T: Z where C impls Y(.Self) and .Z1 = (C as Y(.Self))) {}

class D(V: type);

// Needs to identify `D(.Self) as Y(D(.Self))` without replacing the argument
// `.Self` to `D` since it refers to `T`. But we _do_ need to replace the
// `.Self` in the type of `U`, since it refers to `U` which is being replaced by
// `D(.Self)` in this specific.
fn G(unused generic T: Z where D(.Self) impls Y(D(.Self)) and D(.Self) impls X(D(.Self))) {}

// --- early_class_impls_generic_interface_with_member_designator.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}
interface Y(T: type) {
  let Y1: type;
}
class C;

fn F(generic _: Z where C impls Y(.Z2) and .Z1 = C.(Y(.Z2).Y1)) {}

// --- early_tuple_impls_generic_interface.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}
interface Y(T: type) {
  let Y1: type;
}
interface X {
  let X1: type;
}
class C;

fn F(generic _: Z where () impls Y(.Self) and .Z1 = ().(Y(.Self).Y1)) {}

fn G(generic _: Z where (C, ) impls Y(.Self) and .Z1 = (C, ).(Y(.Self).Y1)) {}

fn H(generic _: Z where (.Self, ) impls X and .Z1 = (.Self, ).(X.X1)) {}

// --- early_struct_impls.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}
interface Y(T: type) {
  let Y1: type;
}
interface X {
  let X1: type;
}
class C;

fn F(generic _: Z where {} impls Y(.Self) and .Z1 = {}.(Y(.Self).Y1)) {}

// --- early_concrete_impls.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}
interface Y(T: type) {
  let Y1: type;
}

// A concrete type that is not a class, tuple, or struct.
fn MakeIntLiteral() -> type = "int_literal.make_type";
alias IntLiteral = MakeIntLiteral();

fn F(generic _: Z where IntLiteral impls Y(.Self) and .Z1 = IntLiteral.(Y(.Self).Y1)) {}

// --- early_impl_named_constraint.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {
  let Z1: type;
  let Z2: Y;
}
constraint N {
  require impls Y;
}

fn F(generic _: Z where .Z1 impls N and .Z2 = .Z1) {}

// --- early_impl_named_constraint_gives_witness_for_different_self.carbon
library "[[@TEST_NAME]]";

class C;

interface Y(T: type) {}
interface Z {
  let Z1: type;
  let Z2: type;
}

constraint N(V: type) {
  require V impls Y(Self);
}

// `C impls N(.Z1)` gives `.Z1 impls Y(C)`.
fn F(generic _: Z where C impls N(.Z1) and .Z2 = (.Z1 as Y(C))) {}

// --- fail_early_impl_named_constraint_gives_wrong_witness_for_different_self.carbon
library "[[@TEST_NAME]]";

class C;

interface Y(T: type) {}
interface Z(U: type) {
  let Z1: type;
  let Z2: type;
}

constraint N(V: type) {
  require V impls Y(Self);
}

// `C impls N(.Z1)` gives `.Z1 impls Y(C)`.
//
// CHECK:STDERR: fail_early_impl_named_constraint_gives_wrong_witness_for_different_self.carbon:[[@LINE+2]]:58-72: error: cannot convert type `.(Z(.Self).Z1)` into type implementing `Y(.Self)` [ConversionFailureTypeToFacet]
// CHECK:STDERR:
fn F(generic _: Z(.Self) where C impls N(.Z1) and .Z2 = (.Z1 as Y(.Self))) {}

// --- early_type_impls_nested_self_impls.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}
interface Y {
  let Y1: type;
}
interface X {}

class C(T: type);
class D(T: X);

constraint NY {
  require impls Y where .Y1 impls X;
}

// A lookup of `C(V).(Y.Y1) as X` requires us to see that the `.Y1 impls X`
// constraint is visible through the `C(.Self) impls NY` constraint and that
// `C(.Self)` is used as the implied `.Self` in `.Y1 impls X`.
fn F(unused generic V: Z where C(.Self) impls NY and .Z1 = D(C(.Self).(Y.Y1))) {}

// --- early_impls_with_period_self_not_replaced.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) {}
class C;

class NeedZ(V: Z(.Self));
class NeedZU(U: type, V: Z(U));

interface S(T: type);

// The conversion `C as Z(.Self)` written explicitly, where the `.Self` refers
// to `T` and thus must not be replaced. It is not `C as Z(C)`.
fn F1(unused generic T: type where C impls Z(.Self) and (C as Z(.Self)) impls S(.Self)) {}

// NeedZU contains a `C as Z(.Self)` conversion and identify step, but the
// `.Self` has been smuggled out of the facet type being constructed and placed
// into a new facet type, which is the type of the `V` parameter. The `.Self`
// refers to `T` and we need to keep track of that so it is not replaced during
// the conversion.
fn F2(unused generic T: type where C impls Z(.Self) and NeedZU(.Self, C) impls S(.Self)) {}

// NeedZ contains a `C as Z(.Self)` conversion but the `.Self` refers to the `V`
// parameter which is being replaced by `C`, so it's `C as Z(C)`. The `.Self`
// there should be replaced during the conversion.
impl C as Z(C) {}
fn F3(unused generic T: type where C impls Z(.Self) and NeedZ(C) impls S(.Self)) {}

// --- fail_early_impls_with_period_self_replaced.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) {}
class C;

class NeedZ(V: Z(.Self));

interface S(T: type);

// NeedZ contains a `C as Z(.Self)` conversion but the `.Self` refers to the `V`
// parameter which is being replaced by `C`, so it's `C as Z(C)`. The `.Self`
// there should be replaced during the conversion. We show that it was replaced
// because the `C impls Z(.Self)` constraint does not satisfy `C as Z(C)` so the
// conversion fails.
// CHECK:STDERR: fail_early_impls_with_period_self_replaced.carbon:[[@LINE+3]]:56-63: error: cannot convert type `C` into type implementing `Z(.Self)` [ConversionFailureTypeToFacet]
// CHECK:STDERR: fail_early_impls_with_period_self_replaced.carbon:[[@LINE-10]]:1-25: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR:
fn F(unused generic T: type where C impls Z(.Self) and NeedZ(C) impls S(.Self)) {}

// --- period_self_impls_named_constraint_used_in_later_constraint.carbon
library "[[@TEST_NAME]]";

interface X {}
interface Y {
  let Y1: type;
}

constraint ConstraintForY {
  require impls Y where .Y1 impls X;
}

interface Z {
  let Z1: type;
}

private constraint ConstraintForZ {
  extend require impls Z where .Z1 impls ConstraintForY;
}

// If we replace `.Self` inside the named constraint `ConstraintForY`, we end up
// replacing `.Self` in the `.Z1` from the specific args for the named
// constraint, which replaces `Self` in the named constraint. Doing this is
// incorrect and was leading to an infinite loop of `.Self` substitution.
fn F(generic _: ConstraintForZ where
           .Z1 impls ConstraintForY and
           .Z1.(Y.Y1) == ()) {}
