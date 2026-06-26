// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/period_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/period_self.carbon

// --- period_self_param.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
interface I(T:! type) {
  let I1:! type;
}

fn F(T:! I(.Self) where .I1 = ()) -> T.I1 {
  return ();
}

fn G(T:! I(.Self as type) where .I1 = ()) -> T.I1 {
  return ();
}
//@dump-sem-ir-end

// --- underscore_identifier_name.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) {
  let I1:! type;
}

// Underscore as the identifier name produces a different parse tree for the
// binding pattern.
fn G(_:! I(.Self) where .I1 = ()) {}

// --- fail_period_self_as_type.carbon
library "[[@TEST_NAME]]";

// TODO: We should diagnose this use of `.Self` directly rather than later when
// it is converted to `type`.
interface I(T:! .Self) {
  // CHECK:STDERR: fail_period_self_as_type.carbon:[[@LINE+7]]:13: error: cannot implicitly convert non-type value of type `.Self` to `type` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   fn G() -> T;
  // CHECK:STDERR:             ^
  // CHECK:STDERR: fail_period_self_as_type.carbon:[[@LINE+4]]:13: note: type `.Self` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   fn G() -> T;
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  fn G() -> T;
}

// --- convert_period_self_to_full_facet_value.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) {}

fn F(U:! I(.Self)) {
  U as I(U);
  (U as type) as I(U);
}

// --- convert_period_self_to_full_facet_value_with_assoc_constant.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) {
  let X:! type;
}

fn F(U:! I(.Self) where .X = .Self) {
  U as (I(U) where .X = U);
  (U as type) as (I(U) where .X = U);
}

// --- fail_todo_return_of_type_period_self.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) {
  fn G() -> T*;
}

interface J(T:! type) {
  fn J1() -> T*;
}

fn F2[U:! I(.Self)](unused T: U*) {}

fn F(U:! I(.Self)) {
  // Caller sees the returned `.Self` type from `F()` as the full `U`.
  // CHECK:STDERR: fail_todo_return_of_type_period_self.carbon:[[@LINE+4]]:3: error: member name `G` not found [MemberNameNotFound]
  // CHECK:STDERR:   U.G()->G()->G();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  U.G()->G()->G();
  // Conversion to `type` retains access to all of `U`.
  // CHECK:STDERR: fail_todo_return_of_type_period_self.carbon:[[@LINE+4]]:3: error: member name `G` not found [MemberNameNotFound]
  // CHECK:STDERR:   (U as type).G()->G()->G();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (U as type).G()->G()->G();

  // Using the returned `.Self` as a facet value works, not just member lookup
  // on its facet type.
  // CHECK:STDERR: fail_todo_return_of_type_period_self.carbon:[[@LINE+4]]:6: error: member name `G` not found [MemberNameNotFound]
  // CHECK:STDERR:   F2(U.G()->G()->G());
  // CHECK:STDERR:      ^~~~~~~~
  // CHECK:STDERR:
  F2(U.G()->G()->G());
}

// --- fail_todo_return_of_type_period_self_extends_interface.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) {
  fn I1() -> T*;
}

interface J(T:! type) {
  fn J1() -> T*;
}

fn F2[U:! I(.Self) & J(.Self)](unused T: U*) {}

fn F(U:! I(.Self) & J(.Self)) {
  // TODO: The returned value of `I1` and `J1` has type `U` which has access to
  // the methods of `I` and `J`.
  // CHECK:STDERR: fail_todo_return_of_type_period_self_extends_interface.carbon:[[@LINE+4]]:3: error: member name `J1` not found [MemberNameNotFound]
  // CHECK:STDERR:   U.I1()->J1()->I1()->J1()->I1()->J1();
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  U.I1()->J1()->I1()->J1()->I1()->J1();
  // CHECK:STDERR: fail_todo_return_of_type_period_self_extends_interface.carbon:[[@LINE+4]]:3: error: member name `J1` not found [MemberNameNotFound]
  // CHECK:STDERR:   (U as type).I1()->J1()->I1()->J1()->I1()->J1();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (U as type).I1()->J1()->I1()->J1()->I1()->J1();

  // TODO: Using the returned value of type `U` as a facet value works.
  // CHECK:STDERR: fail_todo_return_of_type_period_self_extends_interface.carbon:[[@LINE+4]]:6: error: member name `J1` not found [MemberNameNotFound]
  // CHECK:STDERR:   F2(U.I1()->J1()->I1()->J1()->I1()->J1());
  // CHECK:STDERR:      ^~~~~~~~~~
  // CHECK:STDERR:
  F2(U.I1()->J1()->I1()->J1()->I1()->J1());
}

// --- fail_todo_return_of_period_self_impls_interface.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) {
  fn I1() -> T;
}

interface J(T:! type) {
  fn J1() -> T;
}

fn G(U:! I(.Self) where .Self impls J(.Self)) {
  // Compound member lookup through a non-type value is possible for methods
  // which take a `self` parameter. But it's not possible for methods without
  // `self`. For those you need to go directly throug the type.
  // See: https://github.com/carbon-language/carbon-lang/issues/6025

  // TODO: This step should work.
  //
  // CHECK:STDERR: fail_todo_return_of_period_self_impls_interface.carbon:[[@LINE+7]]:14: error: cannot implicitly convert expression of type `.Self` to `U` [ConversionFailure]
  // CHECK:STDERR:   let u: U = U.I1();
  // CHECK:STDERR:              ^~~~~~
  // CHECK:STDERR: fail_todo_return_of_period_self_impls_interface.carbon:[[@LINE+4]]:14: note: type `.Self` does not implement interface `Core.ImplicitAs(U)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let u: U = U.I1();
  // CHECK:STDERR:              ^~~~~~
  // CHECK:STDERR:
  let u: U = U.I1();
  // `u` is a non-type value. Can call methods with `self` through compound
  // member lookup, but can't call methods without `self`. See the
  // `compound_access_through_call_with_self_param.carbon` test for the former.
  //
  // CHECK:STDERR: fail_todo_return_of_period_self_impls_interface.carbon:[[@LINE+4]]:6: error: type `<type of J>` does not support qualified expressions [QualifiedExprUnsupported]
  // CHECK:STDERR:   u.(J.J1)();
  // CHECK:STDERR:      ^~~~
  // CHECK:STDERR:
  u.(J.J1)();

  // This is the same as the above, since U.I1() returns a non-type value of
  // type `U`.
  //
  // CHECK:STDERR: fail_todo_return_of_period_self_impls_interface.carbon:[[@LINE+4]]:11: error: type `<type of J>` does not support qualified expressions [QualifiedExprUnsupported]
  // CHECK:STDERR:   U.I1().(J.J1)();
  // CHECK:STDERR:           ^~~~
  // CHECK:STDERR:
  U.I1().(J.J1)();
}

// --- fail_todo_return_of_type_period_self_has_type_u.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) {
  fn G() -> T;
}

fn F(U:! I(.Self)) {
  // CHECK:STDERR: fail_todo_return_of_type_period_self_has_type_u.carbon:[[@LINE+7]]:21: error: cannot implicitly convert expression of type `.Self` to `U` [ConversionFailure]
  // CHECK:STDERR:   let unused a: U = U.G();
  // CHECK:STDERR:                     ^~~~~
  // CHECK:STDERR: fail_todo_return_of_type_period_self_has_type_u.carbon:[[@LINE+4]]:21: note: type `.Self` does not implement interface `Core.ImplicitAs(U)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused a: U = U.G();
  // CHECK:STDERR:                     ^~~~~
  // CHECK:STDERR:
  let unused a: U = U.G();
}

// --- return_of_type_period_self_assoc_const_has_type_u.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  fn G() -> X;
}

fn F(U:! Core.Destroy & I where .X = .Self) {
  let unused a: U = U.G();
}

// --- period_self_parameter_sees_lhs_of_where_expr.carbon
library "[[@TEST_NAME]]";

interface I(T:! Core.Destroy) {}

// The `.Self` can see the LHS of the `where` to know `U` impls Core.Destroy.
fn F(unused U:! Core.Destroy where .Self impls I(.Self)) {}

// --- fail_todo_period_self_parameter_constraint_satisfied_with_type_and.carbon
library "[[@TEST_NAME]]";

interface I(T:! Core.Destroy) {}

// TODO: Implied constraints that `.Self` impls `Core.Destroy` are satisfied by
// the `&` expression.
//
// CHECK:STDERR: fail_todo_period_self_parameter_constraint_satisfied_with_type_and.carbon:[[@LINE+7]]:32: error: cannot convert type `.Self` that implements `type` into type implementing `Core.Destroy` [ConversionFailureFacetToFacet]
// CHECK:STDERR: fn F(unused U:! Core.Destroy & I(.Self)) {}
// CHECK:STDERR:                                ^~~~~~~~
// CHECK:STDERR: fail_todo_period_self_parameter_constraint_satisfied_with_type_and.carbon:[[@LINE-8]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: interface I(T:! Core.Destroy) {}
// CHECK:STDERR:             ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(unused U:! Core.Destroy & I(.Self)) {}

// --- compound_lookup_on_returned_period_self_parameter.carbon
library "[[@TEST_NAME]]";

interface I(T:! Core.Destroy) {
  fn G(self) -> T;
}

fn F[U:! Core.Destroy where .Self impls I(.Self)](u: U) {
  // This tests that both `I.G` is accessible and that `Destroy` is preserved;
  // we'd get an error for missing Destroy otherwise since G() returns an
  // initializing expression.
  u.(I(U).G)().(I(U).G)().(I(U).G)();
}

// --- impl_as_rewrite_with_period_self.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
}

class C;
impl C as Z where .Z1 = .Self {}

fn F() {
  C as (Z where .Z1 = C);
}

// --- impl_as_impls_with_period_self.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
}
interface Y {}

class C;
impl C as Z where .Z1 = .Self {}
impl C as Y {}

fn F() {
  C as (Z where .Z1 impls Y);
}

// --- fail_infinite_cycle_identifying_associated_constant_as_impls_type_missing_impls_m.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

interface M {}
final impl forall [T:! M] T as L where .W = {} {}

interface Z {}

// This is missing `.Self impls M` so the `final impl` is not used, through
// deduction is attempted.
//
// CHECK:STDERR: fail_infinite_cycle_identifying_associated_constant_as_impls_type_missing_impls_m.carbon:[[@LINE+7]]:46: error: cannot implicitly convert expression of type `T.(L.W)` to `{}` [ConversionFailure]
// CHECK:STDERR: fn F(T:! L where .W impls Z, a: T.W) -> {} { return a; }
// CHECK:STDERR:                                              ^~~~~~~~~
// CHECK:STDERR: fail_infinite_cycle_identifying_associated_constant_as_impls_type_missing_impls_m.carbon:[[@LINE+4]]:46: note: type `T.(L.W)` does not implement interface `Core.ImplicitAs({})` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: fn F(T:! L where .W impls Z, a: T.W) -> {} { return a; }
// CHECK:STDERR:                                              ^~~~~~~~~
// CHECK:STDERR:
fn F(T:! L where .W impls Z, a: T.W) -> {} { return a; }

// --- infinite_cycle_identifying_associated_constant_as_impls_type.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

interface M {}
final impl forall [T:! M] T as L where .W = {} {}

interface Z {}

// This can make an infinite cycle in the toolchain:
// 0. `T.W` looks for a witness for `L` in the facet type of `T`
// 1. We identify the facet type of `T` and replace `.Self` with `T` throughout.
// 2. The `.W` becomes `T.W` which is evaluated and does an impl lookup `T as L`.
// 3. We find the final impl for `T:! M` and try to convert `T as M`.
// 4. We look for a witness for `M` in the facet type of `T`.
// 5. Go to step 1.
//
// Because the `.W` is part of an `impls` constraint, it is substituted as part
// of identifying.
fn F(T:! L where .W impls Z and .Self impls M, a: T.W) -> {} { return a; }

// --- fail_infinite_cycle_identifying_associated_constant_in_impls_interface_as_type_missing_m.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

interface M {}
final impl forall [T:! M] T as L where .W = () {}

interface Z(T:! type) {}

// This is missing `.Self impls M` so the `final impl` is not used, through
// deduction is attempted.
//
// CHECK:STDERR: fail_infinite_cycle_identifying_associated_constant_in_impls_interface_as_type_missing_m.carbon:[[@LINE+7]]:53: error: cannot implicitly convert expression of type `T.(L.W)` to `()` [ConversionFailure]
// CHECK:STDERR: fn F(T:! L where .Self impls Z(.W), a: T.W) -> () { return a; }
// CHECK:STDERR:                                                     ^~~~~~~~~
// CHECK:STDERR: fail_infinite_cycle_identifying_associated_constant_in_impls_interface_as_type_missing_m.carbon:[[@LINE+4]]:53: note: type `T.(L.W)` does not implement interface `Core.ImplicitAs(())` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: fn F(T:! L where .Self impls Z(.W), a: T.W) -> () { return a; }
// CHECK:STDERR:                                                     ^~~~~~~~~
// CHECK:STDERR:
fn F(T:! L where .Self impls Z(.W), a: T.W) -> () { return a; }

// --- infinite_cycle_identifying_associated_constant_in_impls_interface_as_type.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

interface M {}
final impl forall [T:! M] T as L where .W = () {}

interface Z(T:! type) {}

// This can make an infinite cycle in the toolchain:
// 0. `T.W` looks for a witness for `L` in the facet type of `T`
// 1. We identify the facet type of `T` and replace `.Self` with `T` throughout.
// 2. The `.W` becomes `T.W` which is evaluated and does an impl lookup `T as L`.
// 3. We find the final impl for `T:! M` and try to convert `T as M`.
// 4. We look for a witness for `M` in the facet type of `T`.
// 5. Go to step 1.
//
// Because the `.W` is part of an `impls` constraint, it is substituted as part
// of identifying. In this case it is inside a FacetAccessType instruction.
fn F(T:! L where .Self impls Z(.W) and .Self impls M, a: T.W) -> () { return a; }

// --- fail_infinite_cycle_identifying_associated_constant_in_impls_interface_as_facet_type_missing_m.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

interface M {}
final impl forall [T:! M] T as L where .W = () {}

interface Y {}
interface Z(T:! Y) {}

// This is missing `.Self impls M` so the `final impl` is not used, through
// deduction is attempted.
//
// CHECK:STDERR: fail_infinite_cycle_identifying_associated_constant_in_impls_interface_as_facet_type_missing_m.carbon:[[@LINE+7]]:68: error: cannot implicitly convert expression of type `T.(L.W)` to `()` [ConversionFailure]
// CHECK:STDERR: fn F(T:! L where .W impls Y and .Self impls Z(.W), a: T.W) -> () { return a; }
// CHECK:STDERR:                                                                    ^~~~~~~~~
// CHECK:STDERR: fail_infinite_cycle_identifying_associated_constant_in_impls_interface_as_facet_type_missing_m.carbon:[[@LINE+4]]:68: note: type `T.(L.W)` does not implement interface `Core.ImplicitAs(())` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: fn F(T:! L where .W impls Y and .Self impls Z(.W), a: T.W) -> () { return a; }
// CHECK:STDERR:                                                                    ^~~~~~~~~
// CHECK:STDERR:
fn F(T:! L where .W impls Y and .Self impls Z(.W), a: T.W) -> () { return a; }

// --- infinite_cycle_identifying_associated_constant_in_impls_interface_as_facet_type.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

interface M {}
final impl forall [T:! M] T as L where .W = () {}

interface Y {}
interface Z(T:! Y) {}

// This can make an infinite cycle in the toolchain:
// 0. `T.W` looks for a witness for `L` in the facet type of `T`
// 1. We identify the facet type of `T` and replace `.Self` with `T` throughout.
// 2. The `.W` becomes `T.W` which is evaluated and does an impl lookup `T as L`.
// 3. We find the final impl for `T:! M` and try to convert `T as M`.
// 4. We look for a witness for `M` in the facet type of `T`.
// 5. Go to step 1.
//
// Because the `.W` is part of an `impls` constraint, it is substituted as part
// of identifying. In this case it is inside a FacetValue instruction, which is
// converting it to `Y`.
fn F(T:! L where .W impls Y and .Self impls Z(.W) and .Self impls M, a: T.W) -> () { return a; }

// --- concrete_access_witness_m_in_extend_constraint.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

interface M {}
final impl forall [T:! M] T as L where .W = () {}

interface Z(Z1:! type) {}

class C(C1:! type);

// Should pass. `.W` can resolve to `()` immediately through the type of `.Self`
// (since the type of `.Self` contains `M`) and the `final impl`. So we can get
// a witness for `C(())` from `T`.
fn I(T:! (L & M) where C(.W) impls Z(.Self)) {
  C(()) as Z(T);
}

// --- fail_todo_concrete_access_witness_m_in_impls_constraint.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

interface M {}
final impl forall [T:! M] T as L where .W = () {}

interface Z(Z1:! type) {}

class C(C1:! type);

// Should pass. `.W` can resolve to `()` immediately through the type of `.Self`
// (since the type of `.Self` contains `M`) and the `final impl`. So we can get
// a witness for `C(())` from `T`.
//
// TODO: How come we don't get `.W` evaluating to `()` when `M` comes from a
// non-extend constraint?
fn H(T:! (L where .Self impls M) where C(.W) impls Z(.Self)) {
  // CHECK:STDERR: fail_todo_concrete_access_witness_m_in_impls_constraint.carbon:[[@LINE+4]]:3: error: cannot convert type `C(())` into type implementing `Z(T)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   C(()) as Z(T);
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  C(()) as Z(T);
}

// --- fail_todo_concrete_access_witness_m_in_earlier_constraint.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

interface M {}
final impl forall [T:! M] T as L where .W = () {}

interface Z(Z1:! type) {}

class C(C1:! type);


// Should pass. `.W` can resolve to `()` immediately through the type of `.Self`
// (since the type of `.Self` contains `M`) and the `final impl`. So we can get
// a witness for `C(())` from `T`.
//
// TODO: How come we don't get `.W` evaluating to `()` when `M` comes from an
// earlier constraint? Is the `where_stack` not working with `.Self`?
fn G(T:! L where .Self impls M and C(.W) impls Z(.Self)) {
  // CHECK:STDERR: fail_todo_concrete_access_witness_m_in_earlier_constraint.carbon:[[@LINE+4]]:3: error: cannot convert type `C(())` into type implementing `Z(T)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   C(()) as Z(T);
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  C(()) as Z(T);
}

// --- fail_todo_concrete_access_witness.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

interface M {}
final impl forall [T:! M] T as L where .W = () {}

interface Z(Z1:! type) {}

class C(C1:! type);

// This does not pass because `.W` can't resolve to `()` where it is written.
// The fact that `T impls M` comes later in the facet type. And we don't allow
// `.W` to evaluate to `()` later when identifying the type of `T` to look for a
// witness for `C(()) as Z(T)` in order to avoid cycles.
//
// TODO: We would like this to pass though, without introducing an infinite
// cycle that identifies `T`, which deduces `T:! M` for the `impl`, which
// identifies `T`. See TODO in CollectFacetWitnessSources for ideas.
fn F(T:! L where C(.W) impls Z(.Self) and .Self impls M) {
  // CHECK:STDERR: fail_todo_concrete_access_witness.carbon:[[@LINE+4]]:3: error: cannot convert type `C(())` into type implementing `Z(T)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   C(()) as Z(T);
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  C(()) as Z(T);
}

// CHECK:STDOUT: --- period_self_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.d47: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.335: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.3fe: type = facet_type <@I, @I(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.191: %I.type.3fe = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.b71: type = assoc_entity_type @I, @I(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.d20: %I.assoc_type.b71 = assoc_entity element0, @I.WithSelf.%I1 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.246: type = facet_access_type %.Self.c39 [symbolic_self]
// CHECK:STDOUT:   %I.type.f3d: type = facet_type <@I, @I(%.Self.as_type.246)> [symbolic_self]
// CHECK:STDOUT:   %.Self.0f0: %I.type.f3d = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.24d: %I.type.f3d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.ea7: type = assoc_entity_type @I, @I(%.Self.as_type.246) [symbolic_self]
// CHECK:STDOUT:   %assoc0.c72: %I.assoc_type.ea7 = assoc_entity element0, @I.WithSelf.%I1 [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.a1c: type = facet_access_type %.Self.0f0 [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.67f: <witness> = lookup_impl_witness %.Self.0f0, @I, @I(%.Self.as_type.246) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %I.lookup_impl_witness.67f, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I, @I(%.Self.as_type.246) where %impl.elem0 = %empty_tuple.type> [symbolic_self]
// CHECK:STDOUT:   %pattern_type.021: type = pattern_type %I_where.type [symbolic_self]
// CHECK:STDOUT:   %T.patt.e6f: %pattern_type.021 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.8c8: %I_where.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.8c8 [symbolic]
// CHECK:STDOUT:   %facet_value: %type = facet_value %T.as_type, () [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness.506: <witness> = lookup_impl_witness %T.8c8, @I, @I(%T.as_type) [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.f3d = facet_value %T.as_type, (%I.lookup_impl_witness.506) [symbolic]
// CHECK:STDOUT:   %.262: Core.Form = init_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %I.decl: %I.type.335 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc4_14.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc4_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_7.1: %pattern_type.021 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt.e6f)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %impl.elem0.loc8_39 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %I_where.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T.8c8)]
// CHECK:STDOUT:     %T.as_type.loc8_39.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc8_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc8_39.1: type = converted %T.ref, %T.as_type.loc8_39.2 [symbolic = %T.as_type.loc8_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc8_39.2: %I.assoc_type.ea7 = specific_constant @I1.%assoc0, @I.WithSelf(constants.%.Self.as_type.246, constants.%T.8c8) [symbolic_self = constants.%assoc0.c72]
// CHECK:STDOUT:     %I1.ref.loc8_39: %I.assoc_type.ea7 = name_ref I1, %.loc8_39.2 [symbolic_self = constants.%assoc0.c72]
// CHECK:STDOUT:     %facet_value.loc8_39.2: %type = facet_value constants.%T.as_type, () [symbolic = %facet_value.loc8_39.1 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc8_39.3: %type = converted constants.%T.as_type, %facet_value.loc8_39.2 [symbolic = %facet_value.loc8_39.1 (constants.%facet_value)]
// CHECK:STDOUT:     %T.as_type.loc8_39.3: type = facet_access_type constants.%T.8c8 [symbolic = %T.as_type.loc8_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %facet_value.loc8_39.3: %type = facet_value %T.as_type.loc8_39.3, () [symbolic = %facet_value.loc8_39.1 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc8_39.4: %type = converted constants.%T.8c8, %facet_value.loc8_39.3 [symbolic = %facet_value.loc8_39.1 (constants.%facet_value)]
// CHECK:STDOUT:     %T.as_type.loc8_39.4: type = facet_access_type constants.%T.8c8 [symbolic = %T.as_type.loc8_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %facet_value.loc8_39.4: %type = facet_value %T.as_type.loc8_39.4, () [symbolic = %facet_value.loc8_39.1 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc8_39.5: %type = converted constants.%T.8c8, %facet_value.loc8_39.4 [symbolic = %facet_value.loc8_39.1 (constants.%facet_value)]
// CHECK:STDOUT:     %impl.elem0.loc8_39: type = impl_witness_access constants.%I.lookup_impl_witness.506, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc8_39.6: Core.Form = init_form %impl.elem0.loc8_39 [concrete = constants.%.262]
// CHECK:STDOUT:     %.loc8_19.1: type = splice_block %.loc8_19.2 [symbolic_self = constants.%I_where.type] {
// CHECK:STDOUT:       %.Self.loc8_7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %I.ref: %I.type.335 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %.Self.ref.loc8_12: %type = name_ref .Self, %.Self.loc8_7 [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.Self.as_type.loc8_17: type = facet_access_type %.Self.ref.loc8_12 [symbolic_self = constants.%.Self.as_type.246]
// CHECK:STDOUT:       %.loc8_17: type = converted %.Self.ref.loc8_12, %.Self.as_type.loc8_17 [symbolic_self = constants.%.Self.as_type.246]
// CHECK:STDOUT:       %I.type: type = facet_type <@I, @I(constants.%.Self.as_type.246)> [symbolic_self = constants.%I.type.f3d]
// CHECK:STDOUT:       %.Self.loc8_19: %I.type.f3d = symbolic_binding .Self [symbolic_self = constants.%.Self.0f0]
// CHECK:STDOUT:       %.Self.ref.loc8_25: %I.type.f3d = name_ref .Self, %.Self.loc8_19 [symbolic_self = constants.%.Self.0f0]
// CHECK:STDOUT:       %.Self.as_type.loc8_25: type = facet_access_type %.Self.ref.loc8_25 [symbolic_self = constants.%.Self.as_type.a1c]
// CHECK:STDOUT:       %.loc8_25.1: type = converted %.Self.ref.loc8_25, %.Self.as_type.loc8_25 [symbolic_self = constants.%.Self.as_type.a1c]
// CHECK:STDOUT:       %.loc8_25.2: %I.assoc_type.ea7 = specific_constant @I1.%assoc0, @I.WithSelf(constants.%.Self.as_type.246, constants.%.Self.0f0) [symbolic_self = constants.%assoc0.c72]
// CHECK:STDOUT:       %I1.ref.loc8_25: %I.assoc_type.ea7 = name_ref I1, %.loc8_25.2 [symbolic_self = constants.%assoc0.c72]
// CHECK:STDOUT:       %impl.elem0.loc8_25: type = impl_witness_access constants.%I.lookup_impl_witness.67f, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc8_32.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc8_32.2: type = converted %.loc8_32.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc8_19.2: type = where_expr [symbolic_self = constants.%I_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type %I.type
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc8_25, %.loc8_32.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_7.2: %I_where.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T.8c8)]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc12_7.1: %pattern_type.021 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_7.2 (constants.%T.patt.e6f)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %impl.elem0.loc12_47 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %I_where.type = name_ref T, %T.loc12_7.2 [symbolic = %T.loc12_7.1 (constants.%T.8c8)]
// CHECK:STDOUT:     %T.as_type.loc12_47.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc12_47.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc12_47.1: type = converted %T.ref, %T.as_type.loc12_47.2 [symbolic = %T.as_type.loc12_47.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc12_47.2: %I.assoc_type.ea7 = specific_constant @I1.%assoc0, @I.WithSelf(constants.%.Self.as_type.246, constants.%T.8c8) [symbolic_self = constants.%assoc0.c72]
// CHECK:STDOUT:     %I1.ref.loc12_47: %I.assoc_type.ea7 = name_ref I1, %.loc12_47.2 [symbolic_self = constants.%assoc0.c72]
// CHECK:STDOUT:     %T.as_type.loc12_47.3: type = facet_access_type constants.%T.8c8 [symbolic = %T.as_type.loc12_47.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %facet_value.loc12_47.2: %type = facet_value %T.as_type.loc12_47.3, () [symbolic = %facet_value.loc12_47.1 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc12_47.3: %type = converted constants.%T.8c8, %facet_value.loc12_47.2 [symbolic = %facet_value.loc12_47.1 (constants.%facet_value)]
// CHECK:STDOUT:     %impl.elem0.loc12_47: type = impl_witness_access constants.%I.lookup_impl_witness.506, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc12_47.4: Core.Form = init_form %impl.elem0.loc12_47 [concrete = constants.%.262]
// CHECK:STDOUT:     %.loc12_27.1: type = splice_block %.loc12_27.2 [symbolic_self = constants.%I_where.type] {
// CHECK:STDOUT:       %.Self.loc12_7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %I.ref: %I.type.335 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %.Self.ref.loc12_12: %type = name_ref .Self, %.Self.loc12_7 [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc12_21: type = type_literal type [concrete = type]
// CHECK:STDOUT:       %.Self.as_type.loc12_18: type = facet_access_type %.Self.ref.loc12_12 [symbolic_self = constants.%.Self.as_type.246]
// CHECK:STDOUT:       %.loc12_18: type = converted %.Self.ref.loc12_12, %.Self.as_type.loc12_18 [symbolic_self = constants.%.Self.as_type.246]
// CHECK:STDOUT:       %I.type: type = facet_type <@I, @I(constants.%.Self.as_type.246)> [symbolic_self = constants.%I.type.f3d]
// CHECK:STDOUT:       %.Self.loc12_27: %I.type.f3d = symbolic_binding .Self [symbolic_self = constants.%.Self.0f0]
// CHECK:STDOUT:       %.Self.ref.loc12_33: %I.type.f3d = name_ref .Self, %.Self.loc12_27 [symbolic_self = constants.%.Self.0f0]
// CHECK:STDOUT:       %.Self.as_type.loc12_33: type = facet_access_type %.Self.ref.loc12_33 [symbolic_self = constants.%.Self.as_type.a1c]
// CHECK:STDOUT:       %.loc12_33.1: type = converted %.Self.ref.loc12_33, %.Self.as_type.loc12_33 [symbolic_self = constants.%.Self.as_type.a1c]
// CHECK:STDOUT:       %.loc12_33.2: %I.assoc_type.ea7 = specific_constant @I1.%assoc0, @I.WithSelf(constants.%.Self.as_type.246, constants.%.Self.0f0) [symbolic_self = constants.%assoc0.c72]
// CHECK:STDOUT:       %I1.ref.loc12_33: %I.assoc_type.ea7 = name_ref I1, %.loc12_33.2 [symbolic_self = constants.%assoc0.c72]
// CHECK:STDOUT:       %impl.elem0.loc12_33: type = impl_witness_access constants.%I.lookup_impl_witness.67f, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc12_40.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc12_40.2: type = converted %.loc12_40.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc12_27.2: type = where_expr [symbolic_self = constants.%I_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type %I.type
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc12_33, %.loc12_40.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_7.2: %I_where.type = symbolic_binding T, 0 [symbolic = %T.loc12_7.1 (constants.%T.8c8)]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_14.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc4_14.1)> [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:   %Self.loc4_23.2: @I.%I.type (%I.type.3fe) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self.191)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_23.1: @I.%I.type (%I.type.3fe) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self.191)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %I1: type = assoc_const_decl @I1 [concrete] {
// CHECK:STDOUT:       %assoc0: @I.WithSelf.%I.assoc_type (%I.assoc_type.b71) = assoc_entity element0, @I.WithSelf.%I1 [symbolic = @I.WithSelf.%assoc0 (constants.%assoc0.d20)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_23.1
// CHECK:STDOUT:     .I1 = @I1.%assoc0
// CHECK:STDOUT:     witness = (@I.WithSelf.%I1)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_7.2: %I_where.type) {
// CHECK:STDOUT:   %T.patt.loc8_7.2: %pattern_type.021 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt.e6f)]
// CHECK:STDOUT:   %T.loc8_7.1: %I_where.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T.8c8)]
// CHECK:STDOUT:   %T.as_type.loc8_39.1: type = facet_access_type %T.loc8_7.1 [symbolic = %T.as_type.loc8_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %facet_value.loc8_39.1: %type = facet_value %T.as_type.loc8_39.1, () [symbolic = %facet_value.loc8_39.1 (constants.%facet_value)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc9_11.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc9_11.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc9_12: init %empty_tuple.type = converted %.loc9_11.1, %.loc9_11.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     return %.loc9_12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc12_7.2: %I_where.type) {
// CHECK:STDOUT:   %T.patt.loc12_7.2: %pattern_type.021 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_7.2 (constants.%T.patt.e6f)]
// CHECK:STDOUT:   %T.loc12_7.1: %I_where.type = symbolic_binding T, 0 [symbolic = %T.loc12_7.1 (constants.%T.8c8)]
// CHECK:STDOUT:   %T.as_type.loc12_47.1: type = facet_access_type %T.loc12_7.1 [symbolic = %T.as_type.loc12_47.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %facet_value.loc12_47.1: %type = facet_value %T.as_type.loc12_47.1, () [symbolic = %facet_value.loc12_47.1 (constants.%facet_value)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc13_11.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc13_11.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc13_12: init %empty_tuple.type = converted %.loc13_11.1, %.loc13_11.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     return %.loc13_12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T.67d, constants.%Self.191) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%.Self.as_type.246) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%.Self.as_type.246
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.f3d
// CHECK:STDOUT:   %Self.loc4_23.2 => constants.%Self.24d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.as_type.246, constants.%Self.191) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%.Self.as_type.246
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.ea7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.as_type.246, constants.%.Self.0f0) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%.Self.as_type.246
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.ea7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.as_type.246, constants.%T.8c8) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%.Self.as_type.246
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.ea7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T.as_type) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.as_type.246, constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%.Self.as_type.246
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.ea7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.8c8) {
// CHECK:STDOUT:   %T.patt.loc8_7.2 => constants.%T.patt.e6f
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%T.8c8
// CHECK:STDOUT:   %T.as_type.loc8_39.1 => constants.%T.as_type
// CHECK:STDOUT:   %facet_value.loc8_39.1 => constants.%facet_value
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.8c8) {
// CHECK:STDOUT:   %T.patt.loc12_7.2 => constants.%T.patt.e6f
// CHECK:STDOUT:   %T.loc12_7.1 => constants.%T.8c8
// CHECK:STDOUT:   %T.as_type.loc12_47.1 => constants.%T.as_type
// CHECK:STDOUT:   %facet_value.loc12_47.1 => constants.%facet_value
// CHECK:STDOUT: }
// CHECK:STDOUT:
