// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/access.carbon

// --- access_assoc_fn.carbon

library "[[@TEST_NAME]]";

interface I {
  fn DoIt();
}

fn Use(T:! I) {
//@dump-sem-ir-begin
  T.DoIt();
//@dump-sem-ir-end
}

// --- assoc_fn_using_self.carbon

library "[[@TEST_NAME]]";

interface I {
  fn Make() -> Self;
}

fn Use(T:! I) -> T {
//@dump-sem-ir-begin
  return T.Make();
//@dump-sem-ir-end
}

// --- access_assoc_method.carbon

library "[[@TEST_NAME]]";

interface I {
  fn Copy(self) -> Self;
}

//@dump-sem-ir-begin
fn Use[T:! I](x: T) -> T {
  return x.Copy();
}
//@dump-sem-ir-end

// --- access_selfless_method.carbon

library "[[@TEST_NAME]]";

interface I {
  fn Hello();
}

fn Use[T:! I](x: T){
//@dump-sem-ir-begin
  x.Hello();
//@dump-sem-ir-end
}

// --- access_assoc_method_indirect.carbon

library "[[@TEST_NAME]]";

interface I {
  fn Copy(self) -> Self;
}

fn UseIndirect[T:! I](x: T) -> T {
//@dump-sem-ir-begin
  return x.(T.Copy)();
//@dump-sem-ir-end
}

// --- convert_from_period_self_to_full_facet_value.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1:! type;
}

fn F(U:! I where .I1 = .Self) {
  U as (I where .I1 = U);
  (U as type) as (I where .I1 = U);
}

// --- convert_to_period_self.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1:! type;
  let I2:! type;
}

fn F(U:! I where .I1 = .Self and .I2 = ()) {
  U as (I where .I1 = .Self);
  (U as type) as (I where .I1 = .Self);
}

// --- access_through_call.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  fn G() -> X*;
}

fn F2[U:! I](unused V: U*) {}
fn F3[U:! I where .X = .Self](unused V: U*) {}

fn F(U:! I where .X = .Self, unused V: U*) {
  // The returned value of `G` type `U` which has access to the methods of `I`.
  U.G()->G()->G();
  (U as type).G()->G()->G();

  // The returned value of type `U` can be used as a value of type `U`.
  F2(U.G()->G()->G());

  // The constraints in the type `U` are preserved.
  F3(U.G());
  F3(U.G()->G()->G());
}

// --- fail_compound_access_through_call.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  fn G() -> X;
}

fn F(U:! Core.Destroy & I where .X = .Self) {
  // Compound member lookup through a non-type value is possible for methods
  // which take a `self` parameter. But it's not possible for methods without
  // `self`. For those you need to go directly throug the type.
  // See: https://github.com/carbon-language/carbon-lang/issues/6025

  let u: U = U.(I.G)();
  // `u` is a non-type value. Can call methods with `self` through compound
  // member lookup, but can't call methods without `self`. See the
  // `compound_access_through_call_with_self_param.carbon` test for the former.
  //
  // CHECK:STDERR: fail_compound_access_through_call.carbon:[[@LINE+7]]:3: error: cannot implicitly convert non-type value of type `U` into type implementing `I` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   u.(I.G)();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_compound_access_through_call.carbon:[[@LINE+4]]:3: note: type `U` does not implement interface `Core.ImplicitAs(I)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   u.(I.G)();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  u.(I.G)();

  // This is the same as the above, since G() returns a non-type value of type
  // `U`.
  //
  // CHECK:STDERR: fail_compound_access_through_call.carbon:[[@LINE+7]]:3: error: cannot implicitly convert non-type value of type `U` into type implementing `I` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   U.(I.G)().(I.G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_compound_access_through_call.carbon:[[@LINE+4]]:3: note: type `U` does not implement interface `Core.ImplicitAs(I)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   U.(I.G)().(I.G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  U.(I.G)().(I.G)();
}

// --- compound_access_through_call_with_self_param.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  fn G(self) -> X*;
}

fn F(U:! I where .X = .Self, v: U) {
  // Compound member lookup through a non-type value is possible for methods
  // which take a `self` parameter.

  let u: U* = v.(I.G)();
  // `u` is a non-type value. Can call methods with `self` through compound
  // member lookup, but can't call methods without `self`. See the
  // `compound_access_through_call.carbon` test for the latter.
  u->(I.G)();

  // This is the same as the above, since G() returns a non-type value of type
  // `U`. This works because G has a `self` parameter.
  v.(I.G)()->(I.G)();
}

// --- fail_non_const_associated.carbon
library "[[@TEST_NAME]]";

interface I { let T:! type; }
fn Id[U:! type](x: U) -> U { return Id(x); }
impl () as I where .T = () {}
// Type of member expr is associated entity type,
// but value is not constant.
// CHECK:STDERR: fail_non_const_associated.carbon:[[@LINE+4]]:8: error: semantics TODO: `Non-constant associated entity value` [SemanticsTodo]
// CHECK:STDERR: var v: ().(Id(I.T));
// CHECK:STDERR:        ^~~~~~~~~~~~
// CHECK:STDERR:
var v: ().(Id(I.T));

// --- fail_non_const_associated_in_interface.carbon
library "[[@TEST_NAME]]";

fn Id[U:! type](x: U) -> U { return Id(x); }

interface J {
  let T:! type;
  // CHECK:STDERR: fail_non_const_associated_in_interface.carbon:[[@LINE+4]]:13: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   fn F() -> Id(T);
  // CHECK:STDERR:             ^~~~~
  // CHECK:STDERR:
  fn F() -> Id(T);
}

// --- fail_alias_to_non_const_assoc_entity.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
}

// CHECK:STDERR: fail_alias_to_non_const_assoc_entity.carbon:[[@LINE+4]]:8: error: semantics TODO: `HandleAutoTypeLiteral` [SemanticsTodo]
// CHECK:STDERR: let x: auto = I.T;
// CHECK:STDERR:        ^~~~
// CHECK:STDERR:
let x: auto = I.T;

interface J {
  // Is this valid?
  alias U = x;
  // type of U is an assoc entity type, but value is not constant.
  fn F() -> U;
}

// --- to_import.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
}

alias U = I.T;

// --- fail_access_alias_in_imported_library.carbon
library "[[@TEST_NAME]]";

import library "to_import";

interface J {
  // extend I;
  alias V = U;
  // CHECK:STDERR: fail_access_alias_in_imported_library.carbon:[[@LINE+4]]:13: error: cannot convert type `Self` that implements `J` into type implementing `I` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   fn F() -> V;
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  fn F() -> V;
}

// --- access_constant_in_self_facet.carbon
library "[[@TEST_NAME]]";

interface A { let X:! type; }

fn F(AA:! A where .X = ()) -> AA.X {
  return ();
}

// --- access_constant_in_self_facet_with_multiple_interfaces.carbon
library "[[@TEST_NAME]]";

interface A { let X:! type; }
interface B { let Y:! type; }

// The rewrite rules of .X and .Y come in some canonically sorted order. The
// ImplWitnessAccess instruction looks at them to try find a value for the
// rewrite in the conversion target. We use different orderings to more reliably
// create the scenario where the ImplWitnessAccess sees a rewrite of a value in
// an interface other than the one it is accessing before finding the correct
// rewrite.

fn F(AB:! A & B where .X = () and .Y = {}) -> AB.X {
  return ();
}

fn G(AB:! A & B where .X = () and .Y = {}) -> AB.Y {
  return {};
}

// --- fail_todo_access_constant_through_named_constraint.carbon
library "[[@TEST_NAME]]";

interface A { let X:! type; }

constraint N {
  extend require impls A where .X = ();
}

fn F(AA:! N) {
  // TODO: The identified facet type `N` includes `A` and should provide a
  // same-type constraint `A.X == ()` which would allow this conversion.
  // CHECK:STDERR: fail_todo_access_constant_through_named_constraint.carbon:[[@LINE+7]]:3: error: cannot convert expression of type `()` to `AA.(A.X)` with `as` [ConversionFailure]
  // CHECK:STDERR:   () as AA.X;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_access_constant_through_named_constraint.carbon:[[@LINE+4]]:3: note: type `()` does not implement interface `Core.As(AA.(A.X))` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   () as AA.X;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  () as AA.X;
}

// --- symbolic_binding_type_of_impl_witness_access.carbon

interface Y {}
impl () as Y {}

interface Z {
  let Y1:! Y;
  fn G() -> Y1*;
}

// The type of `T` matches exactly the type of `Z.Y1`, which prevents the
// specific in the call from F2 from deducing a `FacetValue`. Instead it just
// passes in the `ImplWitnessAccess` instruction as is.
//
// The `t: T` creates a `T as type`, or a `SymbolicBindingType` that gets
// evaluated against the specific. The facet value that it evaluates against is
// the `ImplWitnessAccess` from the call in F2.
//
// This requires that `SymbolicBindingType` evaluation correctly handles
// arbitrary facet value instructions. Not just the common case of `FacetValue`
// or `SymbolicBinding`.
fn F1(T:! Y, unused t: T*) {}

fn F2(U:! Z) {
  F1(U.Y1, U.G());
}

// --- find_access_value_in_first_nested_access.carbon
library "[[@TEST_NAME]]";

interface W {}
interface X {
  let X1:! type;
}
interface Y(U:! type) {
  let Y1:! X where .X1 = U;
}
interface Z(T:! type) {
  let Z1:! type;
}

fn F(U:! W, V:! Z(.Self) where .Z1 impls Y(U)) {
  // This has to search for a value for `.Z1.Y1.X1` in V. To do so it needs to
  // look for a rewrite of .X1. First it looks in `V.Z1.Y1`, where
  // it finds the rewrite of `.X1 = U`.
  //
  // Only U impls W so we use `as W` to test that the LHS is U.
  V.(Z(V).Z1).(Y(U).Y1).(X.X1) as W;
}

// --- fail_todo_find_access_value_in_second_nested_access.carbon
library "[[@TEST_NAME]]";

interface W {}
interface X {
  let X1:! type;
}
interface Y(T:! type) {
  let Y0:! type;
  let Y1:! type;
  let Y2:! type;
}

constraint NX(V:! type) {
  extend require impls X where .X1 = V;
}

interface Z(PeriodSelf:! type) {
  let Z1:! Y(PeriodSelf) where .Y0 impls NX(())
                           and .Y1 impls NX({})
                           and .Y2 impls NX(());
}

fn F(V:! Z(.Self)) {
  // This has to search for a value for `.Z1.Y1.X1` in V. To do so it needs to
  // look for a rewrite of .X1. First it looks in `V.Z1.Y1`, then `V.Z1`, where
  // it finds the rewrite of `.Y1.X1 = {}`.
  //
  // TODO: The identified facet type `NX` includes `X` and should provide a
  // same-type constraint `X.X1 == {}` which would allow this conversion.
  // CHECK:STDERR: fail_todo_find_access_value_in_second_nested_access.carbon:[[@LINE+7]]:3: error: cannot convert expression of type `{}` to `V.(Z(V).Z1).(Y(V).Y1).(X.X1)` with `as` [ConversionFailure]
  // CHECK:STDERR:   {} as V.(Z(V).Z1).(Y(V).Y1).(X.X1);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_find_access_value_in_second_nested_access.carbon:[[@LINE+4]]:3: note: type `{}` does not implement interface `Core.As(V.(Z(V).Z1).(Y(V).Y1).(X.X1))` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   {} as V.(Z(V).Z1).(Y(V).Y1).(X.X1);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  {} as V.(Z(V).Z1).(Y(V).Y1).(X.X1);
}

// --- fail_todo_find_access_value_in_third_nested_access.carbon
library "[[@TEST_NAME]]";

interface W {}
interface X {
  let X1:! type;
}
interface Y {
  let Y1:! type;
}
interface Z(T:! type) {
  let Z0:! type;
  let Z1:! type;
  let Z2:! type;
}

constraint NX(V:! type) {
  extend require impls X where .X1 = V;
}

constraint NY(V:! type) {
  extend require impls Y where .Y1 = NX(V);
}

fn F(V:! Z(.Self) where .Z0 impls NY(())
                    and .Z1 impls NY({})
                    and .Z2 impls NY(())) {
  // This has to search for a value for `.Z1.Y1.X1` in V. To do so it needs to
  // look for a rewrite of .X1. First it looks in `V.Z1.Y1`, then `V.Z1`, then
  // `V` where it finds the rewrite of `.Z1.Y1.X1 = {}`.
  //
  // TODO: The identified facet type `NX` includes `X` and should provide a
  // same-type constraint `X.X1 == V` which would allow this conversion.
  // CHECK:STDERR: fail_todo_find_access_value_in_third_nested_access.carbon:[[@LINE+4]]:9: error: cannot convert type `V.(Z(V).Z1).(Y.Y1)` into type implementing `X` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   {} as V.(Z(V).Z1).(Y.Y1).(X.X1);
  // CHECK:STDERR:         ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  {} as V.(Z(V).Z1).(Y.Y1).(X.X1);
}

// --- fail_todo_find_access_value_in_facet_from_specific_interface.carbon
library "[[@TEST_NAME]]";

interface Z {}
interface Y(T:! type) {
  let Y1:! type;
}
class C;

constraint NY(PeriodSelf:! type) {
  extend require impls Y(PeriodSelf) where .Y1 = {};
}

// TODO: We need to look in `T` for the rewrite in the access of
// `(C as Y(T)).Y1` but we currently only look in `C`. See the TODO in
// `TryFindValueInRewriteConstraints()`.
fn F(T:! Z where C impls NY(.Self)) -> C.(Y(T).Y1) {
  // CHECK:STDERR: fail_todo_find_access_value_in_facet_from_specific_interface.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `{}` to `C.(Y(T).Y1)` [ConversionFailure]
  // CHECK:STDERR:   return {};
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_find_access_value_in_facet_from_specific_interface.carbon:[[@LINE+4]]:3: note: type `{}` does not implement interface `Core.ImplicitAs(C.(Y(T).Y1))` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return {};
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  return {};
}

// CHECK:STDOUT: --- access_assoc_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.DoIt.decl [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.DoIt.type.0a5: type = fn_type @I.WithSelf.DoIt, @I.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.9ca: type = fn_type_with_self_type %I.WithSelf.DoIt.type.0a5, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.9ca = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.DoIt(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_9.2: %I.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc10_4.2: type = facet_access_type %T.loc8_9.1 [symbolic = %T.as_type.loc10_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %I.WithSelf.DoIt.type: type = fn_type @I.WithSelf.DoIt, @I.WithSelf(%T.loc8_9.1) [symbolic = %I.WithSelf.DoIt.type (constants.%I.WithSelf.DoIt.type.0a5)]
// CHECK:STDOUT:   %.loc10_4.2: type = fn_type_with_self_type %I.WithSelf.DoIt.type, %T.loc8_9.1 [symbolic = %.loc10_4.2 (constants.%.9ca)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_9.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc10_4.2: @Use.%.loc10_4.2 (%.9ca) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_4.2: <specific function> = specific_impl_function %impl.elem0.loc10_4.2, @I.WithSelf.DoIt(%T.loc8_9.1) [symbolic = %specific_impl_fn.loc10_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc8_9.2 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc10_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc10_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_4.1: type = converted %T.ref, %T.as_type.loc10_4.1 [symbolic = %T.as_type.loc10_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %DoIt.ref: %I.assoc_type = name_ref DoIt, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc10_4.1: @Use.%.loc10_4.2 (%.9ca) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_4.1: <specific function> = specific_impl_function %impl.elem0.loc10_4.1, @I.WithSelf.DoIt(constants.%T) [symbolic = %specific_impl_fn.loc10_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %I.WithSelf.DoIt.call: init %empty_tuple.type = call %specific_impl_fn.loc10_4.1()
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc8_9.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- assoc_fn_using_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.Make.decl [concrete]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %.0ce: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.b3a: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %return.param_patt.b82: %pattern_type.b3a = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.2e6: %pattern_type.b3a = return_slot_pattern %return.param_patt.b82, %T.as_type [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Make.type.9c0: type = fn_type @I.WithSelf.Make, @I.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.889: type = fn_type_with_self_type %I.WithSelf.Make.type.9c0, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.889 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.Make(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_9.2: %I.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.WithSelf.Make.type: type = fn_type @I.WithSelf.Make, @I.WithSelf(%T.loc8_9.1) [symbolic = %I.WithSelf.Make.type (constants.%I.WithSelf.Make.type.9c0)]
// CHECK:STDOUT:   %.loc10_11.2: type = fn_type_with_self_type %I.WithSelf.Make.type, %T.loc8_9.1 [symbolic = %.loc10_11.2 (constants.%.889)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_9.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc10_11.2: @Use.%.loc10_11.2 (%.889) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_11.2: <specific function> = specific_impl_function %impl.elem0.loc10_11.2, @I.WithSelf.Make(%T.loc8_9.1) [symbolic = %specific_impl_fn.loc10_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Use.%T.as_type.loc8_18.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc10: %I.type = name_ref T, %T.loc8_9.2 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc10: type = facet_access_type %T.ref.loc10 [symbolic = %T.as_type.loc8_18.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_11.1: type = converted %T.ref.loc10, %T.as_type.loc10 [symbolic = %T.as_type.loc8_18.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %Make.ref: %I.assoc_type = name_ref Make, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc10_11.1: @Use.%.loc10_11.2 (%.889) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_11.1: <specific function> = specific_impl_function %impl.elem0.loc10_11.1, @I.WithSelf.Make(constants.%T) [symbolic = %specific_impl_fn.loc10_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %I.WithSelf.Make.call: init @Use.%T.as_type.loc8_18.1 (%T.as_type) to %.loc8_18.1 = call %specific_impl_fn.loc10_11.1()
// CHECK:STDOUT:     return %I.WithSelf.Make.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc8_9.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_9.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc8_18.1 => constants.%T.as_type
// CHECK:STDOUT:   %.loc8_18.2 => constants.%.0ce
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b3a
// CHECK:STDOUT:   %return.param_patt.loc8_18.2 => constants.%return.param_patt.b82
// CHECK:STDOUT:   %return.patt.loc8_15.2 => constants.%return.patt.2e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_assoc_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.Copy.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.b3a: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.b3a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt: %pattern_type.b3a = at_binding_pattern x, %x.param_patt [symbolic]
// CHECK:STDOUT:   %.0ce: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %return.param_patt.b82: %pattern_type.b3a = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.2e6: %pattern_type.b3a = return_slot_pattern %return.param_patt.b82, %T.as_type [symbolic]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Copy.type.54f: type = fn_type @I.WithSelf.Copy, @I.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.c06: type = fn_type_with_self_type %I.WithSelf.Copy.type.54f, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.c06 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.Copy(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {
// CHECK:STDOUT:     %T.patt.loc9_9.1: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc9_16.1: @Use.%pattern_type (%pattern_type.b3a) = value_param_pattern [symbolic = %x.param_patt.loc9_16.2 (constants.%x.param_patt)]
// CHECK:STDOUT:     %x.patt.loc9_16.1: @Use.%pattern_type (%pattern_type.b3a) = at_binding_pattern x, %x.param_patt.loc9_16.1 [symbolic = %x.patt.loc9_16.2 (constants.%x.patt)]
// CHECK:STDOUT:     %return.param_patt.loc9_24.1: @Use.%pattern_type (%pattern_type.b3a) = out_param_pattern [symbolic = %return.param_patt.loc9_24.2 (constants.%return.param_patt.b82)]
// CHECK:STDOUT:     %return.patt.loc9_21.1: @Use.%pattern_type (%pattern_type.b3a) = return_slot_pattern %return.param_patt.loc9_24.1, %.loc9_24.3 [symbolic = %return.patt.loc9_21.2 (constants.%return.patt.2e6)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_24: %I.type = name_ref T, %T.loc9_9.2 [symbolic = %T.loc9_9.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc9_24: type = facet_access_type %T.ref.loc9_24 [symbolic = %T.as_type.loc9_18.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_24.3: type = converted %T.ref.loc9_24, %T.as_type.loc9_24 [symbolic = %T.as_type.loc9_18.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_24.4: Core.Form = init_form %.loc9_24.3 [symbolic = %.loc9_24.2 (constants.%.0ce)]
// CHECK:STDOUT:     %.loc9_12: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_9.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc9_9.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Use.%T.as_type.loc9_18.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_18.1: type = splice_block %.loc9_18.2 [symbolic = %T.as_type.loc9_18.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc9_18: %I.type = name_ref T, %T.loc9_9.2 [symbolic = %T.loc9_9.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc9_18.2: type = facet_access_type %T.ref.loc9_18 [symbolic = %T.as_type.loc9_18.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_18.2: type = converted %T.ref.loc9_18, %T.as_type.loc9_18.2 [symbolic = %T.as_type.loc9_18.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Use.%T.as_type.loc9_18.1 (%T.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @Use.%T.as_type.loc9_18.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Use.%T.as_type.loc9_18.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc9_9.2: %I.type) {
// CHECK:STDOUT:   %T.patt.loc9_9.2: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc9_9.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc9_9.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc9_18.1: type = facet_access_type %T.loc9_9.1 [symbolic = %T.as_type.loc9_18.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc9_18.1 [symbolic = %pattern_type (constants.%pattern_type.b3a)]
// CHECK:STDOUT:   %x.param_patt.loc9_16.2: @Use.%pattern_type (%pattern_type.b3a) = value_param_pattern [symbolic = %x.param_patt.loc9_16.2 (constants.%x.param_patt)]
// CHECK:STDOUT:   %x.patt.loc9_16.2: @Use.%pattern_type (%pattern_type.b3a) = at_binding_pattern x, %x.param_patt.loc9_16.2 [symbolic = %x.patt.loc9_16.2 (constants.%x.patt)]
// CHECK:STDOUT:   %.loc9_24.2: Core.Form = init_form %T.as_type.loc9_18.1 [symbolic = %.loc9_24.2 (constants.%.0ce)]
// CHECK:STDOUT:   %return.param_patt.loc9_24.2: @Use.%pattern_type (%pattern_type.b3a) = out_param_pattern [symbolic = %return.param_patt.loc9_24.2 (constants.%return.param_patt.b82)]
// CHECK:STDOUT:   %return.patt.loc9_21.2: @Use.%pattern_type (%pattern_type.b3a) = return_slot_pattern %return.param_patt.loc9_24.2, %T.as_type.loc9_18.1 [symbolic = %return.patt.loc9_21.2 (constants.%return.patt.2e6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc9_18.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %I.WithSelf.Copy.type: type = fn_type @I.WithSelf.Copy, @I.WithSelf(%T.loc9_9.1) [symbolic = %I.WithSelf.Copy.type (constants.%I.WithSelf.Copy.type.54f)]
// CHECK:STDOUT:   %.loc10_11: type = fn_type_with_self_type %I.WithSelf.Copy.type, %T.loc9_9.1 [symbolic = %.loc10_11 (constants.%.c06)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc9_9.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc10_11.2: @Use.%.loc10_11 (%.c06) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_11.2: <specific function> = specific_impl_function %impl.elem0.loc10_11.2, @I.WithSelf.Copy(%T.loc9_9.1) [symbolic = %specific_impl_fn.loc10_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Use.%T.as_type.loc9_18.1 (%T.as_type)) -> out %return.param: @Use.%T.as_type.loc9_18.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Use.%T.as_type.loc9_18.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Copy.ref: %I.assoc_type = name_ref Copy, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc10_11.1: @Use.%.loc10_11 (%.c06) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc10_11: <bound method> = bound_method %x.ref, %impl.elem0.loc10_11.1
// CHECK:STDOUT:     %.loc10_17.1: %I.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc9_9.1 (constants.%T)]
// CHECK:STDOUT:     %.loc10_17.2: %I.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc9_9.1 (constants.%T)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_11.1: <specific function> = specific_impl_function %impl.elem0.loc10_11.1, @I.WithSelf.Copy(constants.%T) [symbolic = %specific_impl_fn.loc10_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc10_17: <bound method> = bound_method %x.ref, %specific_impl_fn.loc10_11.1
// CHECK:STDOUT:     %.loc9_24.1: ref @Use.%T.as_type.loc9_18.1 (%T.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %I.WithSelf.Copy.call: init @Use.%T.as_type.loc9_18.1 (%T.as_type) to %.loc9_24.1 = call %bound_method.loc10_17(%x.ref)
// CHECK:STDOUT:     return %I.WithSelf.Copy.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc9_9.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc9_9.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc9_18.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b3a
// CHECK:STDOUT:   %x.param_patt.loc9_16.2 => constants.%x.param_patt
// CHECK:STDOUT:   %x.patt.loc9_16.2 => constants.%x.patt
// CHECK:STDOUT:   %.loc9_24.2 => constants.%.0ce
// CHECK:STDOUT:   %return.param_patt.loc9_24.2 => constants.%return.param_patt.b82
// CHECK:STDOUT:   %return.patt.loc9_21.2 => constants.%return.patt.2e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_selfless_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.Hello.decl [concrete]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.b3a: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.b3a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt: %pattern_type.b3a = at_binding_pattern x, %x.param_patt [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Hello.type.d5c: type = fn_type @I.WithSelf.Hello, @I.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.244: type = fn_type_with_self_type %I.WithSelf.Hello.type.d5c, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.244 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.Hello(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_9.2: %I.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.WithSelf.Hello.type: type = fn_type @I.WithSelf.Hello, @I.WithSelf(%T.loc8_9.1) [symbolic = %I.WithSelf.Hello.type (constants.%I.WithSelf.Hello.type.d5c)]
// CHECK:STDOUT:   %.loc10: type = fn_type_with_self_type %I.WithSelf.Hello.type, %T.loc8_9.1 [symbolic = %.loc10 (constants.%.244)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_9.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc10_4.2: @Use.%.loc10 (%.244) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_4.2: <specific function> = specific_impl_function %impl.elem0.loc10_4.2, @I.WithSelf.Hello(%T.loc8_9.1) [symbolic = %specific_impl_fn.loc10_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Use.%T.as_type.loc8_18.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Use.%T.as_type.loc8_18.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Hello.ref: %I.assoc_type = name_ref Hello, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc10_4.1: @Use.%.loc10 (%.244) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_4.1: <specific function> = specific_impl_function %impl.elem0.loc10_4.1, @I.WithSelf.Hello(constants.%T) [symbolic = %specific_impl_fn.loc10_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %I.WithSelf.Hello.call: init %empty_tuple.type = call %specific_impl_fn.loc10_4.1()
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc8_9.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_9.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc8_18.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b3a
// CHECK:STDOUT:   %x.param_patt.loc8_16.2 => constants.%x.param_patt
// CHECK:STDOUT:   %x.patt.loc8_16.2 => constants.%x.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_assoc_method_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.Copy.decl [concrete]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.b3a: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.b3a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt: %pattern_type.b3a = at_binding_pattern x, %x.param_patt [symbolic]
// CHECK:STDOUT:   %.0ce: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %return.param_patt.b82: %pattern_type.b3a = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.2e6: %pattern_type.b3a = return_slot_pattern %return.param_patt.b82, %T.as_type [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Copy.type.54f: type = fn_type @I.WithSelf.Copy, @I.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.c06: type = fn_type_with_self_type %I.WithSelf.Copy.type.54f, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.c06 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.Copy(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @UseIndirect(%T.loc8_17.2: %I.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.WithSelf.Copy.type: type = fn_type @I.WithSelf.Copy, @I.WithSelf(%T.loc8_17.1) [symbolic = %I.WithSelf.Copy.type (constants.%I.WithSelf.Copy.type.54f)]
// CHECK:STDOUT:   %.loc10_14.2: type = fn_type_with_self_type %I.WithSelf.Copy.type, %T.loc8_17.1 [symbolic = %.loc10_14.2 (constants.%.c06)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_17.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc10_14.2: @UseIndirect.%.loc10_14.2 (%.c06) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_14.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_14.2: <specific function> = specific_impl_function %impl.elem0.loc10_14.2, @I.WithSelf.Copy(%T.loc8_17.1) [symbolic = %specific_impl_fn.loc10_14.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @UseIndirect.%T.as_type.loc8_26.1 (%T.as_type)) -> out %return.param: @UseIndirect.%T.as_type.loc8_26.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @UseIndirect.%T.as_type.loc8_26.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %T.ref.loc10: %I.type = name_ref T, %T.loc8_17.2 [symbolic = %T.loc8_17.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc10: type = facet_access_type %T.ref.loc10 [symbolic = %T.as_type.loc8_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_14.1: type = converted %T.ref.loc10, %T.as_type.loc10 [symbolic = %T.as_type.loc8_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %Copy.ref: %I.assoc_type = name_ref Copy, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc10_14.1: @UseIndirect.%.loc10_14.2 (%.c06) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_14.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc10_11: <bound method> = bound_method %x.ref, %impl.elem0.loc10_14.1
// CHECK:STDOUT:     %.loc10_21.1: %I.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc8_17.1 (constants.%T)]
// CHECK:STDOUT:     %.loc10_21.2: %I.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc8_17.1 (constants.%T)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_14.1: <specific function> = specific_impl_function %impl.elem0.loc10_14.1, @I.WithSelf.Copy(constants.%T) [symbolic = %specific_impl_fn.loc10_14.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc10_21: <bound method> = bound_method %x.ref, %specific_impl_fn.loc10_14.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %I.WithSelf.Copy.call: init @UseIndirect.%T.as_type.loc8_26.1 (%T.as_type) to %.loc8_32.1 = call %bound_method.loc10_21(%x.ref)
// CHECK:STDOUT:     return %I.WithSelf.Copy.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseIndirect(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc8_17.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_17.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc8_26.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b3a
// CHECK:STDOUT:   %x.param_patt.loc8_24.2 => constants.%x.param_patt
// CHECK:STDOUT:   %x.patt.loc8_24.2 => constants.%x.patt
// CHECK:STDOUT:   %.loc8_32.2 => constants.%.0ce
// CHECK:STDOUT:   %return.param_patt.loc8_32.2 => constants.%return.param_patt.b82
// CHECK:STDOUT:   %return.patt.loc8_29.2 => constants.%return.patt.2e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
