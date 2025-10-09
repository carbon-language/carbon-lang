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
  fn Copy[self: Self]() -> Self;
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
  fn Copy[self: Self]() -> Self;
}

fn UseIndirect[T:! I](x: T) -> T {
//@dump-sem-ir-begin
  return x.(T.Copy)();
//@dump-sem-ir-end
}

// --- fail_todo_convert_from_period_self_to_full_facet_value.carbon
library "[[@TEST_NAME]]";

interface I {
  let I1:! type;
}

fn F(U:! I where .I1 = .Self) {
  // CHECK:STDERR: fail_todo_convert_from_period_self_to_full_facet_value.carbon:[[@LINE+4]]:3: error: cannot convert type `U` that implements `I where .(I.I1) = .Self` into type implementing `I where .(I.I1) = U` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   U as (I where .I1 = U);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  U as (I where .I1 = U);
  // CHECK:STDERR: fail_todo_convert_from_period_self_to_full_facet_value.carbon:[[@LINE+4]]:3: error: cannot convert type `U` that implements `I where .(I.I1) = .Self` into type implementing `I where .(I.I1) = U` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   (U as type) as (I where .I1 = U);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
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

// --- access_through_call_once.carbon
library "[[@TEST_NAME]]";

// TODO: Merge this test with the one below once it works.

interface I {
  let X:! type;
  fn G() -> X*;
}

fn F2[U:! I](V: U*) {}

fn F(U:! I where .X = .Self, V: U) {
  // The returned value of `G` type `U` which has access to the methods of `I`.
  U.G()->G();
  (U as type).G()->G();

  // The returned value of type `U` can be used as a value of type `U`.
  F2(U.G());
}

// --- fail_todo_access_through_call.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  fn G() -> X*;
}

fn F2[U:! I](V: U*) {}
fn F3[U:! I where .X = .Self](V: U*) {}

fn F(U:! I where .X = .Self, V: U*) {
  // The returned value of `G` type `U` which has access to the methods of `I`.
  //
  // TODO: These should work.
  // - The first `.` is on a NameRef of type FacetType for `I where .X = .Self`.
  //   - This finds `G` through the FacetType.
  // - The second `.` is on a Call of type FacetAccessType into `BindSymbolicName` with type FacetType for `I`.
  //   - This finds `G` through the FacetType (impl lookup strips off FacetAccessType).
  // - The third `.` is on a Call of type FacetAccessType into `ImplWitnessAccess` of `I.X` into `LookupImplWitness`, which has type `type`
  //   - Can't make calls on an `ImplWitnessAccess`.
  //   - We could expect that the constant value of the `ImplWitnessAccess` would
  //     be the same type that we got for the second lookup.
  // CHECK:STDERR: fail_todo_access_through_call.carbon:[[@LINE+4]]:3: error: type `.(I.X)` does not support qualified expressions [QualifiedExprUnsupported]
  // CHECK:STDERR:   U.G()->G()->G();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  U.G()->G()->G();
  // CHECK:STDERR: fail_todo_access_through_call.carbon:[[@LINE+4]]:3: error: type `.(I.X)` does not support qualified expressions [QualifiedExprUnsupported]
  // CHECK:STDERR:   (U as type).G()->G()->G();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (U as type).G()->G()->G();

  // The returned value of type `U` can be used as a value of type `U`.
  //
  // TODO: This should work.
  // CHECK:STDERR: fail_todo_access_through_call.carbon:[[@LINE+4]]:6: error: type `.(I.X)` does not support qualified expressions [QualifiedExprUnsupported]
  // CHECK:STDERR:   F2(U.G()->G()->G());
  // CHECK:STDERR:      ^~~~~~~~~~~~~
  // CHECK:STDERR:
  F2(U.G()->G()->G());

  // The constraints in the type `U` are preserved.
  //
  // TODO: These should work.
  // CHECK:STDERR: fail_todo_access_through_call.carbon:[[@LINE+7]]:3: error: cannot convert type `.Self` that implements `I` into type implementing `I where .(I.X) = .Self` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F3(U.G());
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_access_through_call.carbon:[[@LINE-40]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F3[U:! I where .X = .Self](V: U*) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F3(U.G());
  // CHECK:STDERR: fail_todo_access_through_call.carbon:[[@LINE+4]]:6: error: type `.(I.X)` does not support qualified expressions [QualifiedExprUnsupported]
  // CHECK:STDERR:   F3(U.G()->G()->G());
  // CHECK:STDERR:      ^~~~~~~~~~~~~
  // CHECK:STDERR:
  F3(U.G()->G()->G());
}

// --- fail_todo_compound_access_through_call.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  fn G() -> X;
}

fn F(U:! I where .X = .Self) {
  // Compound member lookup through a non-type value is possible for methods
  // which take a `self` parameter. But it's not possible for methods without
  // `self`. For those you need to go directly throug the type.
  // See: https://github.com/carbon-language/carbon-lang/issues/6025

  // TODO: This step should work.
  //
  // CHECK:STDERR: fail_todo_compound_access_through_call.carbon:[[@LINE+7]]:14: error: cannot implicitly convert expression of type `.Self` to `U` [ConversionFailure]
  // CHECK:STDERR:   let u: U = U.(I.G)();
  // CHECK:STDERR:              ^~~~~~~~~
  // CHECK:STDERR: fail_todo_compound_access_through_call.carbon:[[@LINE+4]]:14: note: type `.Self` does not implement interface `Core.ImplicitAs(U)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let u: U = U.(I.G)();
  // CHECK:STDERR:              ^~~~~~~~~
  // CHECK:STDERR:
  let u: U = U.(I.G)();
  // `u` is a non-type value. Can call methods with `self` through compound
  // member lookup, but can't call methods without `self`. See the
  // `compound_access_through_call_with_self_param.carbon` test for the former.
  //
  // CHECK:STDERR: fail_todo_compound_access_through_call.carbon:[[@LINE+7]]:3: error: cannot implicitly convert non-type value of type `U` into type implementing `I` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   u.(I.G)();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_todo_compound_access_through_call.carbon:[[@LINE+4]]:3: note: type `U` does not implement interface `Core.ImplicitAs(I)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   u.(I.G)();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  u.(I.G)();

  // This is the same as the above, since G() returns a non-type value of type
  // `U`.
  //
  // CHECK:STDERR: fail_todo_compound_access_through_call.carbon:[[@LINE+7]]:3: error: cannot implicitly convert non-type value of type `.Self` into type implementing `I` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   U.(I.G)().(I.G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_compound_access_through_call.carbon:[[@LINE+4]]:3: note: type `.Self` does not implement interface `Core.ImplicitAs(I)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   U.(I.G)().(I.G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  U.(I.G)().(I.G)();
}

// --- fail_todo_compound_access_through_call_with_self_param.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  fn G[self: Self]() -> X*;
}

fn F(U:! I where .X = .Self, v: U) {
  // Compound member lookup through a non-type value is possible for methods
  // which take a `self` parameter.

  // TODO: This should all work.

  // CHECK:STDERR: fail_todo_compound_access_through_call_with_self_param.carbon:[[@LINE+7]]:15: error: cannot implicitly convert expression of type `.Self*` to `U*` [ConversionFailure]
  // CHECK:STDERR:   let u: U* = v.(I.G)();
  // CHECK:STDERR:               ^~~~~~~~~
  // CHECK:STDERR: fail_todo_compound_access_through_call_with_self_param.carbon:[[@LINE+4]]:15: note: type `.Self*` does not implement interface `Core.ImplicitAs(U*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let u: U* = v.(I.G)();
  // CHECK:STDERR:               ^~~~~~~~~
  // CHECK:STDERR:
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

//@dump-sem-ir-begin
fn F(AA:! A where .X = ()) -> AA.X {
  return ();
}
//@dump-sem-ir-end

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

//@dump-sem-ir-begin
fn F(AB:! A & B where .X = () and .Y = {}) -> AB.X {
  return ();
}

fn G(AB:! A & B where .X = () and .Y = {}) -> AB.Y {
  return {};
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- access_assoc_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.DoIt.type: type = fn_type @I.DoIt [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%I.DoIt.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.edc: type = fn_type_with_self_type %I.DoIt.type, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.edc = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.DoIt(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_8.2: %I.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc8_8.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_8.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %.loc10_4.2: type = fn_type_with_self_type constants.%I.DoIt.type, %T.loc8_8.1 [symbolic = %.loc10_4.2 (constants.%.edc)]
// CHECK:STDOUT:   %impl.elem0.loc10_4.2: @Use.%.loc10_4.2 (%.edc) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_4.2: <specific function> = specific_impl_function %impl.elem0.loc10_4.2, @I.DoIt(%T.loc8_8.1) [symbolic = %specific_impl_fn.loc10_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc8_8.2 [symbolic = %T.loc8_8.1 (constants.%T)]
// CHECK:STDOUT:     %DoIt.ref: %I.assoc_type = name_ref DoIt, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc10_4.1: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc10_4.1: @Use.%.loc10_4.2 (%.edc) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_4.1: <specific function> = specific_impl_function %impl.elem0.loc10_4.1, @I.DoIt(constants.%T) [symbolic = %specific_impl_fn.loc10_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc10_10: init %empty_tuple.type = call %specific_impl_fn.loc10_4.1()
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc8_8.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- assoc_fn_using_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.Make.type: type = fn_type @I.Make [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%I.Make.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.3f7859.2: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.07c: type = fn_type_with_self_type %I.Make.type, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.07c = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.Make(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_8.2: %I.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_8.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %.loc10_11.2: type = fn_type_with_self_type constants.%I.Make.type, %T.loc8_8.1 [symbolic = %.loc10_11.2 (constants.%.07c)]
// CHECK:STDOUT:   %impl.elem0.loc10_11.2: @Use.%.loc10_11.2 (%.07c) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_11.2: <specific function> = specific_impl_function %impl.elem0.loc10_11.2, @I.Make(%T.loc8_8.1) [symbolic = %specific_impl_fn.loc10_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param: @Use.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc10: %I.type = name_ref T, %T.loc8_8.2 [symbolic = %T.loc8_8.1 (constants.%T)]
// CHECK:STDOUT:     %Make.ref: %I.assoc_type = name_ref Make, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc10: type = facet_access_type %T.ref.loc10 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc10_11.1: type = converted %T.ref.loc10, %T.as_type.loc10 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc10_11.1: @Use.%.loc10_11.2 (%.07c) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_11.1: <specific function> = specific_impl_function %impl.elem0.loc10_11.1, @I.Make(constants.%T) [symbolic = %specific_impl_fn.loc10_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc10_17: init @Use.%T.binding.as_type (%T.binding.as_type) = call %specific_impl_fn.loc10_11.1() to %.loc8_15
// CHECK:STDOUT:     return %.loc10_17 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc8_8.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3f7859.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_assoc_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.Copy.type: type = fn_type @I.Copy [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%I.Copy.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.09a: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.3f7859.2: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.6f7: type = fn_type_with_self_type %I.Copy.type, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.6f7 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.Copy(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {
// CHECK:STDOUT:     %T.patt: %pattern_type.09a = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @Use.%pattern_type (%pattern_type.3f7859.2) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @Use.%pattern_type (%pattern_type.3f7859.2) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Use.%pattern_type (%pattern_type.3f7859.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Use.%pattern_type (%pattern_type.3f7859.2) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_24: %I.type = name_ref T, %T.loc9_8.2 [symbolic = %T.loc9_8.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc9_24: type = facet_access_type %T.ref.loc9_24 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc9_24: type = converted %T.ref.loc9_24, %T.as_type.loc9_24 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc9_12: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_8.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_8.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Use.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_18.1: type = splice_block %.loc9_18.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc9_18: %I.type = name_ref T, %T.loc9_8.2 [symbolic = %T.loc9_8.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc9_18: type = facet_access_type %T.ref.loc9_18 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc9_18.2: type = converted %T.ref.loc9_18, %T.as_type.loc9_18 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Use.%T.binding.as_type (%T.binding.as_type) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Use.%T.binding.as_type (%T.binding.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Use.%T.binding.as_type (%T.binding.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc9_8.2: %I.type) {
// CHECK:STDOUT:   %T.loc9_8.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_8.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc9_8.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.3f7859.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc9_8.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %.loc10_11: type = fn_type_with_self_type constants.%I.Copy.type, %T.loc9_8.1 [symbolic = %.loc10_11 (constants.%.6f7)]
// CHECK:STDOUT:   %impl.elem0.loc10_11.2: @Use.%.loc10_11 (%.6f7) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_11.2: <specific function> = specific_impl_function %impl.elem0.loc10_11.2, @I.Copy(%T.loc9_8.1) [symbolic = %specific_impl_fn.loc10_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Use.%T.binding.as_type (%T.binding.as_type)) -> %return.param: @Use.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Use.%T.binding.as_type (%T.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Copy.ref: %I.assoc_type = name_ref Copy, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc10_11.1: @Use.%.loc10_11 (%.6f7) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc10_11: <bound method> = bound_method %x.ref, %impl.elem0.loc10_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_11.1: <specific function> = specific_impl_function %impl.elem0.loc10_11.1, @I.Copy(constants.%T) [symbolic = %specific_impl_fn.loc10_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc10_17: <bound method> = bound_method %x.ref, %specific_impl_fn.loc10_11.1
// CHECK:STDOUT:     %.loc9_21: ref @Use.%T.binding.as_type (%T.binding.as_type) = splice_block %return {}
// CHECK:STDOUT:     %.loc10_17: init @Use.%T.binding.as_type (%T.binding.as_type) = call %bound_method.loc10_17(%x.ref) to %.loc9_21
// CHECK:STDOUT:     return %.loc10_17 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc9_8.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3f7859.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_selfless_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.Hello.type: type = fn_type @I.Hello [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%I.Hello.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.3f7: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.b73: type = fn_type_with_self_type %I.Hello.type, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.b73 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.Hello(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_8.2: %I.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_8.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %.loc10_4: type = fn_type_with_self_type constants.%I.Hello.type, %T.loc8_8.1 [symbolic = %.loc10_4 (constants.%.b73)]
// CHECK:STDOUT:   %impl.elem0.loc10_4.2: @Use.%.loc10_4 (%.b73) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_4.2: <specific function> = specific_impl_function %impl.elem0.loc10_4.2, @I.Hello(%T.loc8_8.1) [symbolic = %specific_impl_fn.loc10_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Use.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Use.%T.binding.as_type (%T.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Hello.ref: %I.assoc_type = name_ref Hello, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc10_4.1: @Use.%.loc10_4 (%.b73) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_4.1: <specific function> = specific_impl_function %impl.elem0.loc10_4.1, @I.Hello(constants.%T) [symbolic = %specific_impl_fn.loc10_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc10_11: init %empty_tuple.type = call %specific_impl_fn.loc10_4.1()
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc8_8.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3f7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_assoc_method_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.Copy.type: type = fn_type @I.Copy [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%I.Copy.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.3f7859.2: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.6f7: type = fn_type_with_self_type %I.Copy.type, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.6f7 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.Copy(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @UseIndirect(%T.loc8_16.2: %I.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_16.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %.loc10_14.2: type = fn_type_with_self_type constants.%I.Copy.type, %T.loc8_16.1 [symbolic = %.loc10_14.2 (constants.%.6f7)]
// CHECK:STDOUT:   %impl.elem0.loc10_14.2: @UseIndirect.%.loc10_14.2 (%.6f7) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_14.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_14.2: <specific function> = specific_impl_function %impl.elem0.loc10_14.2, @I.Copy(%T.loc8_16.1) [symbolic = %specific_impl_fn.loc10_14.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @UseIndirect.%T.binding.as_type (%T.binding.as_type)) -> %return.param: @UseIndirect.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @UseIndirect.%T.binding.as_type (%T.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     %T.ref.loc10: %I.type = name_ref T, %T.loc8_16.2 [symbolic = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:     %Copy.ref: %I.assoc_type = name_ref Copy, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc10: type = facet_access_type %T.ref.loc10 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc10_14.1: type = converted %T.ref.loc10, %T.as_type.loc10 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc10_14.1: @UseIndirect.%.loc10_14.2 (%.6f7) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_14.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc10_11: <bound method> = bound_method %x.ref, %impl.elem0.loc10_14.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_14.1: <specific function> = specific_impl_function %impl.elem0.loc10_14.1, @I.Copy(constants.%T) [symbolic = %specific_impl_fn.loc10_14.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc10_21: <bound method> = bound_method %x.ref, %specific_impl_fn.loc10_14.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc10_21: init @UseIndirect.%T.binding.as_type (%T.binding.as_type) = call %bound_method.loc10_21(%x.ref) to %.loc8_29
// CHECK:STDOUT:     return %.loc10_21 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseIndirect(constants.%T) {
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3f7859.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_constant_in_self_facet.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, @A.%X [concrete]
// CHECK:STDOUT:   %.Self.56d: %A.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.56d [symbolic_self]
// CHECK:STDOUT:   %A.lookup_impl_witness.033: <witness> = lookup_impl_witness %.Self.56d, @A [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %A.lookup_impl_witness.033, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A_where.type: type = facet_type <@A where %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %AA: %A_where.type = bind_symbolic_name AA, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.14a: type = pattern_type %A_where.type [concrete]
// CHECK:STDOUT:   %AA.binding.as_type: type = symbolic_binding_type AA, 0, %AA [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.614: <witness> = lookup_impl_witness %AA, @A [symbolic]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %AA.patt: %pattern_type.14a = symbolic_binding_pattern AA, 0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AA.ref: %A_where.type = name_ref AA, %AA.loc6_6.2 [symbolic = %AA.loc6_6.1 (constants.%AA)]
// CHECK:STDOUT:     %X.ref.loc6_33: %A.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %AA.as_type: type = facet_access_type %AA.ref [symbolic = %AA.binding.as_type (constants.%AA.binding.as_type)]
// CHECK:STDOUT:     %.loc6_33: type = converted %AA.ref, %AA.as_type [symbolic = %AA.binding.as_type (constants.%AA.binding.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc6_33: type = impl_witness_access constants.%A.lookup_impl_witness.614, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc6_13.1: type = splice_block %.loc6_13.2 [concrete = constants.%A_where.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: %A.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.56d]
// CHECK:STDOUT:       %X.ref.loc6_19: %A.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc6_19: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %impl.elem0.loc6_19: type = impl_witness_access constants.%A.lookup_impl_witness.033, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc6_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc6_25.2: type = converted %.loc6_25.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc6_13.2: type = where_expr %.Self.2 [concrete = constants.%A_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%A.type
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc6_19, %.loc6_25.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %AA.loc6_6.2: %A_where.type = bind_symbolic_name AA, 0 [symbolic = %AA.loc6_6.1 (constants.%AA)]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%AA.loc6_6.2: %A_where.type) {
// CHECK:STDOUT:   %AA.loc6_6.1: %A_where.type = bind_symbolic_name AA, 0 [symbolic = %AA.loc6_6.1 (constants.%AA)]
// CHECK:STDOUT:   %AA.binding.as_type: type = symbolic_binding_type AA, 0, %AA.loc6_6.1 [symbolic = %AA.binding.as_type (constants.%AA.binding.as_type)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %AA.loc6_6.1, @A [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness.614)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc7_11.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_11.2: init %empty_tuple.type = tuple_init () to %return [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_12: init %empty_tuple.type = converted %.loc7_11.1, %.loc7_11.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     return %.loc7_12 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%AA) {
// CHECK:STDOUT:   %AA.loc6_6.1 => constants.%AA
// CHECK:STDOUT:   %AA.binding.as_type => constants.%AA.binding.as_type
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.lookup_impl_witness.614
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_constant_in_self_facet_with_multiple_interfaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0.752: %A.assoc_type = assoc_entity element0, @A.%X [concrete]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %B.assoc_type: type = assoc_entity_type @B [concrete]
// CHECK:STDOUT:   %assoc0.081: %B.assoc_type = assoc_entity element0, @B.%Y [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.8a6: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.Op.type.9a3: type = fn_type @BitAndWith.Op, @BitAndWith(type) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.8a6 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.fa7: type = fn_type_with_self_type %BitAndWith.Op.type.9a3, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %A.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.c5c: type = facet_type <@A & @B> [concrete]
// CHECK:STDOUT:   %.Self.f55: %facet_type.c5c = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.f55 [symbolic_self]
// CHECK:STDOUT:   %A.lookup_impl_witness.a95: <witness> = lookup_impl_witness %.Self.f55, @A [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ade: type = impl_witness_access %A.lookup_impl_witness.a95, element0 [symbolic_self]
// CHECK:STDOUT:   %B.lookup_impl_witness.214: <witness> = lookup_impl_witness %.Self.f55, @B [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.818: type = impl_witness_access %B.lookup_impl_witness.214, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %facet_type.a18: type = facet_type <@A & @B where %impl.elem0.ade = %empty_tuple.type and %impl.elem0.818 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %AB: %facet_type.a18 = bind_symbolic_name AB, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.e98: type = pattern_type %facet_type.a18 [concrete]
// CHECK:STDOUT:   %AB.binding.as_type: type = symbolic_binding_type AB, 0, %AB [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.d9a: <witness> = lookup_impl_witness %AB, @A [symbolic]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %B.lookup_impl_witness.628: <witness> = lookup_impl_witness %AB, @B [symbolic]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.636: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.636), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %AB.patt: %pattern_type.e98 = symbolic_binding_pattern AB, 0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AB.ref: %facet_type.a18 = name_ref AB, %AB.loc14_6.2 [symbolic = %AB.loc14_6.1 (constants.%AB)]
// CHECK:STDOUT:     %X.ref.loc14_49: %A.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.752]
// CHECK:STDOUT:     %AB.as_type: type = facet_access_type %AB.ref [symbolic = %AB.binding.as_type (constants.%AB.binding.as_type)]
// CHECK:STDOUT:     %.loc14_49: type = converted %AB.ref, %AB.as_type [symbolic = %AB.binding.as_type (constants.%AB.binding.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc14_49: type = impl_witness_access constants.%A.lookup_impl_witness.d9a, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc14_17.1: type = splice_block %.loc14_17.2 [concrete = constants.%facet_type.a18] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %impl.elem0.loc14_13: %.fa7 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %A.ref, %impl.elem0.loc14_13 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method(%A.ref, %B.ref) [concrete = constants.%facet_type.c5c]
// CHECK:STDOUT:       %.loc14_13.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type.c5c]
// CHECK:STDOUT:       %.loc14_13.2: type = converted %type.as.BitAndWith.impl.Op.call, %.loc14_13.1 [concrete = constants.%facet_type.c5c]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref.loc14_23: %facet_type.c5c = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.f55]
// CHECK:STDOUT:       %X.ref.loc14_23: %A.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.752]
// CHECK:STDOUT:       %.Self.as_type.loc14_23: type = facet_access_type %.Self.ref.loc14_23 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc14_23: type = converted %.Self.ref.loc14_23, %.Self.as_type.loc14_23 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %impl.elem0.loc14_23: type = impl_witness_access constants.%A.lookup_impl_witness.a95, element0 [symbolic_self = constants.%impl.elem0.ade]
// CHECK:STDOUT:       %.loc14_29.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc14_29.2: type = converted %.loc14_29.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.Self.ref.loc14_35: %facet_type.c5c = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.f55]
// CHECK:STDOUT:       %Y.ref: %B.assoc_type = name_ref Y, @Y.%assoc0 [concrete = constants.%assoc0.081]
// CHECK:STDOUT:       %.Self.as_type.loc14_35: type = facet_access_type %.Self.ref.loc14_35 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc14_35: type = converted %.Self.ref.loc14_35, %.Self.as_type.loc14_35 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %impl.elem0.loc14_35: type = impl_witness_access constants.%B.lookup_impl_witness.214, element0 [symbolic_self = constants.%impl.elem0.818]
// CHECK:STDOUT:       %.loc14_41.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc14_41.2: type = converted %.loc14_41.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc14_17.2: type = where_expr %.Self.2 [concrete = constants.%facet_type.a18] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%facet_type.c5c
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc14_23, %.loc14_29.2
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc14_35, %.loc14_41.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %AB.loc14_6.2: %facet_type.a18 = bind_symbolic_name AB, 0 [symbolic = %AB.loc14_6.1 (constants.%AB)]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %AB.patt: %pattern_type.e98 = symbolic_binding_pattern AB, 0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AB.ref: %facet_type.a18 = name_ref AB, %AB.loc18_6.2 [symbolic = %AB.loc18_6.1 (constants.%AB)]
// CHECK:STDOUT:     %Y.ref.loc18_49: %B.assoc_type = name_ref Y, @Y.%assoc0 [concrete = constants.%assoc0.081]
// CHECK:STDOUT:     %AB.as_type: type = facet_access_type %AB.ref [symbolic = %AB.binding.as_type (constants.%AB.binding.as_type)]
// CHECK:STDOUT:     %.loc18_49: type = converted %AB.ref, %AB.as_type [symbolic = %AB.binding.as_type (constants.%AB.binding.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc18_49: type = impl_witness_access constants.%B.lookup_impl_witness.628, element0 [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc18_17.1: type = splice_block %.loc18_17.2 [concrete = constants.%facet_type.a18] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %impl.elem0.loc18_13: %.fa7 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %A.ref, %impl.elem0.loc18_13 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method(%A.ref, %B.ref) [concrete = constants.%facet_type.c5c]
// CHECK:STDOUT:       %.loc18_13.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type.c5c]
// CHECK:STDOUT:       %.loc18_13.2: type = converted %type.as.BitAndWith.impl.Op.call, %.loc18_13.1 [concrete = constants.%facet_type.c5c]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref.loc18_23: %facet_type.c5c = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.f55]
// CHECK:STDOUT:       %X.ref: %A.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.752]
// CHECK:STDOUT:       %.Self.as_type.loc18_23: type = facet_access_type %.Self.ref.loc18_23 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc18_23: type = converted %.Self.ref.loc18_23, %.Self.as_type.loc18_23 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %impl.elem0.loc18_23: type = impl_witness_access constants.%A.lookup_impl_witness.a95, element0 [symbolic_self = constants.%impl.elem0.ade]
// CHECK:STDOUT:       %.loc18_29.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc18_29.2: type = converted %.loc18_29.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.Self.ref.loc18_35: %facet_type.c5c = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.f55]
// CHECK:STDOUT:       %Y.ref.loc18_35: %B.assoc_type = name_ref Y, @Y.%assoc0 [concrete = constants.%assoc0.081]
// CHECK:STDOUT:       %.Self.as_type.loc18_35: type = facet_access_type %.Self.ref.loc18_35 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc18_35: type = converted %.Self.ref.loc18_35, %.Self.as_type.loc18_35 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %impl.elem0.loc18_35: type = impl_witness_access constants.%B.lookup_impl_witness.214, element0 [symbolic_self = constants.%impl.elem0.818]
// CHECK:STDOUT:       %.loc18_41.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc18_41.2: type = converted %.loc18_41.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc18_17.2: type = where_expr %.Self.2 [concrete = constants.%facet_type.a18] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%facet_type.c5c
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc18_23, %.loc18_29.2
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc18_35, %.loc18_41.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %AB.loc18_6.2: %facet_type.a18 = bind_symbolic_name AB, 0 [symbolic = %AB.loc18_6.1 (constants.%AB)]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%AB.loc14_6.2: %facet_type.a18) {
// CHECK:STDOUT:   %AB.loc14_6.1: %facet_type.a18 = bind_symbolic_name AB, 0 [symbolic = %AB.loc14_6.1 (constants.%AB)]
// CHECK:STDOUT:   %AB.binding.as_type: type = symbolic_binding_type AB, 0, %AB.loc14_6.1 [symbolic = %AB.binding.as_type (constants.%AB.binding.as_type)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %AB.loc14_6.1, @A [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness.d9a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc15_11.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_11.2: init %empty_tuple.type = tuple_init () to %return [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc15_12: init %empty_tuple.type = converted %.loc15_11.1, %.loc15_11.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     return %.loc15_12 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%AB.loc18_6.2: %facet_type.a18) {
// CHECK:STDOUT:   %AB.loc18_6.1: %facet_type.a18 = bind_symbolic_name AB, 0 [symbolic = %AB.loc18_6.1 (constants.%AB)]
// CHECK:STDOUT:   %AB.binding.as_type: type = symbolic_binding_type AB, 0, %AB.loc18_6.1 [symbolic = %AB.binding.as_type (constants.%AB.binding.as_type)]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %AB.loc18_6.1, @B [symbolic = %B.lookup_impl_witness (constants.%B.lookup_impl_witness.628)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %empty_struct_type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc19_11.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc19_11.2: init %empty_struct_type = struct_init () to %return [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc19_12: init %empty_struct_type = converted %.loc19_11.1, %.loc19_11.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:     return %.loc19_12 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%AB) {
// CHECK:STDOUT:   %AB.loc14_6.1 => constants.%AB
// CHECK:STDOUT:   %AB.binding.as_type => constants.%AB.binding.as_type
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.lookup_impl_witness.d9a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%AB) {
// CHECK:STDOUT:   %AB.loc18_6.1 => constants.%AB
// CHECK:STDOUT:   %AB.binding.as_type => constants.%AB.binding.as_type
// CHECK:STDOUT:   %B.lookup_impl_witness => constants.%B.lookup_impl_witness.628
// CHECK:STDOUT: }
// CHECK:STDOUT:
