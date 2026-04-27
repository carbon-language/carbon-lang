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

// --- access_through_call_once.carbon
library "[[@TEST_NAME]]";

// TODO: Merge this test with the one below once it works.

interface I {
  let X:! type;
  fn G() -> X*;
}

fn F2[U:! I](unused V: U*) {}

fn F(U:! I where .X = .Self, unused V: U) {
  // The returned value of `G` type `U` which has access to the methods of `I`.
  U.G()->G();
  (U as type).G()->G();

  // The returned value of type `U` can be used as a value of type `U`.
  F2(U.G());
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
  fn G[self: Self]() -> X*;
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

//@dump-sem-ir-begin
interface A { let X:! type; }

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

// CHECK:STDOUT: --- access_assoc_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.DoIt.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.DoIt.type.66b: type = fn_type @I.WithSelf.DoIt, @I.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.f37: type = fn_type_with_self_type %I.WithSelf.DoIt.type.66b, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.f37 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.DoIt(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_9.2: %I.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc8_9.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %I.WithSelf.DoIt.type: type = fn_type @I.WithSelf.DoIt, @I.WithSelf(%T.loc8_9.1) [symbolic = %I.WithSelf.DoIt.type (constants.%I.WithSelf.DoIt.type.66b)]
// CHECK:STDOUT:   %.loc10_4.2: type = fn_type_with_self_type %I.WithSelf.DoIt.type, %T.loc8_9.1 [symbolic = %.loc10_4.2 (constants.%.f37)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_9.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc10_4.2: @Use.%.loc10_4.2 (%.f37) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_4.2: <specific function> = specific_impl_function %impl.elem0.loc10_4.2, @I.WithSelf.DoIt(%T.loc8_9.1) [symbolic = %specific_impl_fn.loc10_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc8_9.2 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc10_4.1: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %DoIt.ref: %I.assoc_type = name_ref DoIt, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc10_4.1: @Use.%.loc10_4.2 (%.f37) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_4.1: <specific function> = specific_impl_function %impl.elem0.loc10_4.1, @I.WithSelf.DoIt(constants.%T) [symbolic = %specific_impl_fn.loc10_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %I.WithSelf.DoIt.call: init %empty_tuple.type = call %specific_impl_fn.loc10_4.1()
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc8_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- assoc_fn_using_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.Make.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %.ff7: Core.Form = init_form %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.422: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Make.type.f20: type = fn_type @I.WithSelf.Make, @I.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %I.WithSelf.Make.type.f20, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.8e2 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.Make(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_9.2: %I.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.WithSelf.Make.type: type = fn_type @I.WithSelf.Make, @I.WithSelf(%T.loc8_9.1) [symbolic = %I.WithSelf.Make.type (constants.%I.WithSelf.Make.type.f20)]
// CHECK:STDOUT:   %.loc10_11.2: type = fn_type_with_self_type %I.WithSelf.Make.type, %T.loc8_9.1 [symbolic = %.loc10_11.2 (constants.%.8e2)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_9.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc10_11.2: @Use.%.loc10_11.2 (%.8e2) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_11.2: <specific function> = specific_impl_function %impl.elem0.loc10_11.2, @I.WithSelf.Make(%T.loc8_9.1) [symbolic = %specific_impl_fn.loc10_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Use.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc10: %I.type = name_ref T, %T.loc8_9.2 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc10: type = facet_access_type %T.ref.loc10 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc10_11.1: type = converted %T.ref.loc10, %T.as_type.loc10 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Make.ref: %I.assoc_type = name_ref Make, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc10_11.1: @Use.%.loc10_11.2 (%.8e2) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_11.1: <specific function> = specific_impl_function %impl.elem0.loc10_11.1, @I.WithSelf.Make(constants.%T) [symbolic = %specific_impl_fn.loc10_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %I.WithSelf.Make.call: init @Use.%T.binding.as_type (%T.binding.as_type) to %.loc8_18.1 = call %specific_impl_fn.loc10_11.1()
// CHECK:STDOUT:     return %I.WithSelf.Make.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc8_9.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %.loc8_18.2 => constants.%.ff7
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.422
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_assoc_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.Copy.decl [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.422: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %.ff7: Core.Form = init_form %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Copy.type.035: type = fn_type @I.WithSelf.Copy, @I.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.fa6: type = fn_type_with_self_type %I.WithSelf.Copy.type.035, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.fa6 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.Copy(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {
// CHECK:STDOUT:     %T.patt: %pattern_type.9d9 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @Use.%pattern_type (%pattern_type.422) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @Use.%pattern_type (%pattern_type.422) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @Use.%pattern_type (%pattern_type.422) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @Use.%pattern_type (%pattern_type.422) = return_slot_pattern %return.param_patt, %.loc9_24.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_24: %I.type = name_ref T, %T.loc9_9.2 [symbolic = %T.loc9_9.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc9_24: type = facet_access_type %T.ref.loc9_24 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc9_24.3: type = converted %T.ref.loc9_24, %T.as_type.loc9_24 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc9_24.4: Core.Form = init_form %.loc9_24.3 [symbolic = %.loc9_24.2 (constants.%.ff7)]
// CHECK:STDOUT:     %.loc9_12: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_9.2: %I.type = symbolic_binding T, 0 [symbolic = %T.loc9_9.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Use.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_18.1: type = splice_block %.loc9_18.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc9_18: %I.type = name_ref T, %T.loc9_9.2 [symbolic = %T.loc9_9.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc9_18: type = facet_access_type %T.ref.loc9_18 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc9_18.2: type = converted %T.ref.loc9_18, %T.as_type.loc9_18 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Use.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @Use.%T.binding.as_type (%T.binding.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Use.%T.binding.as_type (%T.binding.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc9_9.2: %I.type) {
// CHECK:STDOUT:   %T.loc9_9.1: %I.type = symbolic_binding T, 0 [symbolic = %T.loc9_9.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc9_9.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.422)]
// CHECK:STDOUT:   %.loc9_24.2: Core.Form = init_form %T.binding.as_type [symbolic = %.loc9_24.2 (constants.%.ff7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %I.WithSelf.Copy.type: type = fn_type @I.WithSelf.Copy, @I.WithSelf(%T.loc9_9.1) [symbolic = %I.WithSelf.Copy.type (constants.%I.WithSelf.Copy.type.035)]
// CHECK:STDOUT:   %.loc10: type = fn_type_with_self_type %I.WithSelf.Copy.type, %T.loc9_9.1 [symbolic = %.loc10 (constants.%.fa6)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc9_9.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc10_11.2: @Use.%.loc10 (%.fa6) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_11.2: <specific function> = specific_impl_function %impl.elem0.loc10_11.2, @I.WithSelf.Copy(%T.loc9_9.1) [symbolic = %specific_impl_fn.loc10_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Use.%T.binding.as_type (%T.binding.as_type)) -> out %return.param: @Use.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Use.%T.binding.as_type (%T.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Copy.ref: %I.assoc_type = name_ref Copy, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc10_11.1: @Use.%.loc10 (%.fa6) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc10_11: <bound method> = bound_method %x.ref, %impl.elem0.loc10_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_11.1: <specific function> = specific_impl_function %impl.elem0.loc10_11.1, @I.WithSelf.Copy(constants.%T) [symbolic = %specific_impl_fn.loc10_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc10_17: <bound method> = bound_method %x.ref, %specific_impl_fn.loc10_11.1
// CHECK:STDOUT:     %.loc9_24.1: ref @Use.%T.binding.as_type (%T.binding.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %I.WithSelf.Copy.call: init @Use.%T.binding.as_type (%T.binding.as_type) to %.loc9_24.1 = call %bound_method.loc10_17(%x.ref)
// CHECK:STDOUT:     return %I.WithSelf.Copy.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc9_9.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.422
// CHECK:STDOUT:   %.loc9_24.2 => constants.%.ff7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_selfless_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.Hello.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.422: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Hello.type.3eb: type = fn_type @I.WithSelf.Hello, @I.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.8de: type = fn_type_with_self_type %I.WithSelf.Hello.type.3eb, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.8de = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.Hello(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_9.2: %I.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.WithSelf.Hello.type: type = fn_type @I.WithSelf.Hello, @I.WithSelf(%T.loc8_9.1) [symbolic = %I.WithSelf.Hello.type (constants.%I.WithSelf.Hello.type.3eb)]
// CHECK:STDOUT:   %.loc10: type = fn_type_with_self_type %I.WithSelf.Hello.type, %T.loc8_9.1 [symbolic = %.loc10 (constants.%.8de)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_9.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc10_4.2: @Use.%.loc10 (%.8de) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_4.2: <specific function> = specific_impl_function %impl.elem0.loc10_4.2, @I.WithSelf.Hello(%T.loc8_9.1) [symbolic = %specific_impl_fn.loc10_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Use.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Use.%T.binding.as_type (%T.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Hello.ref: %I.assoc_type = name_ref Hello, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc10_4.1: @Use.%.loc10 (%.8de) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_4.1: <specific function> = specific_impl_function %impl.elem0.loc10_4.1, @I.WithSelf.Hello(constants.%T) [symbolic = %specific_impl_fn.loc10_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %I.WithSelf.Hello.call: init %empty_tuple.type = call %specific_impl_fn.loc10_4.1()
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc8_9.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.422
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_assoc_method_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.Copy.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.422: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %.ff7: Core.Form = init_form %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %I.WithSelf.Copy.type.035: type = fn_type @I.WithSelf.Copy, @I.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.fa6: type = fn_type_with_self_type %I.WithSelf.Copy.type.035, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.fa6 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.Copy(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @UseIndirect(%T.loc8_17.2: %I.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.WithSelf.Copy.type: type = fn_type @I.WithSelf.Copy, @I.WithSelf(%T.loc8_17.1) [symbolic = %I.WithSelf.Copy.type (constants.%I.WithSelf.Copy.type.035)]
// CHECK:STDOUT:   %.loc10_14.2: type = fn_type_with_self_type %I.WithSelf.Copy.type, %T.loc8_17.1 [symbolic = %.loc10_14.2 (constants.%.fa6)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_17.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc10_14.2: @UseIndirect.%.loc10_14.2 (%.fa6) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_14.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_14.2: <specific function> = specific_impl_function %impl.elem0.loc10_14.2, @I.WithSelf.Copy(%T.loc8_17.1) [symbolic = %specific_impl_fn.loc10_14.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @UseIndirect.%T.binding.as_type (%T.binding.as_type)) -> out %return.param: @UseIndirect.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @UseIndirect.%T.binding.as_type (%T.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     %T.ref.loc10: %I.type = name_ref T, %T.loc8_17.2 [symbolic = %T.loc8_17.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc10: type = facet_access_type %T.ref.loc10 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc10_14.1: type = converted %T.ref.loc10, %T.as_type.loc10 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Copy.ref: %I.assoc_type = name_ref Copy, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc10_14.1: @UseIndirect.%.loc10_14.2 (%.fa6) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_14.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc10_11: <bound method> = bound_method %x.ref, %impl.elem0.loc10_14.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_14.1: <specific function> = specific_impl_function %impl.elem0.loc10_14.1, @I.WithSelf.Copy(constants.%T) [symbolic = %specific_impl_fn.loc10_14.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc10_21: <bound method> = bound_method %x.ref, %specific_impl_fn.loc10_14.1
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %I.WithSelf.Copy.call: init @UseIndirect.%T.binding.as_type (%T.binding.as_type) to %.loc8_32.1 = call %bound_method.loc10_21(%x.ref)
// CHECK:STDOUT:     return %I.WithSelf.Copy.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseIndirect(constants.%T) {
// CHECK:STDOUT:   %T.loc8_17.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.422
// CHECK:STDOUT:   %.loc8_32.2 => constants.%.ff7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_constant_in_self_facet.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, @A.WithSelf.%X [concrete]
// CHECK:STDOUT:   %.Self.091: %A.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.091 [symbolic_self]
// CHECK:STDOUT:   %A.lookup_impl_witness.6a5: <witness> = lookup_impl_witness %.Self.091, @A [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %A.lookup_impl_witness.6a5, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %A_where.type: type = facet_type <@A where %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %pattern_type.dc8: type = pattern_type %A_where.type [concrete]
// CHECK:STDOUT:   %AA: %A_where.type = symbolic_binding AA, 0 [symbolic]
// CHECK:STDOUT:   %AA.binding.as_type: type = symbolic_binding_type AA, 0, %AA [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.6c3: <witness> = lookup_impl_witness %AA, @A [symbolic]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %AA.binding.as_type, (%A.lookup_impl_witness.6c3) [symbolic]
// CHECK:STDOUT:   %.262: Core.Form = init_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %AA.patt: %pattern_type.dc8 = symbolic_binding_pattern AA, 0 [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %impl.elem0.loc6_33 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AA.ref: %A_where.type = name_ref AA, %AA.loc6_8.2 [symbolic = %AA.loc6_8.1 (constants.%AA)]
// CHECK:STDOUT:     %AA.as_type: type = facet_access_type %AA.ref [symbolic = %AA.binding.as_type (constants.%AA.binding.as_type)]
// CHECK:STDOUT:     %.loc6_33.1: type = converted %AA.ref, %AA.as_type [symbolic = %AA.binding.as_type (constants.%AA.binding.as_type)]
// CHECK:STDOUT:     %X.ref.loc6_33: %A.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc6_33: type = impl_witness_access constants.%A.lookup_impl_witness.6c3, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc6_33.2: Core.Form = init_form %impl.elem0.loc6_33 [concrete = constants.%.262]
// CHECK:STDOUT:     %.loc6_13.1: type = splice_block %.loc6_13.2 [concrete = constants.%A_where.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: %A.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.091]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc6_19: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %X.ref.loc6_19: %A.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc6_19: type = impl_witness_access constants.%A.lookup_impl_witness.6a5, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc6_25.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc6_25.2: type = converted %.loc6_25.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc6_13.2: type = where_expr %.Self.2 [concrete = constants.%A_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%A.type
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc6_19, %.loc6_25.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %AA.loc6_8.2: %A_where.type = symbolic_binding AA, 0 [symbolic = %AA.loc6_8.1 (constants.%AA)]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:     %assoc0: %A.assoc_type = assoc_entity element0, @A.WithSelf.%X [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .X = @X.%assoc0
// CHECK:STDOUT:   witness = (@A.WithSelf.%X)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%AA.loc6_8.2: %A_where.type) {
// CHECK:STDOUT:   %AA.loc6_8.1: %A_where.type = symbolic_binding AA, 0 [symbolic = %AA.loc6_8.1 (constants.%AA)]
// CHECK:STDOUT:   %AA.binding.as_type: type = symbolic_binding_type AA, 0, %AA.loc6_8.1 [symbolic = %AA.binding.as_type (constants.%AA.binding.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc7_11.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_11.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_12: init %empty_tuple.type = converted %.loc7_11.1, %.loc7_11.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     return %.loc7_12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%.Self.091) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%AA) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%A.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%AA) {
// CHECK:STDOUT:   %AA.loc6_8.1 => constants.%AA
// CHECK:STDOUT:   %AA.binding.as_type => constants.%AA.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_constant_in_self_facet_with_multiple_interfaces.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0.df7: %A.assoc_type = assoc_entity element0, @A.WithSelf.%X [concrete]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %B.assoc_type: type = assoc_entity_type @B [concrete]
// CHECK:STDOUT:   %assoc0.6fa: %B.assoc_type = assoc_entity element0, @B.WithSelf.%Y [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.b10: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.b10 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.4bd: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.d15: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.4bd, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %A.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.9bb: type = facet_type <@A & @B> [concrete]
// CHECK:STDOUT:   %.Self.e7e: %facet_type.9bb = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.e7e [symbolic_self]
// CHECK:STDOUT:   %A.lookup_impl_witness.6b3: <witness> = lookup_impl_witness %.Self.e7e, @A [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.eb8: type = impl_witness_access %A.lookup_impl_witness.6b3, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %B.lookup_impl_witness.d4f: <witness> = lookup_impl_witness %.Self.e7e, @B [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.11d: type = impl_witness_access %B.lookup_impl_witness.d4f, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_type.82c: type = facet_type <@A & @B where %impl.elem0.eb8 = %empty_tuple.type and %impl.elem0.11d = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %pattern_type.77e: type = pattern_type %facet_type.82c [concrete]
// CHECK:STDOUT:   %AB: %facet_type.82c = symbolic_binding AB, 0 [symbolic]
// CHECK:STDOUT:   %AB.binding.as_type: type = symbolic_binding_type AB, 0, %AB [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.1b9: <witness> = lookup_impl_witness %AB, @A [symbolic]
// CHECK:STDOUT:   %B.lookup_impl_witness.97b: <witness> = lookup_impl_witness %AB, @B [symbolic]
// CHECK:STDOUT:   %.262: Core.Form = init_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %.469: Core.Form = init_form %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.8d3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.8d3), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %AB.patt: %pattern_type.77e = symbolic_binding_pattern AB, 0 [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %impl.elem0.loc14_49 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AB.ref: %facet_type.82c = name_ref AB, %AB.loc14_8.2 [symbolic = %AB.loc14_8.1 (constants.%AB)]
// CHECK:STDOUT:     %AB.as_type: type = facet_access_type %AB.ref [symbolic = %AB.binding.as_type (constants.%AB.binding.as_type)]
// CHECK:STDOUT:     %.loc14_49.1: type = converted %AB.ref, %AB.as_type [symbolic = %AB.binding.as_type (constants.%AB.binding.as_type)]
// CHECK:STDOUT:     %X.ref.loc14_49: %A.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.df7]
// CHECK:STDOUT:     %impl.elem0.loc14_49: type = impl_witness_access constants.%A.lookup_impl_witness.1b9, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc14_49.2: Core.Form = init_form %impl.elem0.loc14_49 [concrete = constants.%.262]
// CHECK:STDOUT:     %.loc14_17.1: type = splice_block %.loc14_17.2 [concrete = constants.%facet_type.82c] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %impl.elem0.loc14_13: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %A.ref, %impl.elem0.loc14_13 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method(%A.ref, %B.ref) [concrete = constants.%facet_type.9bb]
// CHECK:STDOUT:       %.loc14_13.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type.9bb]
// CHECK:STDOUT:       %.loc14_13.2: type = converted %type.as.BitAndWith.impl.Op.call, %.loc14_13.1 [concrete = constants.%facet_type.9bb]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref.loc14_23: %facet_type.9bb = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.e7e]
// CHECK:STDOUT:       %.Self.as_type.loc14_23: type = facet_access_type %.Self.ref.loc14_23 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc14_23: type = converted %.Self.ref.loc14_23, %.Self.as_type.loc14_23 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %X.ref.loc14_23: %A.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.df7]
// CHECK:STDOUT:       %impl.elem0.loc14_23: type = impl_witness_access constants.%A.lookup_impl_witness.6b3, element0 [symbolic_self = constants.%impl.elem0.eb8]
// CHECK:STDOUT:       %.loc14_29.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc14_29.2: type = converted %.loc14_29.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.Self.ref.loc14_35: %facet_type.9bb = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.e7e]
// CHECK:STDOUT:       %.Self.as_type.loc14_35: type = facet_access_type %.Self.ref.loc14_35 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc14_35: type = converted %.Self.ref.loc14_35, %.Self.as_type.loc14_35 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %Y.ref: %B.assoc_type = name_ref Y, @Y.%assoc0 [concrete = constants.%assoc0.6fa]
// CHECK:STDOUT:       %impl.elem0.loc14_35: type = impl_witness_access constants.%B.lookup_impl_witness.d4f, element0 [symbolic_self = constants.%impl.elem0.11d]
// CHECK:STDOUT:       %.loc14_41.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc14_41.2: type = converted %.loc14_41.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc14_17.2: type = where_expr %.Self.2 [concrete = constants.%facet_type.82c] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%facet_type.9bb
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc14_23, %.loc14_29.2
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc14_35, %.loc14_41.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %AB.loc14_8.2: %facet_type.82c = symbolic_binding AB, 0 [symbolic = %AB.loc14_8.1 (constants.%AB)]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %AB.patt: %pattern_type.77e = symbolic_binding_pattern AB, 0 [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern %return.param_patt, %impl.elem0.loc18_49 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AB.ref: %facet_type.82c = name_ref AB, %AB.loc18_8.2 [symbolic = %AB.loc18_8.1 (constants.%AB)]
// CHECK:STDOUT:     %AB.as_type: type = facet_access_type %AB.ref [symbolic = %AB.binding.as_type (constants.%AB.binding.as_type)]
// CHECK:STDOUT:     %.loc18_49.1: type = converted %AB.ref, %AB.as_type [symbolic = %AB.binding.as_type (constants.%AB.binding.as_type)]
// CHECK:STDOUT:     %Y.ref.loc18_49: %B.assoc_type = name_ref Y, @Y.%assoc0 [concrete = constants.%assoc0.6fa]
// CHECK:STDOUT:     %impl.elem0.loc18_49: type = impl_witness_access constants.%B.lookup_impl_witness.97b, element0 [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc18_49.2: Core.Form = init_form %impl.elem0.loc18_49 [concrete = constants.%.469]
// CHECK:STDOUT:     %.loc18_17.1: type = splice_block %.loc18_17.2 [concrete = constants.%facet_type.82c] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %impl.elem0.loc18_13: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %A.ref, %impl.elem0.loc18_13 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method(%A.ref, %B.ref) [concrete = constants.%facet_type.9bb]
// CHECK:STDOUT:       %.loc18_13.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type.9bb]
// CHECK:STDOUT:       %.loc18_13.2: type = converted %type.as.BitAndWith.impl.Op.call, %.loc18_13.1 [concrete = constants.%facet_type.9bb]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref.loc18_23: %facet_type.9bb = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.e7e]
// CHECK:STDOUT:       %.Self.as_type.loc18_23: type = facet_access_type %.Self.ref.loc18_23 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc18_23: type = converted %.Self.ref.loc18_23, %.Self.as_type.loc18_23 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %X.ref: %A.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.df7]
// CHECK:STDOUT:       %impl.elem0.loc18_23: type = impl_witness_access constants.%A.lookup_impl_witness.6b3, element0 [symbolic_self = constants.%impl.elem0.eb8]
// CHECK:STDOUT:       %.loc18_29.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc18_29.2: type = converted %.loc18_29.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.Self.ref.loc18_35: %facet_type.9bb = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.e7e]
// CHECK:STDOUT:       %.Self.as_type.loc18_35: type = facet_access_type %.Self.ref.loc18_35 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %.loc18_35: type = converted %.Self.ref.loc18_35, %.Self.as_type.loc18_35 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:       %Y.ref.loc18_35: %B.assoc_type = name_ref Y, @Y.%assoc0 [concrete = constants.%assoc0.6fa]
// CHECK:STDOUT:       %impl.elem0.loc18_35: type = impl_witness_access constants.%B.lookup_impl_witness.d4f, element0 [symbolic_self = constants.%impl.elem0.11d]
// CHECK:STDOUT:       %.loc18_41.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc18_41.2: type = converted %.loc18_41.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc18_17.2: type = where_expr %.Self.2 [concrete = constants.%facet_type.82c] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%facet_type.9bb
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc18_23, %.loc18_29.2
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc18_35, %.loc18_41.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %AB.loc18_8.2: %facet_type.82c = symbolic_binding AB, 0 [symbolic = %AB.loc18_8.1 (constants.%AB)]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%AB.loc14_8.2: %facet_type.82c) {
// CHECK:STDOUT:   %AB.loc14_8.1: %facet_type.82c = symbolic_binding AB, 0 [symbolic = %AB.loc14_8.1 (constants.%AB)]
// CHECK:STDOUT:   %AB.binding.as_type: type = symbolic_binding_type AB, 0, %AB.loc14_8.1 [symbolic = %AB.binding.as_type (constants.%AB.binding.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc15_11.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc15_11.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc15_12: init %empty_tuple.type = converted %.loc15_11.1, %.loc15_11.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     return %.loc15_12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%AB.loc18_8.2: %facet_type.82c) {
// CHECK:STDOUT:   %AB.loc18_8.1: %facet_type.82c = symbolic_binding AB, 0 [symbolic = %AB.loc18_8.1 (constants.%AB)]
// CHECK:STDOUT:   %AB.binding.as_type: type = symbolic_binding_type AB, 0, %AB.loc18_8.1 [symbolic = %AB.binding.as_type (constants.%AB.binding.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %empty_struct_type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc19_11.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc19_11.2: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc19_12: init %empty_struct_type = converted %.loc19_11.1, %.loc19_11.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:     return %.loc19_12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%AB) {
// CHECK:STDOUT:   %AB.loc14_8.1 => constants.%AB
// CHECK:STDOUT:   %AB.binding.as_type => constants.%AB.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%AB) {
// CHECK:STDOUT:   %AB.loc18_8.1 => constants.%AB
// CHECK:STDOUT:   %AB.binding.as_type => constants.%AB.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
