// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/facet_types.carbon
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

// --- fail_non_const_associated.carbon
library "[[@TEST_NAME]]";

interface I { let T:! type; }
fn Id[U:! type](x: U) -> U { return x; }
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

fn Id[U:! type](x: U) -> U { return x; }

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
// CHECK:STDOUT:   %DoIt.type: type = fn_type @DoIt [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%DoIt.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.ded: type = fn_type_with_self_type %DoIt.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.ded = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @DoIt(%I.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_8.1: %I.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc10_4.2: type = facet_access_type %T.loc8_8.2 [symbolic = %T.as_type.loc10_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_8.2, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type.loc10_4.2, (%I.lookup_impl_witness) [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc10_4.2: type = fn_type_with_self_type constants.%DoIt.type, %I.facet [symbolic = %.loc10_4.2 (constants.%.ded)]
// CHECK:STDOUT:   %impl.elem0.loc10_4.2: @Use.%.loc10_4.2 (%.ded) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_4.2: <specific function> = specific_impl_function %impl.elem0.loc10_4.2, @DoIt(%I.facet) [symbolic = %specific_impl_fn.loc10_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc8_8.1 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:     %DoIt.ref: %I.assoc_type = name_ref DoIt, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc10_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc10_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_4.1: type = converted %T.ref, %T.as_type.loc10_4.1 [symbolic = %T.as_type.loc10_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc10_4.1: @Use.%.loc10_4.2 (%.ded) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_4.1: <specific function> = specific_impl_function %impl.elem0.loc10_4.1, @DoIt(constants.%I.facet) [symbolic = %specific_impl_fn.loc10_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc10_10: init %empty_tuple.type = call %specific_impl_fn.loc10_4.1()
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc8_8.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- assoc_fn_using_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%Make.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.6de4e4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.44c: type = fn_type_with_self_type %Make.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.44c = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Make(%I.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_8.1: %I.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_8.2, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type.loc8_18.2, (%I.lookup_impl_witness) [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc10_11.2: type = fn_type_with_self_type constants.%Make.type, %I.facet [symbolic = %.loc10_11.2 (constants.%.44c)]
// CHECK:STDOUT:   %impl.elem0.loc10_11.2: @Use.%.loc10_11.2 (%.44c) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_11.2: <specific function> = specific_impl_function %impl.elem0.loc10_11.2, @Make(%I.facet) [symbolic = %specific_impl_fn.loc10_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Use.%T.as_type.loc8_18.2 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc10: %I.type = name_ref T, %T.loc8_8.1 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:     %Make.ref: %I.assoc_type = name_ref Make, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc10: type = facet_access_type %T.ref.loc10 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_11.1: type = converted %T.ref.loc10, %T.as_type.loc10 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc10_11.1: @Use.%.loc10_11.2 (%.44c) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_11.1: <specific function> = specific_impl_function %impl.elem0.loc10_11.1, @Make(constants.%I.facet) [symbolic = %specific_impl_fn.loc10_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc10_17: init @Use.%T.as_type.loc8_18.2 (%T.as_type) = call %specific_impl_fn.loc10_11.1()
// CHECK:STDOUT:     %.loc10_18.1: @Use.%T.as_type.loc8_18.2 (%T.as_type) = value_of_initializer %.loc10_17
// CHECK:STDOUT:     %.loc10_18.2: @Use.%T.as_type.loc8_18.2 (%T.as_type) = converted %.loc10_17, %.loc10_18.1
// CHECK:STDOUT:     return %.loc10_18.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc8_8.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc8_18.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de4e4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_assoc_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Copy.type: type = fn_type @Copy [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%Copy.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.2b5: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.6de4e4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.e3b: type = fn_type_with_self_type %Copy.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.e3b = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Copy(%I.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {
// CHECK:STDOUT:     %T.patt: %pattern_type.2b5 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @Use.%pattern_type (%pattern_type.6de4e4.2) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @Use.%pattern_type (%pattern_type.6de4e4.2) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Use.%pattern_type (%pattern_type.6de4e4.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Use.%pattern_type (%pattern_type.6de4e4.2) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_24: %I.type = name_ref T, %T.loc9_8.1 [symbolic = %T.loc9_8.2 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc9_24: type = facet_access_type %T.ref.loc9_24 [symbolic = %T.as_type.loc9_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_24: type = converted %T.ref.loc9_24, %T.as_type.loc9_24 [symbolic = %T.as_type.loc9_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc9_8.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_8.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Use.%T.as_type.loc9_18.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_18.1: type = splice_block %.loc9_18.2 [symbolic = %T.as_type.loc9_18.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc9_18: %I.type = name_ref T, %T.loc9_8.1 [symbolic = %T.loc9_8.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc9_18.1: type = facet_access_type %T.ref.loc9_18 [symbolic = %T.as_type.loc9_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_18.2: type = converted %T.ref.loc9_18, %T.as_type.loc9_18.1 [symbolic = %T.as_type.loc9_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Use.%T.as_type.loc9_18.2 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Use.%T.as_type.loc9_18.2 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Use.%T.as_type.loc9_18.2 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc9_8.1: %I.type) {
// CHECK:STDOUT:   %T.loc9_8.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_8.2 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc9_18.2: type = facet_access_type %T.loc9_8.2 [symbolic = %T.as_type.loc9_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc9_18.2 [symbolic = %pattern_type (constants.%pattern_type.6de4e4.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc9_18.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc9_8.2, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type.loc9_18.2, (%I.lookup_impl_witness) [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc10_11.2: type = fn_type_with_self_type constants.%Copy.type, %I.facet [symbolic = %.loc10_11.2 (constants.%.e3b)]
// CHECK:STDOUT:   %impl.elem0.loc10_11.2: @Use.%.loc10_11.2 (%.e3b) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_11.2: <specific function> = specific_impl_function %impl.elem0.loc10_11.2, @Copy(%I.facet) [symbolic = %specific_impl_fn.loc10_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Use.%T.as_type.loc9_18.2 (%T.as_type)) -> @Use.%T.as_type.loc9_18.2 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Use.%T.as_type.loc9_18.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Copy.ref: %I.assoc_type = name_ref Copy, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc10: type = facet_access_type constants.%T [symbolic = %T.as_type.loc9_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_11.1: type = converted constants.%T, %T.as_type.loc10 [symbolic = %T.as_type.loc9_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc10_11.1: @Use.%.loc10_11.2 (%.e3b) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc10_11: <bound method> = bound_method %x.ref, %impl.elem0.loc10_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_11.1: <specific function> = specific_impl_function %impl.elem0.loc10_11.1, @Copy(constants.%I.facet) [symbolic = %specific_impl_fn.loc10_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc10_17: <bound method> = bound_method %x.ref, %specific_impl_fn.loc10_11.1
// CHECK:STDOUT:     %.loc10_17: init @Use.%T.as_type.loc9_18.2 (%T.as_type) = call %bound_method.loc10_17(%x.ref)
// CHECK:STDOUT:     %.loc10_18.1: @Use.%T.as_type.loc9_18.2 (%T.as_type) = value_of_initializer %.loc10_17
// CHECK:STDOUT:     %.loc10_18.2: @Use.%T.as_type.loc9_18.2 (%T.as_type) = converted %.loc10_17, %.loc10_18.1
// CHECK:STDOUT:     return %.loc10_18.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc9_8.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc9_18.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de4e4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_selfless_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Hello.type: type = fn_type @Hello [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%Hello.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.e1d: type = fn_type_with_self_type %Hello.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.e1d = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Hello(%I.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_8.1: %I.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_8.2, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type.loc8_18.2, (%I.lookup_impl_witness) [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc10_4.2: type = fn_type_with_self_type constants.%Hello.type, %I.facet [symbolic = %.loc10_4.2 (constants.%.e1d)]
// CHECK:STDOUT:   %impl.elem0.loc10_4.2: @Use.%.loc10_4.2 (%.e1d) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_4.2: <specific function> = specific_impl_function %impl.elem0.loc10_4.2, @Hello(%I.facet) [symbolic = %specific_impl_fn.loc10_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Use.%T.as_type.loc8_18.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Use.%T.as_type.loc8_18.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Hello.ref: %I.assoc_type = name_ref Hello, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc10: type = facet_access_type constants.%T [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_4.1: type = converted constants.%T, %T.as_type.loc10 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc10_4.1: @Use.%.loc10_4.2 (%.e1d) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_4.1: <specific function> = specific_impl_function %impl.elem0.loc10_4.1, @Hello(constants.%I.facet) [symbolic = %specific_impl_fn.loc10_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc10_11: init %empty_tuple.type = call %specific_impl_fn.loc10_4.1()
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc8_8.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc8_18.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_assoc_method_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Copy.type: type = fn_type @Copy [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%Copy.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.6de4e4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.e3b: type = fn_type_with_self_type %Copy.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.e3b = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Copy(%I.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @UseIndirect(%T.loc8_16.1: %I.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_16.2, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type.loc8_26.2, (%I.lookup_impl_witness) [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc10_14.2: type = fn_type_with_self_type constants.%Copy.type, %I.facet [symbolic = %.loc10_14.2 (constants.%.e3b)]
// CHECK:STDOUT:   %impl.elem0.loc10_14.2: @UseIndirect.%.loc10_14.2 (%.e3b) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_14.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_14.2: <specific function> = specific_impl_function %impl.elem0.loc10_14.2, @Copy(%I.facet) [symbolic = %specific_impl_fn.loc10_14.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type)) -> @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %T.ref.loc10: %I.type = name_ref T, %T.loc8_16.1 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:     %Copy.ref: %I.assoc_type = name_ref Copy, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc10: type = facet_access_type %T.ref.loc10 [symbolic = %T.as_type.loc8_26.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_14.1: type = converted %T.ref.loc10, %T.as_type.loc10 [symbolic = %T.as_type.loc8_26.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc10_14.1: @UseIndirect.%.loc10_14.2 (%.e3b) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_14.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc10_11: <bound method> = bound_method %x.ref, %impl.elem0.loc10_14.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_14.1: <specific function> = specific_impl_function %impl.elem0.loc10_14.1, @Copy(constants.%I.facet) [symbolic = %specific_impl_fn.loc10_14.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc10_21: <bound method> = bound_method %x.ref, %specific_impl_fn.loc10_14.1
// CHECK:STDOUT:     %.loc10_21: init @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = call %bound_method.loc10_21(%x.ref)
// CHECK:STDOUT:     %.loc10_22.1: @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = value_of_initializer %.loc10_21
// CHECK:STDOUT:     %.loc10_22.2: @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = converted %.loc10_21, %.loc10_22.1
// CHECK:STDOUT:     return %.loc10_22.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseIndirect(constants.%T) {
// CHECK:STDOUT:   %T.loc8_16.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc8_26.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de4e4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_constant_in_self_facet.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, @A.%X [concrete]
// CHECK:STDOUT:   %.Self: %A.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %A.lookup_impl_witness.5ad: <witness> = lookup_impl_witness %.Self, @A [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %A.lookup_impl_witness.5ad, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A_where.type: type = facet_type <@A where %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %AA: %A_where.type = bind_symbolic_name AA, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.c57: type = pattern_type %A_where.type [concrete]
// CHECK:STDOUT:   %AA.as_type: type = facet_access_type %AA [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.c50: <witness> = lookup_impl_witness %AA, @A [symbolic]
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
// CHECK:STDOUT:     %AA.patt: %pattern_type.c57 = symbolic_binding_pattern AA, 0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AA.ref: %A_where.type = name_ref AA, %AA.loc6_6.1 [symbolic = %AA.loc6_6.2 (constants.%AA)]
// CHECK:STDOUT:     %X.ref.loc6_33: %A.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %AA.as_type.loc6_33.1: type = facet_access_type %AA.ref [symbolic = %AA.as_type.loc6_33.2 (constants.%AA.as_type)]
// CHECK:STDOUT:     %.loc6_33: type = converted %AA.ref, %AA.as_type.loc6_33.1 [symbolic = %AA.as_type.loc6_33.2 (constants.%AA.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc6_33: type = impl_witness_access constants.%A.lookup_impl_witness.c50, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc6_13.1: type = splice_block %.loc6_13.2 [concrete = constants.%A_where.type] {
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: %A.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %X.ref.loc6_19: %A.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc6_19: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem0.loc6_19: type = impl_witness_access constants.%A.lookup_impl_witness.5ad, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc6_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc6_25.2: type = converted %.loc6_25.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc6_13.2: type = where_expr %.Self [concrete = constants.%A_where.type] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc6_19, %.loc6_25.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %AA.loc6_6.1: %A_where.type = bind_symbolic_name AA, 0 [symbolic = %AA.loc6_6.2 (constants.%AA)]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%AA.loc6_6.1: %A_where.type) {
// CHECK:STDOUT:   %AA.loc6_6.2: %A_where.type = bind_symbolic_name AA, 0 [symbolic = %AA.loc6_6.2 (constants.%AA)]
// CHECK:STDOUT:   %AA.as_type.loc6_33.2: type = facet_access_type %AA.loc6_6.2 [symbolic = %AA.as_type.loc6_33.2 (constants.%AA.as_type)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %AA.loc6_6.2, @A [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness.c50)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc7_11: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_12: %empty_tuple.type = converted %.loc7_11, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     return %.loc7_12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%AA) {
// CHECK:STDOUT:   %AA.loc6_6.2 => constants.%AA
// CHECK:STDOUT:   %AA.as_type.loc6_33.2 => constants.%AA.as_type
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.lookup_impl_witness.c50
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
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Op.type.27a: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %BitAnd.impl_witness: <witness> = impl_witness imports.%BitAnd.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value type, (%BitAnd.impl_witness) [concrete]
// CHECK:STDOUT:   %.6bf: type = fn_type_with_self_type %Op.type.27a, %BitAnd.facet [concrete]
// CHECK:STDOUT:   %Op.type.c1b: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.0a6: %Op.type.c1b = struct_value () [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %A.type, %Op.0a6 [concrete]
// CHECK:STDOUT:   %facet_type.938: type = facet_type <@A & @B> [concrete]
// CHECK:STDOUT:   %.Self: %facet_type.938 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %A.lookup_impl_witness.878: <witness> = lookup_impl_witness %.Self, @A [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c9d: type = impl_witness_access %A.lookup_impl_witness.878, element0 [symbolic_self]
// CHECK:STDOUT:   %B.lookup_impl_witness.a3c: <witness> = lookup_impl_witness %.Self, @B [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.7b2: type = impl_witness_access %B.lookup_impl_witness.a3c, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %facet_type.6d1: type = facet_type <@A & @B where %impl.elem0.c9d = %empty_tuple.type and %impl.elem0.7b2 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %AB: %facet_type.6d1 = bind_symbolic_name AB, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.bca: type = pattern_type %facet_type.6d1 [concrete]
// CHECK:STDOUT:   %AB.as_type: type = facet_access_type %AB [symbolic]
// CHECK:STDOUT:   %A.lookup_impl_witness.d31: <witness> = lookup_impl_witness %AB, @A [symbolic]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %B.lookup_impl_witness.fc8: <witness> = lookup_impl_witness %AB, @B [symbolic]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.d90: %Op.type.c1b = import_ref Core//prelude, loc13_42, loaded [concrete = constants.%Op.0a6]
// CHECK:STDOUT:   %BitAnd.impl_witness_table = impl_witness_table (%Core.import_ref.d90), @impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %AB.patt: %pattern_type.bca = symbolic_binding_pattern AB, 0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AB.ref: %facet_type.6d1 = name_ref AB, %AB.loc14_6.1 [symbolic = %AB.loc14_6.2 (constants.%AB)]
// CHECK:STDOUT:     %X.ref.loc14_49: %A.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.752]
// CHECK:STDOUT:     %AB.as_type.loc14_49.1: type = facet_access_type %AB.ref [symbolic = %AB.as_type.loc14_49.2 (constants.%AB.as_type)]
// CHECK:STDOUT:     %.loc14_49: type = converted %AB.ref, %AB.as_type.loc14_49.1 [symbolic = %AB.as_type.loc14_49.2 (constants.%AB.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc14_49: type = impl_witness_access constants.%A.lookup_impl_witness.d31, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc14_17.1: type = splice_block %.loc14_17.2 [concrete = constants.%facet_type.6d1] {
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %impl.elem0.loc14_13: %.6bf = impl_witness_access constants.%BitAnd.impl_witness, element0 [concrete = constants.%Op.0a6]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %A.ref, %impl.elem0.loc14_13 [concrete = constants.%Op.bound]
// CHECK:STDOUT:       %type.and: init type = call %bound_method(%A.ref, %B.ref) [concrete = constants.%facet_type.938]
// CHECK:STDOUT:       %.loc14_13.1: type = value_of_initializer %type.and [concrete = constants.%facet_type.938]
// CHECK:STDOUT:       %.loc14_13.2: type = converted %type.and, %.loc14_13.1 [concrete = constants.%facet_type.938]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref.loc14_23: %facet_type.938 = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %X.ref.loc14_23: %A.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.752]
// CHECK:STDOUT:       %.Self.as_type.loc14_23: type = facet_access_type %.Self.ref.loc14_23 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc14_23: type = converted %.Self.ref.loc14_23, %.Self.as_type.loc14_23 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem0.loc14_23: type = impl_witness_access constants.%A.lookup_impl_witness.878, element0 [symbolic_self = constants.%impl.elem0.c9d]
// CHECK:STDOUT:       %.loc14_29.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc14_29.2: type = converted %.loc14_29.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.Self.ref.loc14_35: %facet_type.938 = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Y.ref: %B.assoc_type = name_ref Y, @Y.%assoc0 [concrete = constants.%assoc0.081]
// CHECK:STDOUT:       %.Self.as_type.loc14_35: type = facet_access_type %.Self.ref.loc14_35 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc14_35: type = converted %.Self.ref.loc14_35, %.Self.as_type.loc14_35 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem0.loc14_35: type = impl_witness_access constants.%B.lookup_impl_witness.a3c, element0 [symbolic_self = constants.%impl.elem0.7b2]
// CHECK:STDOUT:       %.loc14_41.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc14_41.2: type = converted %.loc14_41.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc14_17.2: type = where_expr %.Self [concrete = constants.%facet_type.6d1] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc14_23, %.loc14_29.2
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc14_35, %.loc14_41.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %AB.loc14_6.1: %facet_type.6d1 = bind_symbolic_name AB, 0 [symbolic = %AB.loc14_6.2 (constants.%AB)]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %AB.patt: %pattern_type.bca = symbolic_binding_pattern AB, 0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AB.ref: %facet_type.6d1 = name_ref AB, %AB.loc18_6.1 [symbolic = %AB.loc18_6.2 (constants.%AB)]
// CHECK:STDOUT:     %Y.ref.loc18_49: %B.assoc_type = name_ref Y, @Y.%assoc0 [concrete = constants.%assoc0.081]
// CHECK:STDOUT:     %AB.as_type.loc18_49.1: type = facet_access_type %AB.ref [symbolic = %AB.as_type.loc18_49.2 (constants.%AB.as_type)]
// CHECK:STDOUT:     %.loc18_49: type = converted %AB.ref, %AB.as_type.loc18_49.1 [symbolic = %AB.as_type.loc18_49.2 (constants.%AB.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc18_49: type = impl_witness_access constants.%B.lookup_impl_witness.fc8, element0 [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc18_17.1: type = splice_block %.loc18_17.2 [concrete = constants.%facet_type.6d1] {
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %impl.elem0.loc18_13: %.6bf = impl_witness_access constants.%BitAnd.impl_witness, element0 [concrete = constants.%Op.0a6]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %A.ref, %impl.elem0.loc18_13 [concrete = constants.%Op.bound]
// CHECK:STDOUT:       %type.and: init type = call %bound_method(%A.ref, %B.ref) [concrete = constants.%facet_type.938]
// CHECK:STDOUT:       %.loc18_13.1: type = value_of_initializer %type.and [concrete = constants.%facet_type.938]
// CHECK:STDOUT:       %.loc18_13.2: type = converted %type.and, %.loc18_13.1 [concrete = constants.%facet_type.938]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref.loc18_23: %facet_type.938 = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %X.ref: %A.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.752]
// CHECK:STDOUT:       %.Self.as_type.loc18_23: type = facet_access_type %.Self.ref.loc18_23 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc18_23: type = converted %.Self.ref.loc18_23, %.Self.as_type.loc18_23 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem0.loc18_23: type = impl_witness_access constants.%A.lookup_impl_witness.878, element0 [symbolic_self = constants.%impl.elem0.c9d]
// CHECK:STDOUT:       %.loc18_29.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc18_29.2: type = converted %.loc18_29.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.Self.ref.loc18_35: %facet_type.938 = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Y.ref.loc18_35: %B.assoc_type = name_ref Y, @Y.%assoc0 [concrete = constants.%assoc0.081]
// CHECK:STDOUT:       %.Self.as_type.loc18_35: type = facet_access_type %.Self.ref.loc18_35 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc18_35: type = converted %.Self.ref.loc18_35, %.Self.as_type.loc18_35 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem0.loc18_35: type = impl_witness_access constants.%B.lookup_impl_witness.a3c, element0 [symbolic_self = constants.%impl.elem0.7b2]
// CHECK:STDOUT:       %.loc18_41.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc18_41.2: type = converted %.loc18_41.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc18_17.2: type = where_expr %.Self [concrete = constants.%facet_type.6d1] {
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc18_23, %.loc18_29.2
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc18_35, %.loc18_41.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %AB.loc18_6.1: %facet_type.6d1 = bind_symbolic_name AB, 0 [symbolic = %AB.loc18_6.2 (constants.%AB)]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%AB.loc14_6.1: %facet_type.6d1) {
// CHECK:STDOUT:   %AB.loc14_6.2: %facet_type.6d1 = bind_symbolic_name AB, 0 [symbolic = %AB.loc14_6.2 (constants.%AB)]
// CHECK:STDOUT:   %AB.as_type.loc14_49.2: type = facet_access_type %AB.loc14_6.2 [symbolic = %AB.as_type.loc14_49.2 (constants.%AB.as_type)]
// CHECK:STDOUT:   %A.lookup_impl_witness: <witness> = lookup_impl_witness %AB.loc14_6.2, @A [symbolic = %A.lookup_impl_witness (constants.%A.lookup_impl_witness.d31)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc15_11: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc15_12: %empty_tuple.type = converted %.loc15_11, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     return %.loc15_12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%AB.loc18_6.1: %facet_type.6d1) {
// CHECK:STDOUT:   %AB.loc18_6.2: %facet_type.6d1 = bind_symbolic_name AB, 0 [symbolic = %AB.loc18_6.2 (constants.%AB)]
// CHECK:STDOUT:   %AB.as_type.loc18_49.2: type = facet_access_type %AB.loc18_6.2 [symbolic = %AB.as_type.loc18_49.2 (constants.%AB.as_type)]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %AB.loc18_6.2, @B [symbolic = %B.lookup_impl_witness (constants.%B.lookup_impl_witness.fc8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %empty_struct_type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc19_11: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc19_12: %empty_struct_type = converted %.loc19_11, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:     return %.loc19_12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%AB) {
// CHECK:STDOUT:   %AB.loc14_6.2 => constants.%AB
// CHECK:STDOUT:   %AB.as_type.loc14_49.2 => constants.%AB.as_type
// CHECK:STDOUT:   %A.lookup_impl_witness => constants.%A.lookup_impl_witness.d31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%AB) {
// CHECK:STDOUT:   %AB.loc18_6.2 => constants.%AB
// CHECK:STDOUT:   %AB.as_type.loc18_49.2 => constants.%AB.as_type
// CHECK:STDOUT:   %B.lookup_impl_witness => constants.%B.lookup_impl_witness.fc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
