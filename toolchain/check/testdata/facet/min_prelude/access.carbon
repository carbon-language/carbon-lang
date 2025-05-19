// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/min_prelude/access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/min_prelude/access.carbon

// --- access_assoc_fn.carbon

library "[[@TEST_NAME]]";

interface I {
  fn DoIt();
}

fn Use(T:! I) {
  T.DoIt();
}

// --- assoc_fn_using_self.carbon

library "[[@TEST_NAME]]";

interface I {
  fn Make() -> Self;
}

fn Use(T:! I) -> T {
  return T.Make();
}

// --- access_assoc_method.carbon

library "[[@TEST_NAME]]";

interface I {
  fn Copy[self: Self]() -> Self;
}

fn Use[T:! I](x: T) -> T {
  return x.Copy();
}

// --- access_selfless_method.carbon

library "[[@TEST_NAME]]";

interface I {
  fn Hello();
}

fn Use[T:! I](x: T){
  x.Hello();
}

// --- access_assoc_method_indirect.carbon

library "[[@TEST_NAME]]";

interface I {
  fn Copy[self: Self]() -> Self;
}

fn UseIndirect[T:! I](x: T) -> T {
  return x.(T.Copy)();
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

// CHECK:STDOUT: --- access_assoc_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %DoIt.type: type = fn_type @DoIt [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DoIt: %DoIt.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%DoIt.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.ded: type = fn_type_with_self_type %DoIt.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.ded = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @DoIt(%I.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc8_8.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %DoIt.decl: %DoIt.type = fn_decl @DoIt [concrete = constants.%DoIt] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %DoIt.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .DoIt = %assoc0
// CHECK:STDOUT:   witness = (%DoIt.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @DoIt(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_8.1: %I.type) {
// CHECK:STDOUT:   %T.loc8_8.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc9_4.2: type = facet_access_type %T.loc8_8.2 [symbolic = %T.as_type.loc9_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_8.2, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type.loc9_4.2, (%I.lookup_impl_witness) [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc9_4.2: type = fn_type_with_self_type constants.%DoIt.type, %I.facet [symbolic = %.loc9_4.2 (constants.%.ded)]
// CHECK:STDOUT:   %impl.elem0.loc9_4.2: @Use.%.loc9_4.2 (%.ded) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_4.2: <specific function> = specific_impl_function %impl.elem0.loc9_4.2, @DoIt(%I.facet) [symbolic = %specific_impl_fn.loc9_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc8_8.1 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:     %DoIt.ref: %I.assoc_type = name_ref DoIt, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc9_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_4.1: type = converted %T.ref, %T.as_type.loc9_4.1 [symbolic = %T.as_type.loc9_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc9_4.1: @Use.%.loc9_4.2 (%.ded) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_4.1: <specific function> = specific_impl_function %impl.elem0.loc9_4.1, @DoIt(constants.%I.facet) [symbolic = %specific_impl_fn.loc9_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc9_10: init %empty_tuple.type = call %specific_impl_fn.loc9_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DoIt(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc8_8.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DoIt(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- assoc_fn_using_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.6de4e4.1: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%Make.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.2b5: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.6de4e4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.44c: type = fn_type_with_self_type %Make.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.44c = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Make(%I.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {
// CHECK:STDOUT:     %T.patt: %pattern_type.2b5 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %return.patt: @Use.%pattern_type (%pattern_type.6de4e4.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Use.%pattern_type (%pattern_type.6de4e4.2) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8: %I.type = name_ref T, %T.loc8_8.1 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc8_18.1: type = facet_access_type %T.ref.loc8 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc8: type = converted %T.ref.loc8, %T.as_type.loc8_18.1 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc8_8.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:     %return.param: ref @Use.%T.as_type.loc8_18.2 (%T.as_type) = out_param call_param0
// CHECK:STDOUT:     %return: ref @Use.%T.as_type.loc8_18.2 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.patt: @Make.%pattern_type (%pattern_type.6de4e4.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Make.%pattern_type (%pattern_type.6de4e4.1) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc5_16.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc5: type = converted %Self.ref, %Self.as_type.loc5_16.2 [symbolic = %Self.as_type.loc5_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %return.param: ref @Make.%Self.as_type.loc5_16.1 (%Self.as_type) = out_param call_param0
// CHECK:STDOUT:     %return: ref @Make.%Self.as_type.loc5_16.1 (%Self.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %Make.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Make = %assoc0
// CHECK:STDOUT:   witness = (%Make.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_16.1 [symbolic = %pattern_type (constants.%pattern_type.6de4e4.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.%Self.as_type.loc5_16.1 (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_8.1: %I.type) {
// CHECK:STDOUT:   %T.loc8_8.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc8_18.2: type = facet_access_type %T.loc8_8.2 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc8_18.2 [symbolic = %pattern_type (constants.%pattern_type.6de4e4.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc8_18.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_8.2, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type.loc8_18.2, (%I.lookup_impl_witness) [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc9_11.2: type = fn_type_with_self_type constants.%Make.type, %I.facet [symbolic = %.loc9_11.2 (constants.%.44c)]
// CHECK:STDOUT:   %impl.elem0.loc9_11.2: @Use.%.loc9_11.2 (%.44c) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_11.2: <specific function> = specific_impl_function %impl.elem0.loc9_11.2, @Make(%I.facet) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Use.%T.as_type.loc8_18.2 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc9: %I.type = name_ref T, %T.loc8_8.1 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:     %Make.ref: %I.assoc_type = name_ref Make, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc9: type = facet_access_type %T.ref.loc9 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_11.1: type = converted %T.ref.loc9, %T.as_type.loc9 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc9_11.1: @Use.%.loc9_11.2 (%.44c) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_11.1: <specific function> = specific_impl_function %impl.elem0.loc9_11.1, @Make(constants.%I.facet) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc9_17: init @Use.%T.as_type.loc8_18.2 (%T.as_type) = call %specific_impl_fn.loc9_11.1()
// CHECK:STDOUT:     %.loc9_18.1: @Use.%T.as_type.loc8_18.2 (%T.as_type) = value_of_initializer %.loc9_17
// CHECK:STDOUT:     %.loc9_18.2: @Use.%T.as_type.loc8_18.2 (%T.as_type) = converted %.loc9_17, %.loc9_18.1
// CHECK:STDOUT:     return %.loc9_18.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_16.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de4e4.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc8_8.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc8_18.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de4e4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc5_16.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de4e4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_assoc_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.6de4e4.1: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Copy.type: type = fn_type @Copy [concrete]
// CHECK:STDOUT:   %Copy: %Copy.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {
// CHECK:STDOUT:     %T.patt: %pattern_type.2b5 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @Use.%pattern_type (%pattern_type.6de4e4.2) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @Use.%pattern_type (%pattern_type.6de4e4.2) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Use.%pattern_type (%pattern_type.6de4e4.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Use.%pattern_type (%pattern_type.6de4e4.2) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_24: %I.type = name_ref T, %T.loc8_8.1 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc8_24: type = facet_access_type %T.ref.loc8_24 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc8_24: type = converted %T.ref.loc8_24, %T.as_type.loc8_24 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc8_8.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Use.%T.as_type.loc8_18.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_18.1: type = splice_block %.loc8_18.2 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc8_18: %I.type = name_ref T, %T.loc8_8.1 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc8_18.1: type = facet_access_type %T.ref.loc8_18 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc8_18.2: type = converted %T.ref.loc8_18, %T.as_type.loc8_18.1 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Use.%T.as_type.loc8_18.2 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Use.%T.as_type.loc8_18.2 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Use.%T.as_type.loc8_18.2 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Copy.decl: %Copy.type = fn_decl @Copy [concrete = constants.%Copy] {
// CHECK:STDOUT:     %self.patt: @Copy.%pattern_type (%pattern_type.6de4e4.1) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Copy.%pattern_type (%pattern_type.6de4e4.1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Copy.%pattern_type (%pattern_type.6de4e4.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Copy.%pattern_type (%pattern_type.6de4e4.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc5_28: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc5_28: type = facet_access_type %Self.ref.loc5_28 [symbolic = %Self.as_type.loc5_17.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc5_28: type = converted %Self.ref.loc5_28, %Self.as_type.loc5_28 [symbolic = %Self.as_type.loc5_17.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %self.param: @Copy.%Self.as_type.loc5_17.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_17.1: type = splice_block %.loc5_17.2 [symbolic = %Self.as_type.loc5_17.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc5_17: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_17.2: type = facet_access_type %Self.ref.loc5_17 [symbolic = %Self.as_type.loc5_17.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_17.2: type = converted %Self.ref.loc5_17, %Self.as_type.loc5_17.2 [symbolic = %Self.as_type.loc5_17.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Copy.%Self.as_type.loc5_17.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref @Copy.%Self.as_type.loc5_17.1 (%Self.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Copy.%Self.as_type.loc5_17.1 (%Self.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %Copy.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Copy = %assoc0
// CHECK:STDOUT:   witness = (%Copy.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Copy(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_17.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_17.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_17.1 [symbolic = %pattern_type (constants.%pattern_type.6de4e4.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Copy.%Self.as_type.loc5_17.1 (%Self.as_type)) -> @Copy.%Self.as_type.loc5_17.1 (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_8.1: %I.type) {
// CHECK:STDOUT:   %T.loc8_8.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc8_18.2: type = facet_access_type %T.loc8_8.2 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc8_18.2 [symbolic = %pattern_type (constants.%pattern_type.6de4e4.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc8_18.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_8.2, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type.loc8_18.2, (%I.lookup_impl_witness) [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc9_11.2: type = fn_type_with_self_type constants.%Copy.type, %I.facet [symbolic = %.loc9_11.2 (constants.%.e3b)]
// CHECK:STDOUT:   %impl.elem0.loc9_11.2: @Use.%.loc9_11.2 (%.e3b) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_11.2: <specific function> = specific_impl_function %impl.elem0.loc9_11.2, @Copy(%I.facet) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Use.%T.as_type.loc8_18.2 (%T.as_type)) -> @Use.%T.as_type.loc8_18.2 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Use.%T.as_type.loc8_18.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Copy.ref: %I.assoc_type = name_ref Copy, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc9: type = facet_access_type constants.%T [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_11.1: type = converted constants.%T, %T.as_type.loc9 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc9_11.1: @Use.%.loc9_11.2 (%.e3b) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc9_11: <bound method> = bound_method %x.ref, %impl.elem0.loc9_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_11.1: <specific function> = specific_impl_function %impl.elem0.loc9_11.1, @Copy(constants.%I.facet) [symbolic = %specific_impl_fn.loc9_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc9_17: <bound method> = bound_method %x.ref, %specific_impl_fn.loc9_11.1
// CHECK:STDOUT:     %.loc9_17: init @Use.%T.as_type.loc8_18.2 (%T.as_type) = call %bound_method.loc9_17(%x.ref)
// CHECK:STDOUT:     %.loc9_18.1: @Use.%T.as_type.loc8_18.2 (%T.as_type) = value_of_initializer %.loc9_17
// CHECK:STDOUT:     %.loc9_18.2: @Use.%T.as_type.loc8_18.2 (%T.as_type) = converted %.loc9_17, %.loc9_18.1
// CHECK:STDOUT:     return %.loc9_18.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Copy(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_17.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de4e4.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc8_8.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc8_18.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de4e4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Copy(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc5_17.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de4e4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_selfless_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Hello.type: type = fn_type @Hello [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Hello: %Hello.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%Hello.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.2b5: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.e1d: type = fn_type_with_self_type %Hello.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.e1d = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Hello(%I.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {
// CHECK:STDOUT:     %T.patt: %pattern_type.2b5 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @Use.%pattern_type (%pattern_type.6de) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @Use.%pattern_type (%pattern_type.6de) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc8_8.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Use.%T.as_type.loc8_18.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_18.1: type = splice_block %.loc8_18.2 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %I.type = name_ref T, %T.loc8_8.1 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc8_18.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc8_18.2: type = converted %T.ref, %T.as_type.loc8_18.1 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Use.%T.as_type.loc8_18.2 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Hello.decl: %Hello.type = fn_decl @Hello [concrete = constants.%Hello] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %Hello.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Hello = %assoc0
// CHECK:STDOUT:   witness = (%Hello.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Hello(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_8.1: %I.type) {
// CHECK:STDOUT:   %T.loc8_8.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc8_18.2: type = facet_access_type %T.loc8_8.2 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc8_18.2 [symbolic = %pattern_type (constants.%pattern_type.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc8_18.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_8.2, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type.loc8_18.2, (%I.lookup_impl_witness) [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc9_4.2: type = fn_type_with_self_type constants.%Hello.type, %I.facet [symbolic = %.loc9_4.2 (constants.%.e1d)]
// CHECK:STDOUT:   %impl.elem0.loc9_4.2: @Use.%.loc9_4.2 (%.e1d) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_4.2: <specific function> = specific_impl_function %impl.elem0.loc9_4.2, @Hello(%I.facet) [symbolic = %specific_impl_fn.loc9_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Use.%T.as_type.loc8_18.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Use.%T.as_type.loc8_18.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Hello.ref: %I.assoc_type = name_ref Hello, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc9: type = facet_access_type constants.%T [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_4.1: type = converted constants.%T, %T.as_type.loc9 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc9_4.1: @Use.%.loc9_4.2 (%.e1d) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_4.1: <specific function> = specific_impl_function %impl.elem0.loc9_4.1, @Hello(constants.%I.facet) [symbolic = %specific_impl_fn.loc9_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc9_11: init %empty_tuple.type = call %specific_impl_fn.loc9_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Hello(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc8_8.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc8_18.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Hello(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_assoc_method_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.6de4e4.1: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %Copy.type: type = fn_type @Copy [concrete]
// CHECK:STDOUT:   %Copy: %Copy.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%Copy.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.2b5: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.6de4e4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %UseIndirect.type: type = fn_type @UseIndirect [concrete]
// CHECK:STDOUT:   %UseIndirect: %UseIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.e3b: type = fn_type_with_self_type %Copy.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.e3b = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Copy(%I.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .UseIndirect = %UseIndirect.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %UseIndirect.decl: %UseIndirect.type = fn_decl @UseIndirect [concrete = constants.%UseIndirect] {
// CHECK:STDOUT:     %T.patt: %pattern_type.2b5 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @UseIndirect.%pattern_type (%pattern_type.6de4e4.2) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @UseIndirect.%pattern_type (%pattern_type.6de4e4.2) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @UseIndirect.%pattern_type (%pattern_type.6de4e4.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @UseIndirect.%pattern_type (%pattern_type.6de4e4.2) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_32: %I.type = name_ref T, %T.loc8_16.1 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc8_32: type = facet_access_type %T.ref.loc8_32 [symbolic = %T.as_type.loc8_26.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc8_32: type = converted %T.ref.loc8_32, %T.as_type.loc8_32 [symbolic = %T.as_type.loc8_26.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc8_16.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_26.1: type = splice_block %.loc8_26.2 [symbolic = %T.as_type.loc8_26.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc8_26: %I.type = name_ref T, %T.loc8_16.1 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc8_26.1: type = facet_access_type %T.ref.loc8_26 [symbolic = %T.as_type.loc8_26.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc8_26.2: type = converted %T.ref.loc8_26, %T.as_type.loc8_26.1 [symbolic = %T.as_type.loc8_26.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Copy.decl: %Copy.type = fn_decl @Copy [concrete = constants.%Copy] {
// CHECK:STDOUT:     %self.patt: @Copy.%pattern_type (%pattern_type.6de4e4.1) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Copy.%pattern_type (%pattern_type.6de4e4.1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Copy.%pattern_type (%pattern_type.6de4e4.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Copy.%pattern_type (%pattern_type.6de4e4.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc5_28: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc5_28: type = facet_access_type %Self.ref.loc5_28 [symbolic = %Self.as_type.loc5_17.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc5_28: type = converted %Self.ref.loc5_28, %Self.as_type.loc5_28 [symbolic = %Self.as_type.loc5_17.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %self.param: @Copy.%Self.as_type.loc5_17.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_17.1: type = splice_block %.loc5_17.2 [symbolic = %Self.as_type.loc5_17.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc5_17: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_17.2: type = facet_access_type %Self.ref.loc5_17 [symbolic = %Self.as_type.loc5_17.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_17.2: type = converted %Self.ref.loc5_17, %Self.as_type.loc5_17.2 [symbolic = %Self.as_type.loc5_17.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Copy.%Self.as_type.loc5_17.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref @Copy.%Self.as_type.loc5_17.1 (%Self.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Copy.%Self.as_type.loc5_17.1 (%Self.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %Copy.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Copy = %assoc0
// CHECK:STDOUT:   witness = (%Copy.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Copy(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_17.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_17.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_17.1 [symbolic = %pattern_type (constants.%pattern_type.6de4e4.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Copy.%Self.as_type.loc5_17.1 (%Self.as_type)) -> @Copy.%Self.as_type.loc5_17.1 (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @UseIndirect(%T.loc8_16.1: %I.type) {
// CHECK:STDOUT:   %T.loc8_16.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc8_26.2: type = facet_access_type %T.loc8_16.2 [symbolic = %T.as_type.loc8_26.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc8_26.2 [symbolic = %pattern_type (constants.%pattern_type.6de4e4.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc8_26.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_16.2, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type.loc8_26.2, (%I.lookup_impl_witness) [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc9_14.2: type = fn_type_with_self_type constants.%Copy.type, %I.facet [symbolic = %.loc9_14.2 (constants.%.e3b)]
// CHECK:STDOUT:   %impl.elem0.loc9_14.2: @UseIndirect.%.loc9_14.2 (%.e3b) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_14.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_14.2: <specific function> = specific_impl_function %impl.elem0.loc9_14.2, @Copy(%I.facet) [symbolic = %specific_impl_fn.loc9_14.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type)) -> @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %T.ref.loc9: %I.type = name_ref T, %T.loc8_16.1 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:     %Copy.ref: %I.assoc_type = name_ref Copy, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc9: type = facet_access_type %T.ref.loc9 [symbolic = %T.as_type.loc8_26.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_14.1: type = converted %T.ref.loc9, %T.as_type.loc9 [symbolic = %T.as_type.loc8_26.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc9_14.1: @UseIndirect.%.loc9_14.2 (%.e3b) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_14.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc9_11: <bound method> = bound_method %x.ref, %impl.elem0.loc9_14.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_14.1: <specific function> = specific_impl_function %impl.elem0.loc9_14.1, @Copy(constants.%I.facet) [symbolic = %specific_impl_fn.loc9_14.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc9_21: <bound method> = bound_method %x.ref, %specific_impl_fn.loc9_14.1
// CHECK:STDOUT:     %.loc9_21: init @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = call %bound_method.loc9_21(%x.ref)
// CHECK:STDOUT:     %.loc9_22.1: @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = value_of_initializer %.loc9_21
// CHECK:STDOUT:     %.loc9_22.2: @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = converted %.loc9_21, %.loc9_22.1
// CHECK:STDOUT:     return %.loc9_22.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Copy(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_17.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de4e4.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseIndirect(constants.%T) {
// CHECK:STDOUT:   %T.loc8_16.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc8_26.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de4e4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Copy(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc5_17.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de4e4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_const_associated.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %Id.type: type = fn_type @Id [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Id: %Id.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %.Self.as_type, (%I.lookup_impl_witness) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.40b: type = pattern_type %I.assoc_type [concrete]
// CHECK:STDOUT:   %Id.specific_fn: <specific function> = specific_function %Id, @Id(%I.assoc_type) [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %I.assoc_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .Id = %Id.decl
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %Id.decl: %Id.type = fn_decl @Id [concrete = constants.%Id] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @Id.%pattern_type (%pattern_type.7dc) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @Id.%pattern_type (%pattern_type.7dc) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Id.%pattern_type (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Id.%pattern_type (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc4_26: type = name_ref U, %U.loc4_7.1 [symbolic = %U.loc4_7.2 (constants.%U)]
// CHECK:STDOUT:     %U.loc4_7.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc4_7.2 (constants.%U)]
// CHECK:STDOUT:     %x.param: @Id.%U.loc4_7.2 (%U) = value_param call_param0
// CHECK:STDOUT:     %U.ref.loc4_20: type = name_ref U, %U.loc4_7.1 [symbolic = %U.loc4_7.2 (constants.%U)]
// CHECK:STDOUT:     %x: @Id.%U.loc4_7.2 (%U) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Id.%U.loc4_7.2 (%U) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Id.%U.loc4_7.2 (%U) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc5_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_7.2: type = converted %.loc5_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc5_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc5_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_26.2: type = converted %.loc5_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc5_14: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc5_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: <error> = binding_pattern v [concrete]
// CHECK:STDOUT:     %.loc12_1: <error> = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref <error> = var v
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %.loc12_9: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %Id.ref: %Id.type = name_ref Id, %Id.decl [concrete = constants.%Id]
// CHECK:STDOUT:     %I.ref: type = name_ref I, %I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.ref: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %Id.specific_fn: <specific function> = specific_function %Id.ref, @Id(constants.%I.assoc_type) [concrete = constants.%Id.specific_fn]
// CHECK:STDOUT:     %Id.call: init %I.assoc_type = call %Id.specific_fn(%T.ref)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: <error> = bind_name v, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc5_7.2 as %.loc5_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Id(%U.loc4_7.1: type) {
// CHECK:STDOUT:   %U.loc4_7.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc4_7.2 (constants.%U)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.loc4_7.2 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.loc4_7.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Id.%U.loc4_7.2 (%U)) -> @Id.%U.loc4_7.2 (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Id.%U.loc4_7.2 (%U) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Id(constants.%U) {
// CHECK:STDOUT:   %U.loc4_7.2 => constants.%U
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Id(constants.%I.assoc_type) {
// CHECK:STDOUT:   %U.loc4_7.2 => constants.%I.assoc_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.40b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_const_associated_in_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %Id.type: type = fn_type @Id [concrete]
// CHECK:STDOUT:   %Id: %Id.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%T [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %Self.as_type, (%J.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %Id.specific_fn: <specific function> = specific_function %Id, @Id(type) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%F.decl [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Id = %Id.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Id.decl: %Id.type = fn_decl @Id [concrete = constants.%Id] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @Id.%pattern_type (%pattern_type.7dc) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @Id.%pattern_type (%pattern_type.7dc) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Id.%pattern_type (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Id.%pattern_type (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc3_26: type = name_ref U, %U.loc3_7.1 [symbolic = %U.loc3_7.2 (constants.%U)]
// CHECK:STDOUT:     %U.loc3_7.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc3_7.2 (constants.%U)]
// CHECK:STDOUT:     %x.param: @Id.%U.loc3_7.2 (%U) = value_param call_param0
// CHECK:STDOUT:     %U.ref.loc3_20: type = name_ref U, %U.loc3_7.1 [symbolic = %U.loc3_7.2 (constants.%U)]
// CHECK:STDOUT:     %x: @Id.%U.loc3_7.2 (%U) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Id.%U.loc3_7.2 (%U) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Id.%U.loc3_7.2 (%U) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%T [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Id.ref: %Id.type = name_ref Id, file.%Id.decl [concrete = constants.%Id]
// CHECK:STDOUT:     %impl.elem0.loc11_16.2: type = impl_witness_access constants.%J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_16.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %impl.elem0.loc11_16.2 [symbolic = %impl.elem0.loc11_16.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %Id.specific_fn: <specific function> = specific_function %Id.ref, @Id(type) [concrete = constants.%Id.specific_fn]
// CHECK:STDOUT:     %Id.call: init type = call %Id.specific_fn(%T.ref)
// CHECK:STDOUT:     %.loc11_17.1: type = value_of_initializer %Id.call
// CHECK:STDOUT:     %.loc11_17.2: type = converted %Id.call, %.loc11_17.1
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   .Id = <poisoned>
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%T, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Id(%U.loc3_7.1: type) {
// CHECK:STDOUT:   %U.loc3_7.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc3_7.2 (constants.%U)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.loc3_7.2 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.loc3_7.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Id.%U.loc3_7.2 (%U)) -> @Id.%U.loc3_7.2 (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Id.%U.loc3_7.2 (%U) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc11_16.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_16.1 (constants.%impl.elem0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Id(constants.%U) {
// CHECK:STDOUT:   %U.loc3_7.2 => constants.%U
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Id(type) {
// CHECK:STDOUT:   %U.loc3_7.2 => type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness
// CHECK:STDOUT:   %impl.elem0.loc11_16.1 => constants.%impl.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_alias_to_non_const_assoc_entity.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- to_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .U = %U
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %I.ref: type = name_ref I, %I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %T.ref: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %U: %I.assoc_type = bind_alias U, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_access_alias_in_imported_library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.f88: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.652 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.922: %J.assoc_type = assoc_entity element0, @J.%F.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I = import_ref Main//to_import, I, unloaded
// CHECK:STDOUT:   %Main.U: %I.assoc_type = import_ref Main//to_import, U, loaded [concrete = constants.%assoc0.f88]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//to_import, inst19 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.33a = import_ref Main//to_import, loc4_8, unloaded
// CHECK:STDOUT:   %Main.T = import_ref Main//to_import, T, unloaded
// CHECK:STDOUT:   %Main.import_ref.652: type = import_ref Main//to_import, loc4_8, loaded [concrete = %T]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.5dd: %I.type = import_ref Main//to_import, inst19 [no loc], loaded [symbolic = constants.%Self.826]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .U = imports.%Main.U
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %U.ref: %I.assoc_type = name_ref U, imports.%Main.U [concrete = constants.%assoc0.f88]
// CHECK:STDOUT:   %V: %I.assoc_type = bind_alias V, imports.%Main.U [concrete = constants.%assoc0.f88]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V.ref: <error> = name_ref V, <error> [concrete = <error>]
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.922]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = <poisoned>
// CHECK:STDOUT:   .V = %V
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "to_import.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .T = imports.%Main.import_ref.33a
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(imports.%Main.import_ref.5dd: %I.type) [from "to_import.carbon"] {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@J.%Self: %J.type) {
// CHECK:STDOUT:   fn() -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.ccd) {}
// CHECK:STDOUT:
