// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/no_prelude/access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/no_prelude/access.carbon

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

// CHECK:STDOUT: --- access_assoc_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %DoIt.type: type = fn_type @DoIt [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DoIt: %DoIt.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%DoIt.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %I.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %T.as_wit: <witness> = facet_access_witness %T [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type, %T.as_wit [symbolic]
// CHECK:STDOUT:   %.d9e: type = fn_type_with_self_type %DoIt.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.d9e = impl_witness_access %T.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @DoIt(%I.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {
// CHECK:STDOUT:     %T.patt.loc8_8.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_8.2 (constants.%T.patt)]
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
// CHECK:STDOUT:   %T.patt.loc8_8.2: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_8.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc9_4.2: type = facet_access_type %T.loc8_8.2 [symbolic = %T.as_type.loc9_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %T.as_wit.loc9_4.2: <witness> = facet_access_witness %T.loc8_8.2 [symbolic = %T.as_wit.loc9_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type.loc9_4.2, %T.as_wit.loc9_4.2 [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc9_4.2: type = fn_type_with_self_type constants.%DoIt.type, %I.facet [symbolic = %.loc9_4.2 (constants.%.d9e)]
// CHECK:STDOUT:   %impl.elem0.loc9_4.2: @Use.%.loc9_4.2 (%.d9e) = impl_witness_access %T.as_wit.loc9_4.2, element0 [symbolic = %impl.elem0.loc9_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_fn.loc9_4.2: <specific function> = specific_function %impl.elem0.loc9_4.2, @DoIt(%I.facet) [symbolic = %specific_fn.loc9_4.2 (constants.%specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc8_8.1: %I.type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc8_8.1 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:     %DoIt.ref: %I.assoc_type = name_ref DoIt, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc9_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_4.1: type = converted %T.ref, %T.as_type.loc9_4.1 [symbolic = %T.as_type.loc9_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.loc9_4.1: <witness> = facet_access_witness %T.ref [symbolic = %T.as_wit.loc9_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:     %impl.elem0.loc9_4.1: @Use.%.loc9_4.2 (%.d9e) = impl_witness_access %T.as_wit.loc9_4.1, element0 [symbolic = %impl.elem0.loc9_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_fn.loc9_4.1: <specific function> = specific_function %impl.elem0.loc9_4.1, @DoIt(constants.%I.facet) [symbolic = %specific_fn.loc9_4.2 (constants.%specific_fn)]
// CHECK:STDOUT:     %DoIt.call: init %empty_tuple.type = call %specific_fn.loc9_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DoIt(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc8_8.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_8.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DoIt(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DoIt(@Use.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- assoc_fn_using_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%Make.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %I.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %T.as_wit: <witness> = facet_access_witness %T [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type, %T.as_wit [symbolic]
// CHECK:STDOUT:   %.aaf: type = fn_type_with_self_type %Make.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.aaf = impl_witness_access %T.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Make(%I.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {
// CHECK:STDOUT:     %T.patt.loc8_8.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_8.2 (constants.%T.patt)]
// CHECK:STDOUT:     %return.patt: @Use.%T.as_type.loc8_18.2 (%T.as_type) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Use.%T.as_type.loc8_18.2 (%T.as_type) = out_param_pattern %return.patt, call_param0
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
// CHECK:STDOUT:     %return.patt: @Make.%Self.as_type.loc5_16.1 (%Self.as_type) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Make.%Self.as_type.loc5_16.1 (%Self.as_type) = out_param_pattern %return.patt, call_param0
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
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Make.%Self.as_type.loc5_16.1 (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_8.1: %I.type) {
// CHECK:STDOUT:   %T.loc8_8.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_8.2: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_8.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc8_18.2: type = facet_access_type %T.loc8_8.2 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Use.%T.as_type.loc8_18.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %T.as_wit.loc9_11.2: <witness> = facet_access_witness %T.loc8_8.2 [symbolic = %T.as_wit.loc9_11.2 (constants.%T.as_wit)]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type.loc8_18.2, %T.as_wit.loc9_11.2 [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc9_11.2: type = fn_type_with_self_type constants.%Make.type, %I.facet [symbolic = %.loc9_11.2 (constants.%.aaf)]
// CHECK:STDOUT:   %impl.elem0.loc9_11.2: @Use.%.loc9_11.2 (%.aaf) = impl_witness_access %T.as_wit.loc9_11.2, element0 [symbolic = %impl.elem0.loc9_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_fn.loc9_11.2: <specific function> = specific_function %impl.elem0.loc9_11.2, @Make(%I.facet) [symbolic = %specific_fn.loc9_11.2 (constants.%specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc8_8.1: %I.type) -> @Use.%T.as_type.loc8_18.2 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc9: %I.type = name_ref T, %T.loc8_8.1 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:     %Make.ref: %I.assoc_type = name_ref Make, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc9: type = facet_access_type %T.ref.loc9 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_11.1: type = converted %T.ref.loc9, %T.as_type.loc9 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.loc9_11.1: <witness> = facet_access_witness %T.ref.loc9 [symbolic = %T.as_wit.loc9_11.2 (constants.%T.as_wit)]
// CHECK:STDOUT:     %impl.elem0.loc9_11.1: @Use.%.loc9_11.2 (%.aaf) = impl_witness_access %T.as_wit.loc9_11.1, element0 [symbolic = %impl.elem0.loc9_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_fn.loc9_11.1: <specific function> = specific_function %impl.elem0.loc9_11.1, @Make(constants.%I.facet) [symbolic = %specific_fn.loc9_11.2 (constants.%specific_fn)]
// CHECK:STDOUT:     %Make.call: init @Use.%T.as_type.loc8_18.2 (%T.as_type) = call %specific_fn.loc9_11.1()
// CHECK:STDOUT:     %.loc9_18.1: @Use.%T.as_type.loc8_18.2 (%T.as_type) = value_of_initializer %Make.call
// CHECK:STDOUT:     %.loc9_18.2: @Use.%T.as_type.loc8_18.2 (%T.as_type) = converted %Make.call, %.loc9_18.1
// CHECK:STDOUT:     return %.loc9_18.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_16.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc8_8.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_8.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc8_18.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc5_16.1 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(@Use.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_assoc_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Copy.type: type = fn_type @Copy [concrete]
// CHECK:STDOUT:   %Copy: %Copy.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%Copy.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %I.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %T.as_wit: <witness> = facet_access_witness %T [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type, %T.as_wit [symbolic]
// CHECK:STDOUT:   %.4ef: type = fn_type_with_self_type %Copy.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.4ef = impl_witness_access %T.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Copy(%I.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {
// CHECK:STDOUT:     %T.patt.loc8_8.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_8.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @Use.%T.as_type.loc8_18.2 (%T.as_type) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Use.%T.as_type.loc8_18.2 (%T.as_type) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @Use.%T.as_type.loc8_18.2 (%T.as_type) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Use.%T.as_type.loc8_18.2 (%T.as_type) = out_param_pattern %return.patt, call_param1
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
// CHECK:STDOUT:     %self.patt: @Copy.%Self.as_type.loc5_17.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Copy.%Self.as_type.loc5_17.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: @Copy.%Self.as_type.loc5_17.1 (%Self.as_type) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Copy.%Self.as_type.loc5_17.1 (%Self.as_type) = out_param_pattern %return.patt, call_param1
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
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Copy.%Self.as_type.loc5_17.1 (%Self.as_type)]() -> @Copy.%Self.as_type.loc5_17.1 (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Use(%T.loc8_8.1: %I.type) {
// CHECK:STDOUT:   %T.loc8_8.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_8.2: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_8.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc8_18.2: type = facet_access_type %T.loc8_8.2 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Use.%T.as_type.loc8_18.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %T.as_wit.loc9_11.2: <witness> = facet_access_witness %T.loc8_8.2 [symbolic = %T.as_wit.loc9_11.2 (constants.%T.as_wit)]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type.loc8_18.2, %T.as_wit.loc9_11.2 [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc9_11.2: type = fn_type_with_self_type constants.%Copy.type, %I.facet [symbolic = %.loc9_11.2 (constants.%.4ef)]
// CHECK:STDOUT:   %impl.elem0.loc9_11.2: @Use.%.loc9_11.2 (%.4ef) = impl_witness_access %T.as_wit.loc9_11.2, element0 [symbolic = %impl.elem0.loc9_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_fn.loc9_11.2: <specific function> = specific_function %impl.elem0.loc9_11.2, @Copy(%I.facet) [symbolic = %specific_fn.loc9_11.2 (constants.%specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc8_8.1: %I.type](%x.param_patt: @Use.%T.as_type.loc8_18.2 (%T.as_type)) -> @Use.%T.as_type.loc8_18.2 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Use.%T.as_type.loc8_18.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Copy.ref: %I.assoc_type = name_ref Copy, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc9: type = facet_access_type constants.%T [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_11.1: type = converted constants.%T, %T.as_type.loc9 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.loc9_11.1: <witness> = facet_access_witness constants.%T [symbolic = %T.as_wit.loc9_11.2 (constants.%T.as_wit)]
// CHECK:STDOUT:     %impl.elem0.loc9_11.1: @Use.%.loc9_11.2 (%.4ef) = impl_witness_access %T.as_wit.loc9_11.1, element0 [symbolic = %impl.elem0.loc9_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc9_11: <bound method> = bound_method %x.ref, %impl.elem0.loc9_11.1
// CHECK:STDOUT:     %specific_fn.loc9_11.1: <specific function> = specific_function %impl.elem0.loc9_11.1, @Copy(constants.%I.facet) [symbolic = %specific_fn.loc9_11.2 (constants.%specific_fn)]
// CHECK:STDOUT:     %bound_method.loc9_17: <bound method> = bound_method %x.ref, %specific_fn.loc9_11.1
// CHECK:STDOUT:     %Copy.call: init @Use.%T.as_type.loc8_18.2 (%T.as_type) = call %bound_method.loc9_17(%x.ref)
// CHECK:STDOUT:     %.loc9_18.1: @Use.%T.as_type.loc8_18.2 (%T.as_type) = value_of_initializer %Copy.call
// CHECK:STDOUT:     %.loc9_18.2: @Use.%T.as_type.loc8_18.2 (%T.as_type) = converted %Copy.call, %.loc9_18.1
// CHECK:STDOUT:     return %.loc9_18.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Copy(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_17.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc8_8.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_8.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc8_18.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Copy(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc5_17.1 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Copy(@Use.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_selfless_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Hello.type: type = fn_type @Hello [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Hello: %Hello.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%Hello.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %I.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %T.as_wit: <witness> = facet_access_witness %T [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type, %T.as_wit [symbolic]
// CHECK:STDOUT:   %.33b: type = fn_type_with_self_type %Hello.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.33b = impl_witness_access %T.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Hello(%I.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {
// CHECK:STDOUT:     %T.patt.loc8_8.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_8.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @Use.%T.as_type.loc8_18.2 (%T.as_type) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Use.%T.as_type.loc8_18.2 (%T.as_type) = value_param_pattern %x.patt, call_param0
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
// CHECK:STDOUT:   %T.patt.loc8_8.2: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_8.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc8_18.2: type = facet_access_type %T.loc8_8.2 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Use.%T.as_type.loc8_18.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %T.as_wit.loc9_4.2: <witness> = facet_access_witness %T.loc8_8.2 [symbolic = %T.as_wit.loc9_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type.loc8_18.2, %T.as_wit.loc9_4.2 [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc9_4.2: type = fn_type_with_self_type constants.%Hello.type, %I.facet [symbolic = %.loc9_4.2 (constants.%.33b)]
// CHECK:STDOUT:   %impl.elem0.loc9_4.2: @Use.%.loc9_4.2 (%.33b) = impl_witness_access %T.as_wit.loc9_4.2, element0 [symbolic = %impl.elem0.loc9_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_fn.loc9_4.2: <specific function> = specific_function %impl.elem0.loc9_4.2, @Hello(%I.facet) [symbolic = %specific_fn.loc9_4.2 (constants.%specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc8_8.1: %I.type](%x.param_patt: @Use.%T.as_type.loc8_18.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Use.%T.as_type.loc8_18.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Hello.ref: %I.assoc_type = name_ref Hello, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc9: type = facet_access_type constants.%T [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_4.1: type = converted constants.%T, %T.as_type.loc9 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.loc9_4.1: <witness> = facet_access_witness constants.%T [symbolic = %T.as_wit.loc9_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:     %impl.elem0.loc9_4.1: @Use.%.loc9_4.2 (%.33b) = impl_witness_access %T.as_wit.loc9_4.1, element0 [symbolic = %impl.elem0.loc9_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_fn.loc9_4.1: <specific function> = specific_function %impl.elem0.loc9_4.1, @Hello(constants.%I.facet) [symbolic = %specific_fn.loc9_4.2 (constants.%specific_fn)]
// CHECK:STDOUT:     %Hello.call: init %empty_tuple.type = call %specific_fn.loc9_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Hello(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Use(constants.%T) {
// CHECK:STDOUT:   %T.loc8_8.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_8.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc8_18.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Hello(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Hello(@Use.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- access_assoc_method_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Copy.type: type = fn_type @Copy [concrete]
// CHECK:STDOUT:   %Copy: %Copy.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%Copy.decl [concrete]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %I.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %UseIndirect.type: type = fn_type @UseIndirect [concrete]
// CHECK:STDOUT:   %UseIndirect: %UseIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %T.as_wit: <witness> = facet_access_witness %T [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type, %T.as_wit [symbolic]
// CHECK:STDOUT:   %.4ef: type = fn_type_with_self_type %Copy.type, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.4ef = impl_witness_access %T.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Copy(%I.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .UseIndirect = %UseIndirect.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %UseIndirect.decl: %UseIndirect.type = fn_decl @UseIndirect [concrete = constants.%UseIndirect] {
// CHECK:STDOUT:     %T.patt.loc8_16.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = out_param_pattern %return.patt, call_param1
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
// CHECK:STDOUT:     %self.patt: @Copy.%Self.as_type.loc5_17.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Copy.%Self.as_type.loc5_17.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: @Copy.%Self.as_type.loc5_17.1 (%Self.as_type) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Copy.%Self.as_type.loc5_17.1 (%Self.as_type) = out_param_pattern %return.patt, call_param1
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
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Copy.%Self.as_type.loc5_17.1 (%Self.as_type)]() -> @Copy.%Self.as_type.loc5_17.1 (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @UseIndirect(%T.loc8_16.1: %I.type) {
// CHECK:STDOUT:   %T.loc8_16.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_16.2: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc8_26.2: type = facet_access_type %T.loc8_16.2 [symbolic = %T.as_type.loc8_26.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %T.as_wit.loc9_14.2: <witness> = facet_access_witness %T.loc8_16.2 [symbolic = %T.as_wit.loc9_14.2 (constants.%T.as_wit)]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type.loc8_26.2, %T.as_wit.loc9_14.2 [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc9_14.2: type = fn_type_with_self_type constants.%Copy.type, %I.facet [symbolic = %.loc9_14.2 (constants.%.4ef)]
// CHECK:STDOUT:   %impl.elem0.loc9_14.2: @UseIndirect.%.loc9_14.2 (%.4ef) = impl_witness_access %T.as_wit.loc9_14.2, element0 [symbolic = %impl.elem0.loc9_14.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_fn.loc9_14.2: <specific function> = specific_function %impl.elem0.loc9_14.2, @Copy(%I.facet) [symbolic = %specific_fn.loc9_14.2 (constants.%specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc8_16.1: %I.type](%x.param_patt: @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type)) -> @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %T.ref.loc9: %I.type = name_ref T, %T.loc8_16.1 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:     %Copy.ref: %I.assoc_type = name_ref Copy, @I.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc9: type = facet_access_type %T.ref.loc9 [symbolic = %T.as_type.loc8_26.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_14.1: type = converted %T.ref.loc9, %T.as_type.loc9 [symbolic = %T.as_type.loc8_26.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.loc9_14.1: <witness> = facet_access_witness %T.ref.loc9 [symbolic = %T.as_wit.loc9_14.2 (constants.%T.as_wit)]
// CHECK:STDOUT:     %impl.elem0.loc9_14.1: @UseIndirect.%.loc9_14.2 (%.4ef) = impl_witness_access %T.as_wit.loc9_14.1, element0 [symbolic = %impl.elem0.loc9_14.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc9_11: <bound method> = bound_method %x.ref, %impl.elem0.loc9_14.1
// CHECK:STDOUT:     %specific_fn.loc9_14.1: <specific function> = specific_function %impl.elem0.loc9_14.1, @Copy(constants.%I.facet) [symbolic = %specific_fn.loc9_14.2 (constants.%specific_fn)]
// CHECK:STDOUT:     %bound_method.loc9_21: <bound method> = bound_method %x.ref, %specific_fn.loc9_14.1
// CHECK:STDOUT:     %Copy.call: init @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = call %bound_method.loc9_21(%x.ref)
// CHECK:STDOUT:     %.loc9_22.1: @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = value_of_initializer %Copy.call
// CHECK:STDOUT:     %.loc9_22.2: @UseIndirect.%T.as_type.loc8_26.2 (%T.as_type) = converted %Copy.call, %.loc9_22.1
// CHECK:STDOUT:     return %.loc9_22.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Copy(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_17.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseIndirect(constants.%T) {
// CHECK:STDOUT:   %T.loc8_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc8_26.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Copy(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc5_17.1 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Copy(@UseIndirect.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_const_associated.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %Id.type: type = fn_type @Id [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Id: %Id.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic_self]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %.Self.as_type, %.Self.as_wit [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Id.specific_fn: <specific function> = specific_function %Id, @Id(%I.assoc_type) [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %I.assoc_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .Id = %Id.decl
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %Id.decl: %Id.type = fn_decl @Id [concrete = constants.%Id] {
// CHECK:STDOUT:     %U.patt.loc4_7.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc4_7.2 (constants.%U.patt)]
// CHECK:STDOUT:     %x.patt: @Id.%U.loc4_7.2 (%U) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Id.%U.loc4_7.2 (%U) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @Id.%U.loc4_7.2 (%U) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Id.%U.loc4_7.2 (%U) = out_param_pattern %return.patt, call_param1
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
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc5_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_26.2: type = converted %.loc5_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc5_14: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc5_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_tuple.type) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: <error> = binding_pattern v
// CHECK:STDOUT:     %.loc12_1: <error> = var_pattern %v.patt
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
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Id(%U.loc4_7.1: type) {
// CHECK:STDOUT:   %U.loc4_7.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc4_7.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc4_7.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc4_7.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Id.%U.loc4_7.2 (%U) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%U.patt.loc4_7.1: type](%x.param_patt: @Id.%U.loc4_7.2 (%U)) -> @Id.%U.loc4_7.2 (%U) {
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
// CHECK:STDOUT:   %U.patt.loc4_7.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Id(constants.%I.assoc_type) {
// CHECK:STDOUT:   %U.loc4_7.2 => constants.%I.assoc_type
// CHECK:STDOUT:   %U.patt.loc4_7.2 => constants.%I.assoc_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
