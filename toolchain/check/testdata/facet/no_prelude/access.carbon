// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/no_prelude/access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/no_prelude/access.carbon

// --- simple.carbon

library "[[@TEST_NAME]]";

interface I {
  fn DoIt();
}

fn Use(T:! I) {
  // CHECK:STDERR: simple.carbon:[[@LINE+7]]:3: error: cannot deduce value for generic parameter `Self` [DeductionIncomplete]
  // CHECK:STDERR:   T.DoIt();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: simple.carbon:[[@LINE-7]]:3: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:   fn DoIt();
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  T.DoIt();
}

// --- fail_todo_use_self.carbon

library "[[@TEST_NAME]]";

interface I {
  fn Make() -> Self;
}

fn Use(T:! I) -> T {
  // CHECK:STDERR: fail_todo_use_self.carbon:[[@LINE+7]]:10: error: cannot deduce value for generic parameter `Self` [DeductionIncomplete]
  // CHECK:STDERR:   return T.Make();
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR: fail_todo_use_self.carbon:[[@LINE-7]]:3: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:   fn Make() -> Self;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return T.Make();
}

// CHECK:STDOUT: --- simple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %DoIt.type: type = fn_type @DoIt [template]
// CHECK:STDOUT:   %DoIt: %DoIt.type = struct_value () [template]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [template]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%DoIt.decl [template]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %I.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [template]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [template]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %T.as_wit: <witness> = facet_access_witness %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %DoIt.type = impl_witness_access %T.as_wit, element0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [template = constants.%Use] {
// CHECK:STDOUT:     %T.patt.loc8_8.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_8.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: %I.type = value_param_pattern %T.patt.loc8_8.1, runtime_param<none> [symbolic = %T.patt.loc8_8.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: %I.type = value_param runtime_param<none>
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %T.loc8_8.1: %I.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %DoIt.decl: %DoIt.type = fn_decl @DoIt [template = constants.%DoIt] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %DoIt.decl [template = constants.%assoc0]
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
// CHECK:STDOUT:   %T.as_type.loc16_4.2: type = facet_access_type %T.loc8_8.2 [symbolic = %T.as_type.loc16_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %T.as_wit.loc16_4.2: <witness> = facet_access_witness %T.loc8_8.2 [symbolic = %T.as_wit.loc16_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:   %impl.elem0.loc16_4.2: %DoIt.type = impl_witness_access %T.as_wit.loc16_4.2, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %I.type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc8_8.1 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:     %DoIt.ref: %I.assoc_type = name_ref DoIt, @I.%assoc0 [template = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc16_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc16_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc16: type = converted %T.ref, %T.as_type.loc16_4.1 [symbolic = %T.as_type.loc16_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.loc16_4.1: <witness> = facet_access_witness %T.ref [symbolic = %T.as_wit.loc16_4.2 (constants.%T.as_wit)]
// CHECK:STDOUT:     %impl.elem0.loc16_4.1: %DoIt.type = impl_witness_access %T.as_wit.loc16_4.1, element0 [symbolic = %impl.elem0.loc16_4.2 (constants.%impl.elem0)]
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
// CHECK:STDOUT: --- fail_todo_use_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [template]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%Make.decl [template]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %I.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [template]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %T.as_wit: <witness> = facet_access_witness %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %Make.type = impl_witness_access %T.as_wit, element0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [template = constants.%Use] {
// CHECK:STDOUT:     %T.patt.loc8_8.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_8.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: %I.type = value_param_pattern %T.patt.loc8_8.1, runtime_param<none> [symbolic = %T.patt.loc8_8.2 (constants.%T.patt)]
// CHECK:STDOUT:     %return.patt: @Use.%T.as_type.loc8_18.2 (%T.as_type) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Use.%T.as_type.loc8_18.2 (%T.as_type) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8: %I.type = name_ref T, %T.loc8_8.1 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc8_18.1: type = facet_access_type %T.ref.loc8 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc8: type = converted %T.ref.loc8, %T.as_type.loc8_18.1 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.param: %I.type = value_param runtime_param<none>
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %T.loc8_8.1: %I.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:     %return.param: ref @Use.%T.as_type.loc8_18.2 (%T.as_type) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @Use.%T.as_type.loc8_18.2 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [template = constants.%Make] {
// CHECK:STDOUT:     %return.patt: @Make.%Self.as_type.loc5_16.1 (%Self.as_type) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Make.%Self.as_type.loc5_16.1 (%Self.as_type) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc5_16.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc5: type = converted %Self.ref, %Self.as_type.loc5_16.2 [symbolic = %Self.as_type.loc5_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %return.param: ref @Make.%Self.as_type.loc5_16.1 (%Self.as_type) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @Make.%Self.as_type.loc5_16.1 (%Self.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %Make.decl [template = constants.%assoc0]
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
// CHECK:STDOUT:   %T.as_wit.loc16_11.2: <witness> = facet_access_witness %T.loc8_8.2 [symbolic = %T.as_wit.loc16_11.2 (constants.%T.as_wit)]
// CHECK:STDOUT:   %impl.elem0.loc16_11.2: %Make.type = impl_witness_access %T.as_wit.loc16_11.2, element0 [symbolic = %impl.elem0.loc16_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: %I.type) -> @Use.%T.as_type.loc8_18.2 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc16: %I.type = name_ref T, %T.loc8_8.1 [symbolic = %T.loc8_8.2 (constants.%T)]
// CHECK:STDOUT:     %Make.ref: %I.assoc_type = name_ref Make, @I.%assoc0 [template = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc16: type = facet_access_type %T.ref.loc16 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc16: type = converted %T.ref.loc16, %T.as_type.loc16 [symbolic = %T.as_type.loc8_18.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_wit.loc16_11.1: <witness> = facet_access_witness %T.ref.loc16 [symbolic = %T.as_wit.loc16_11.2 (constants.%T.as_wit)]
// CHECK:STDOUT:     %impl.elem0.loc16_11.1: %Make.type = impl_witness_access %T.as_wit.loc16_11.1, element0 [symbolic = %impl.elem0.loc16_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     return <error>
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
