// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/no_prelude/call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/no_prelude/call.carbon

// --- explicit.carbon

library "[[@TEST_NAME]]";

fn Function(T:! type, x: T) -> T {
  return x;
}

fn CallGeneric(T:! type, x: T) -> T {
  return Function(T, x);
}

fn CallGenericPtr(T:! type, x: T*) -> T* {
  return Function(T*, x);
}

class C {}

fn CallSpecific(x: C) -> C {
  return Function(C, x);
}

// --- deduced.carbon

library "[[@TEST_NAME]]";

fn Function[T:! type](x: T) -> T {
  return x;
}

fn CallGeneric(T:! type, x: T) -> T {
  return Function(x);
}

fn CallGenericPtr(T:! type, x: T*) -> T* {
  return Function(x);
}

class C {}

fn CallSpecific(x: C) -> C {
  return Function(x);
}

// CHECK:STDOUT: --- explicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Function.type: type = fn_type @Function [concrete]
// CHECK:STDOUT:   %Function: %Function.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %Function.specific_fn.46f: <specific function> = specific_function %Function, @Function(%T) [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %CallGenericPtr.type: type = fn_type @CallGenericPtr [concrete]
// CHECK:STDOUT:   %CallGenericPtr: %CallGenericPtr.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %Function.specific_fn.4d7: <specific function> = specific_function %Function, @Function(%ptr.79f) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %CallSpecific.type: type = fn_type @CallSpecific [concrete]
// CHECK:STDOUT:   %CallSpecific: %CallSpecific.type = struct_value () [concrete]
// CHECK:STDOUT:   %Function.specific_fn.1b5: <specific function> = specific_function %Function, @Function(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Function = %Function.decl
// CHECK:STDOUT:     .CallGeneric = %CallGeneric.decl
// CHECK:STDOUT:     .CallGenericPtr = %CallGenericPtr.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .CallSpecific = %CallSpecific.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Function.decl: %Function.type = fn_decl @Function [concrete = constants.%Function] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @Function.%T.loc4_13.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Function.%T.loc4_13.2 (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @Function.%T.loc4_13.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Function.%T.loc4_13.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_32: type = name_ref T, %T.loc4_13.1 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Function.%T.loc4_13.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc4_26: type = name_ref T, %T.loc4_13.1 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %x: @Function.%T.loc4_13.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Function.%T.loc4_13.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Function.%T.loc4_13.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt.loc8_16.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @CallGeneric.%T.loc8_16.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @CallGeneric.%T.loc8_16.2 (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @CallGeneric.%T.loc8_16.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @CallGeneric.%T.loc8_16.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_35: type = name_ref T, %T.loc8_16.1 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc8_16.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @CallGeneric.%T.loc8_16.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc8_29: type = name_ref T, %T.loc8_16.1 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:     %x: @CallGeneric.%T.loc8_16.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @CallGeneric.%T.loc8_16.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @CallGeneric.%T.loc8_16.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericPtr.decl: %CallGenericPtr.type = fn_decl @CallGenericPtr [concrete = constants.%CallGenericPtr] {
// CHECK:STDOUT:     %T.patt.loc12_19.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_19.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12_39: type = name_ref T, %T.loc12_19.1 [symbolic = %T.loc12_19.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc12_40: type = ptr_type %T.ref.loc12_39 [symbolic = %ptr.loc12_33.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     %T.loc12_19.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_19.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = value_param call_param0
// CHECK:STDOUT:     %.loc12: type = splice_block %ptr.loc12_33.1 [symbolic = %ptr.loc12_33.2 (constants.%ptr.79f)] {
// CHECK:STDOUT:       %T.ref.loc12_32: type = name_ref T, %T.loc12_19.1 [symbolic = %T.loc12_19.2 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc12_33.1: type = ptr_type %T.ref.loc12_32 [symbolic = %ptr.loc12_33.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = out_param call_param1
// CHECK:STDOUT:     %return: ref @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %CallSpecific.decl: %CallSpecific.type = fn_decl @CallSpecific [concrete = constants.%CallSpecific] {
// CHECK:STDOUT:     %x.patt: %C = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc18_26: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %x.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc18_20: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %x: %C = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Function(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Function.%T.loc4_13.2 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc4_13.1: type, %x.param_patt: @Function.%T.loc4_13.2 (%T)) -> @Function.%T.loc4_13.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Function.%T.loc4_13.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc8_16.1: type) {
// CHECK:STDOUT:   %T.loc8_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_16.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_16.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @CallGeneric.%T.loc8_16.2 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Function.specific_fn.loc9_10.2: <specific function> = specific_function constants.%Function, @Function(%T.loc8_16.2) [symbolic = %Function.specific_fn.loc9_10.2 (constants.%Function.specific_fn.46f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc8_16.1: type, %x.param_patt: @CallGeneric.%T.loc8_16.2 (%T)) -> @CallGeneric.%T.loc8_16.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %T.ref.loc9: type = name_ref T, %T.loc8_16.1 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:     %x.ref: @CallGeneric.%T.loc8_16.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %Function.specific_fn.loc9_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%T) [symbolic = %Function.specific_fn.loc9_10.2 (constants.%Function.specific_fn.46f)]
// CHECK:STDOUT:     %Function.call: init @CallGeneric.%T.loc8_16.2 (%T) = call %Function.specific_fn.loc9_10.1(%x.ref)
// CHECK:STDOUT:     %.loc9_24.1: @CallGeneric.%T.loc8_16.2 (%T) = value_of_initializer %Function.call
// CHECK:STDOUT:     %.loc9_24.2: @CallGeneric.%T.loc8_16.2 (%T) = converted %Function.call, %.loc9_24.1
// CHECK:STDOUT:     return %.loc9_24.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericPtr(%T.loc12_19.1: type) {
// CHECK:STDOUT:   %T.loc12_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_19.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc12_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_19.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc12_33.2: type = ptr_type %T.loc12_19.2 [symbolic = %ptr.loc12_33.2 (constants.%ptr.79f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) [symbolic = %require_complete (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %Function.specific_fn.loc13_10.2: <specific function> = specific_function constants.%Function, @Function(%ptr.loc12_33.2) [symbolic = %Function.specific_fn.loc13_10.2 (constants.%Function.specific_fn.4d7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc12_19.1: type, %x.param_patt: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f)) -> @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %T.ref.loc13: type = name_ref T, %T.loc12_19.1 [symbolic = %T.loc12_19.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc13: type = ptr_type %T.ref.loc13 [symbolic = %ptr.loc12_33.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     %x.ref: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = name_ref x, %x
// CHECK:STDOUT:     %Function.specific_fn.loc13_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%ptr.79f) [symbolic = %Function.specific_fn.loc13_10.2 (constants.%Function.specific_fn.4d7)]
// CHECK:STDOUT:     %Function.call: init @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = call %Function.specific_fn.loc13_10.1(%x.ref)
// CHECK:STDOUT:     %.loc13_25.1: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = value_of_initializer %Function.call
// CHECK:STDOUT:     %.loc13_25.2: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = converted %Function.call, %.loc13_25.1
// CHECK:STDOUT:     return %.loc13_25.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallSpecific(%x.param_patt: %C) -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:   %C.ref.loc19: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %x.ref: %C = name_ref x, %x
// CHECK:STDOUT:   %Function.specific_fn: <specific function> = specific_function %Function.ref, @Function(constants.%C) [concrete = constants.%Function.specific_fn.1b5]
// CHECK:STDOUT:   %.loc18: ref %C = splice_block %return {}
// CHECK:STDOUT:   %Function.call: init %C = call %Function.specific_fn(%x.ref) to %.loc18
// CHECK:STDOUT:   return %Function.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T) {
// CHECK:STDOUT:   %T.loc8_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(@CallGeneric.%T.loc8_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericPtr(constants.%T) {
// CHECK:STDOUT:   %T.loc12_19.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc12_19.2 => constants.%T.patt
// CHECK:STDOUT:   %ptr.loc12_33.2 => constants.%ptr.79f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%ptr.79f) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%ptr.79f
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6e5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(@CallGenericPtr.%ptr.loc12_33.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%C) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Function.type: type = fn_type @Function [concrete]
// CHECK:STDOUT:   %Function: %Function.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %Function.specific_fn.46f: <specific function> = specific_function %Function, @Function(%T) [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %CallGenericPtr.type: type = fn_type @CallGenericPtr [concrete]
// CHECK:STDOUT:   %CallGenericPtr: %CallGenericPtr.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %Function.specific_fn.4d7: <specific function> = specific_function %Function, @Function(%ptr.79f) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %CallSpecific.type: type = fn_type @CallSpecific [concrete]
// CHECK:STDOUT:   %CallSpecific: %CallSpecific.type = struct_value () [concrete]
// CHECK:STDOUT:   %Function.specific_fn.1b5: <specific function> = specific_function %Function, @Function(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Function = %Function.decl
// CHECK:STDOUT:     .CallGeneric = %CallGeneric.decl
// CHECK:STDOUT:     .CallGenericPtr = %CallGenericPtr.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .CallSpecific = %CallSpecific.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Function.decl: %Function.type = fn_decl @Function [concrete = constants.%Function] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @Function.%T.loc4_13.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Function.%T.loc4_13.2 (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @Function.%T.loc4_13.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Function.%T.loc4_13.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_32: type = name_ref T, %T.loc4_13.1 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Function.%T.loc4_13.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc4_26: type = name_ref T, %T.loc4_13.1 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %x: @Function.%T.loc4_13.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Function.%T.loc4_13.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Function.%T.loc4_13.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt.loc8_16.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @CallGeneric.%T.loc8_16.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @CallGeneric.%T.loc8_16.2 (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @CallGeneric.%T.loc8_16.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @CallGeneric.%T.loc8_16.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_35: type = name_ref T, %T.loc8_16.1 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc8_16.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @CallGeneric.%T.loc8_16.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc8_29: type = name_ref T, %T.loc8_16.1 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:     %x: @CallGeneric.%T.loc8_16.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @CallGeneric.%T.loc8_16.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @CallGeneric.%T.loc8_16.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericPtr.decl: %CallGenericPtr.type = fn_decl @CallGenericPtr [concrete = constants.%CallGenericPtr] {
// CHECK:STDOUT:     %T.patt.loc12_19.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_19.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12_39: type = name_ref T, %T.loc12_19.1 [symbolic = %T.loc12_19.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc12_40: type = ptr_type %T.ref.loc12_39 [symbolic = %ptr.loc12_33.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     %T.loc12_19.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_19.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = value_param call_param0
// CHECK:STDOUT:     %.loc12: type = splice_block %ptr.loc12_33.1 [symbolic = %ptr.loc12_33.2 (constants.%ptr.79f)] {
// CHECK:STDOUT:       %T.ref.loc12_32: type = name_ref T, %T.loc12_19.1 [symbolic = %T.loc12_19.2 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc12_33.1: type = ptr_type %T.ref.loc12_32 [symbolic = %ptr.loc12_33.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = out_param call_param1
// CHECK:STDOUT:     %return: ref @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %CallSpecific.decl: %CallSpecific.type = fn_decl @CallSpecific [concrete = constants.%CallSpecific] {
// CHECK:STDOUT:     %x.patt: %C = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc18_26: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %x.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc18_20: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %x: %C = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Function(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Function.%T.loc4_13.2 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_13.1: type](%x.param_patt: @Function.%T.loc4_13.2 (%T)) -> @Function.%T.loc4_13.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Function.%T.loc4_13.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc8_16.1: type) {
// CHECK:STDOUT:   %T.loc8_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_16.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_16.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @CallGeneric.%T.loc8_16.2 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Function.specific_fn.loc9_10.2: <specific function> = specific_function constants.%Function, @Function(%T.loc8_16.2) [symbolic = %Function.specific_fn.loc9_10.2 (constants.%Function.specific_fn.46f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc8_16.1: type, %x.param_patt: @CallGeneric.%T.loc8_16.2 (%T)) -> @CallGeneric.%T.loc8_16.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %x.ref: @CallGeneric.%T.loc8_16.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %Function.specific_fn.loc9_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%T) [symbolic = %Function.specific_fn.loc9_10.2 (constants.%Function.specific_fn.46f)]
// CHECK:STDOUT:     %Function.call: init @CallGeneric.%T.loc8_16.2 (%T) = call %Function.specific_fn.loc9_10.1(%x.ref)
// CHECK:STDOUT:     %.loc9_21.1: @CallGeneric.%T.loc8_16.2 (%T) = value_of_initializer %Function.call
// CHECK:STDOUT:     %.loc9_21.2: @CallGeneric.%T.loc8_16.2 (%T) = converted %Function.call, %.loc9_21.1
// CHECK:STDOUT:     return %.loc9_21.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericPtr(%T.loc12_19.1: type) {
// CHECK:STDOUT:   %T.loc12_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_19.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc12_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_19.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc12_33.2: type = ptr_type %T.loc12_19.2 [symbolic = %ptr.loc12_33.2 (constants.%ptr.79f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) [symbolic = %require_complete (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %Function.specific_fn.loc13_10.2: <specific function> = specific_function constants.%Function, @Function(%ptr.loc12_33.2) [symbolic = %Function.specific_fn.loc13_10.2 (constants.%Function.specific_fn.4d7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc12_19.1: type, %x.param_patt: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f)) -> @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %x.ref: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = name_ref x, %x
// CHECK:STDOUT:     %Function.specific_fn.loc13_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%ptr.79f) [symbolic = %Function.specific_fn.loc13_10.2 (constants.%Function.specific_fn.4d7)]
// CHECK:STDOUT:     %Function.call: init @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = call %Function.specific_fn.loc13_10.1(%x.ref)
// CHECK:STDOUT:     %.loc13_21.1: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = value_of_initializer %Function.call
// CHECK:STDOUT:     %.loc13_21.2: @CallGenericPtr.%ptr.loc12_33.2 (%ptr.79f) = converted %Function.call, %.loc13_21.1
// CHECK:STDOUT:     return %.loc13_21.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallSpecific(%x.param_patt: %C) -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:   %x.ref: %C = name_ref x, %x
// CHECK:STDOUT:   %Function.specific_fn: <specific function> = specific_function %Function.ref, @Function(constants.%C) [concrete = constants.%Function.specific_fn.1b5]
// CHECK:STDOUT:   %.loc18: ref %C = splice_block %return {}
// CHECK:STDOUT:   %Function.call: init %C = call %Function.specific_fn(%x.ref) to %.loc18
// CHECK:STDOUT:   return %Function.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T) {
// CHECK:STDOUT:   %T.loc8_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(@CallGeneric.%T.loc8_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericPtr(constants.%T) {
// CHECK:STDOUT:   %T.loc12_19.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc12_19.2 => constants.%T.patt
// CHECK:STDOUT:   %ptr.loc12_33.2 => constants.%ptr.79f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%ptr.79f) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%ptr.79f
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6e5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(@CallGenericPtr.%ptr.loc12_33.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%C) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
