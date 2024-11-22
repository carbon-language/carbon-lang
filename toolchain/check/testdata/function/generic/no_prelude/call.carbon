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
// CHECK:STDOUT:   %Function.type: type = fn_type @Function [template]
// CHECK:STDOUT:   %Function: %Function.type = struct_value () [template]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [template]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [template]
// CHECK:STDOUT:   %.1: <specific function> = specific_function %Function, @Function(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %CallGenericPtr.type: type = fn_type @CallGenericPtr [template]
// CHECK:STDOUT:   %CallGenericPtr: %CallGenericPtr.type = struct_value () [template]
// CHECK:STDOUT:   %.3: <specific function> = specific_function %Function, @Function(%.2) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.4: type = struct_type {} [template]
// CHECK:STDOUT:   %.5: <witness> = complete_type_witness %.4 [template]
// CHECK:STDOUT:   %CallSpecific.type: type = fn_type @CallSpecific [template]
// CHECK:STDOUT:   %CallSpecific: %CallSpecific.type = struct_value () [template]
// CHECK:STDOUT:   %.7: <specific function> = specific_function %Function, @Function(%C) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Function = %Function.decl
// CHECK:STDOUT:     .CallGeneric = %CallGeneric.decl
// CHECK:STDOUT:     .CallGenericPtr = %CallGenericPtr.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .CallSpecific = %CallSpecific.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Function.decl: %Function.type = fn_decl @Function [template = constants.%Function] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_13.1, runtime_param<invalid> [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @Function.%T.loc4_13.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Function.%T.loc4_13.2 (%T) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @Function.%T.loc4_13.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Function.%T.loc4_13.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_26: type = name_ref T, %T.loc4_13.1 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc4_32: type = name_ref T, %T.loc4_13.1 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Function.%T.loc4_13.2 (%T) = value_param runtime_param0
// CHECK:STDOUT:     %x: @Function.%T.loc4_13.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Function.%T.loc4_13.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @Function.%T.loc4_13.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [template = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt.loc8_16.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_16.1, runtime_param<invalid> [symbolic = %T.patt.loc8_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @CallGeneric.%T.loc8_16.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @CallGeneric.%T.loc8_16.2 (%T) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @CallGeneric.%T.loc8_16.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @CallGeneric.%T.loc8_16.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_29: type = name_ref T, %T.loc8_16.1 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc8_35: type = name_ref T, %T.loc8_16.1 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_16.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @CallGeneric.%T.loc8_16.2 (%T) = value_param runtime_param0
// CHECK:STDOUT:     %x: @CallGeneric.%T.loc8_16.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @CallGeneric.%T.loc8_16.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @CallGeneric.%T.loc8_16.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericPtr.decl: %CallGenericPtr.type = fn_decl @CallGenericPtr [template = constants.%CallGenericPtr] {
// CHECK:STDOUT:     %T.patt.loc12_19.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_19.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc12_19.1, runtime_param<invalid> [symbolic = %T.patt.loc12_19.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @CallGenericPtr.%.loc12_33.2 (%.2) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @CallGenericPtr.%.loc12_33.2 (%.2) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @CallGenericPtr.%.loc12_33.2 (%.2) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @CallGenericPtr.%.loc12_33.2 (%.2) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12_32: type = name_ref T, %T.loc12_19.1 [symbolic = %T.loc12_19.2 (constants.%T)]
// CHECK:STDOUT:     %.loc12_33.1: type = ptr_type %T [symbolic = %.loc12_33.2 (constants.%.2)]
// CHECK:STDOUT:     %T.ref.loc12_39: type = name_ref T, %T.loc12_19.1 [symbolic = %T.loc12_19.2 (constants.%T)]
// CHECK:STDOUT:     %.loc12_40: type = ptr_type %T [symbolic = %.loc12_33.2 (constants.%.2)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc12_19.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc12_19.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @CallGenericPtr.%.loc12_33.2 (%.2) = value_param runtime_param0
// CHECK:STDOUT:     %x: @CallGenericPtr.%.loc12_33.2 (%.2) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @CallGenericPtr.%.loc12_33.2 (%.2) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @CallGenericPtr.%.loc12_33.2 (%.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %CallSpecific.decl: %CallSpecific.type = fn_decl @CallSpecific [template = constants.%CallSpecific] {
// CHECK:STDOUT:     %x.patt: %C = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc18_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc18_26: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %x.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %x: %C = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc16: <witness> = complete_type_witness %.4 [template = constants.%.5]
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
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type, %x.param_patt: @Function.%T.loc4_13.2 (%T)) -> @Function.%T.loc4_13.2 (%T) {
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
// CHECK:STDOUT:   %.loc9_10.2: <specific function> = specific_function constants.%Function, @Function(%T.loc8_16.2) [symbolic = %.loc9_10.2 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type, %x.param_patt: @CallGeneric.%T.loc8_16.2 (%T)) -> @CallGeneric.%T.loc8_16.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [template = constants.%Function]
// CHECK:STDOUT:     %T.ref.loc9: type = name_ref T, %T.loc8_16.1 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:     %x.ref: @CallGeneric.%T.loc8_16.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc9_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%T) [symbolic = %.loc9_10.2 (constants.%.1)]
// CHECK:STDOUT:     %Function.call: init @CallGeneric.%T.loc8_16.2 (%T) = call %.loc9_10.1(%x.ref)
// CHECK:STDOUT:     %.loc9_24.1: @CallGeneric.%T.loc8_16.2 (%T) = value_of_initializer %Function.call
// CHECK:STDOUT:     %.loc9_24.2: @CallGeneric.%T.loc8_16.2 (%T) = converted %Function.call, %.loc9_24.1
// CHECK:STDOUT:     return %.loc9_24.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericPtr(%T.loc12_19.1: type) {
// CHECK:STDOUT:   %T.loc12_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_19.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc12_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_19.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc12_33.2: type = ptr_type @CallGenericPtr.%T.loc12_19.2 (%T) [symbolic = %.loc12_33.2 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc13_10.2: <specific function> = specific_function constants.%Function, @Function(%.loc12_33.2) [symbolic = %.loc13_10.2 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type, %x.param_patt: @CallGenericPtr.%.loc12_33.2 (%.2)) -> @CallGenericPtr.%.loc12_33.2 (%.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [template = constants.%Function]
// CHECK:STDOUT:     %T.ref.loc13: type = name_ref T, %T.loc12_19.1 [symbolic = %T.loc12_19.2 (constants.%T)]
// CHECK:STDOUT:     %.loc13_20: type = ptr_type %T [symbolic = %.loc12_33.2 (constants.%.2)]
// CHECK:STDOUT:     %x.ref: @CallGenericPtr.%.loc12_33.2 (%.2) = name_ref x, %x
// CHECK:STDOUT:     %.loc13_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%.2) [symbolic = %.loc13_10.2 (constants.%.3)]
// CHECK:STDOUT:     %Function.call: init @CallGenericPtr.%.loc12_33.2 (%.2) = call %.loc13_10.1(%x.ref)
// CHECK:STDOUT:     %.loc13_25.1: @CallGenericPtr.%.loc12_33.2 (%.2) = value_of_initializer %Function.call
// CHECK:STDOUT:     %.loc13_25.2: @CallGenericPtr.%.loc12_33.2 (%.2) = converted %Function.call, %.loc13_25.1
// CHECK:STDOUT:     return %.loc13_25.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallSpecific(%x.param_patt: %C) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Function.ref: %Function.type = name_ref Function, file.%Function.decl [template = constants.%Function]
// CHECK:STDOUT:   %C.ref.loc19: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %x.ref: %C = name_ref x, %x
// CHECK:STDOUT:   %.loc19: <specific function> = specific_function %Function.ref, @Function(constants.%C) [template = constants.%.7]
// CHECK:STDOUT:   %.loc18_23.2: ref %C = splice_block %return {}
// CHECK:STDOUT:   %Function.call: init %C = call %.loc19(%x.ref) to %.loc18_23.2
// CHECK:STDOUT:   return %Function.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T) {
// CHECK:STDOUT:   %T.loc8_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(@CallGeneric.%T.loc8_16.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericPtr(constants.%T) {
// CHECK:STDOUT:   %T.loc12_19.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc12_19.2 => constants.%T
// CHECK:STDOUT:   %.loc12_33.2 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%.2
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(@CallGenericPtr.%.loc12_33.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%.2
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%C) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Function.type: type = fn_type @Function [template]
// CHECK:STDOUT:   %Function: %Function.type = struct_value () [template]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [template]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [template]
// CHECK:STDOUT:   %.1: <specific function> = specific_function %Function, @Function(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %CallGenericPtr.type: type = fn_type @CallGenericPtr [template]
// CHECK:STDOUT:   %CallGenericPtr: %CallGenericPtr.type = struct_value () [template]
// CHECK:STDOUT:   %.3: <specific function> = specific_function %Function, @Function(%.2) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.4: type = struct_type {} [template]
// CHECK:STDOUT:   %.5: <witness> = complete_type_witness %.4 [template]
// CHECK:STDOUT:   %CallSpecific.type: type = fn_type @CallSpecific [template]
// CHECK:STDOUT:   %CallSpecific: %CallSpecific.type = struct_value () [template]
// CHECK:STDOUT:   %.7: <specific function> = specific_function %Function, @Function(%C) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Function = %Function.decl
// CHECK:STDOUT:     .CallGeneric = %CallGeneric.decl
// CHECK:STDOUT:     .CallGenericPtr = %CallGenericPtr.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .CallSpecific = %CallSpecific.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Function.decl: %Function.type = fn_decl @Function [template = constants.%Function] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_13.1, runtime_param<invalid> [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @Function.%T.loc4_13.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Function.%T.loc4_13.2 (%T) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @Function.%T.loc4_13.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Function.%T.loc4_13.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_26: type = name_ref T, %T.loc4_13.1 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc4_32: type = name_ref T, %T.loc4_13.1 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Function.%T.loc4_13.2 (%T) = value_param runtime_param0
// CHECK:STDOUT:     %x: @Function.%T.loc4_13.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Function.%T.loc4_13.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @Function.%T.loc4_13.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [template = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt.loc8_16.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_16.1, runtime_param<invalid> [symbolic = %T.patt.loc8_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @CallGeneric.%T.loc8_16.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @CallGeneric.%T.loc8_16.2 (%T) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @CallGeneric.%T.loc8_16.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @CallGeneric.%T.loc8_16.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_29: type = name_ref T, %T.loc8_16.1 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc8_35: type = name_ref T, %T.loc8_16.1 [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_16.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_16.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @CallGeneric.%T.loc8_16.2 (%T) = value_param runtime_param0
// CHECK:STDOUT:     %x: @CallGeneric.%T.loc8_16.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @CallGeneric.%T.loc8_16.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @CallGeneric.%T.loc8_16.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericPtr.decl: %CallGenericPtr.type = fn_decl @CallGenericPtr [template = constants.%CallGenericPtr] {
// CHECK:STDOUT:     %T.patt.loc12_19.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_19.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc12_19.1, runtime_param<invalid> [symbolic = %T.patt.loc12_19.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @CallGenericPtr.%.loc12_33.2 (%.2) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @CallGenericPtr.%.loc12_33.2 (%.2) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @CallGenericPtr.%.loc12_33.2 (%.2) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @CallGenericPtr.%.loc12_33.2 (%.2) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12_32: type = name_ref T, %T.loc12_19.1 [symbolic = %T.loc12_19.2 (constants.%T)]
// CHECK:STDOUT:     %.loc12_33.1: type = ptr_type %T [symbolic = %.loc12_33.2 (constants.%.2)]
// CHECK:STDOUT:     %T.ref.loc12_39: type = name_ref T, %T.loc12_19.1 [symbolic = %T.loc12_19.2 (constants.%T)]
// CHECK:STDOUT:     %.loc12_40: type = ptr_type %T [symbolic = %.loc12_33.2 (constants.%.2)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc12_19.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc12_19.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @CallGenericPtr.%.loc12_33.2 (%.2) = value_param runtime_param0
// CHECK:STDOUT:     %x: @CallGenericPtr.%.loc12_33.2 (%.2) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @CallGenericPtr.%.loc12_33.2 (%.2) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @CallGenericPtr.%.loc12_33.2 (%.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %CallSpecific.decl: %CallSpecific.type = fn_decl @CallSpecific [template = constants.%CallSpecific] {
// CHECK:STDOUT:     %x.patt: %C = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %C = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc18_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc18_26: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %x.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %x: %C = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc16: <witness> = complete_type_witness %.4 [template = constants.%.5]
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
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @Function.%T.loc4_13.2 (%T)) -> @Function.%T.loc4_13.2 (%T) {
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
// CHECK:STDOUT:   %.loc9_10.2: <specific function> = specific_function constants.%Function, @Function(%T.loc8_16.2) [symbolic = %.loc9_10.2 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type, %x.param_patt: @CallGeneric.%T.loc8_16.2 (%T)) -> @CallGeneric.%T.loc8_16.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [template = constants.%Function]
// CHECK:STDOUT:     %x.ref: @CallGeneric.%T.loc8_16.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc9_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%T) [symbolic = %.loc9_10.2 (constants.%.1)]
// CHECK:STDOUT:     %Function.call: init @CallGeneric.%T.loc8_16.2 (%T) = call %.loc9_10.1(%x.ref)
// CHECK:STDOUT:     %.loc9_21.1: @CallGeneric.%T.loc8_16.2 (%T) = value_of_initializer %Function.call
// CHECK:STDOUT:     %.loc9_21.2: @CallGeneric.%T.loc8_16.2 (%T) = converted %Function.call, %.loc9_21.1
// CHECK:STDOUT:     return %.loc9_21.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericPtr(%T.loc12_19.1: type) {
// CHECK:STDOUT:   %T.loc12_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_19.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc12_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_19.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc12_33.2: type = ptr_type @CallGenericPtr.%T.loc12_19.2 (%T) [symbolic = %.loc12_33.2 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc13_10.2: <specific function> = specific_function constants.%Function, @Function(%.loc12_33.2) [symbolic = %.loc13_10.2 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type, %x.param_patt: @CallGenericPtr.%.loc12_33.2 (%.2)) -> @CallGenericPtr.%.loc12_33.2 (%.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [template = constants.%Function]
// CHECK:STDOUT:     %x.ref: @CallGenericPtr.%.loc12_33.2 (%.2) = name_ref x, %x
// CHECK:STDOUT:     %.loc13_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%.2) [symbolic = %.loc13_10.2 (constants.%.3)]
// CHECK:STDOUT:     %Function.call: init @CallGenericPtr.%.loc12_33.2 (%.2) = call %.loc13_10.1(%x.ref)
// CHECK:STDOUT:     %.loc13_21.1: @CallGenericPtr.%.loc12_33.2 (%.2) = value_of_initializer %Function.call
// CHECK:STDOUT:     %.loc13_21.2: @CallGenericPtr.%.loc12_33.2 (%.2) = converted %Function.call, %.loc13_21.1
// CHECK:STDOUT:     return %.loc13_21.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallSpecific(%x.param_patt: %C) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Function.ref: %Function.type = name_ref Function, file.%Function.decl [template = constants.%Function]
// CHECK:STDOUT:   %x.ref: %C = name_ref x, %x
// CHECK:STDOUT:   %.loc19: <specific function> = specific_function %Function.ref, @Function(constants.%C) [template = constants.%.7]
// CHECK:STDOUT:   %.loc18_23.2: ref %C = splice_block %return {}
// CHECK:STDOUT:   %Function.call: init %C = call %.loc19(%x.ref) to %.loc18_23.2
// CHECK:STDOUT:   return %Function.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T) {
// CHECK:STDOUT:   %T.loc8_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(@CallGeneric.%T.loc8_16.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericPtr(constants.%T) {
// CHECK:STDOUT:   %T.loc12_19.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc12_19.2 => constants.%T
// CHECK:STDOUT:   %.loc12_33.2 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%.2
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(@CallGenericPtr.%.loc12_33.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%.2
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%C) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
