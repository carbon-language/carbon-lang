// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/call.carbon

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
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Function.type: type = fn_type @Function [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Function: %Function.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %Function.specific_fn.46f: <specific function> = specific_function %Function, @Function(%T) [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %CallGenericPtr.type: type = fn_type @CallGenericPtr [concrete]
// CHECK:STDOUT:   %CallGenericPtr: %CallGenericPtr.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %Function.specific_fn.4d7: <specific function> = specific_function %Function, @Function(%ptr.79f) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.edd: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %CallSpecific.type: type = fn_type @CallSpecific [concrete]
// CHECK:STDOUT:   %CallSpecific: %CallSpecific.type = struct_value () [concrete]
// CHECK:STDOUT:   %Function.specific_fn.1b5: <specific function> = specific_function %Function, @Function(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Function = %Function.decl
// CHECK:STDOUT:     .CallGeneric = %CallGeneric.decl
// CHECK:STDOUT:     .CallGenericPtr = %CallGenericPtr.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .CallSpecific = %CallSpecific.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Function.decl: %Function.type = fn_decl @Function [concrete = constants.%Function] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @Function.%pattern_type (%pattern_type.7dc) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @Function.%pattern_type (%pattern_type.7dc) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Function.%pattern_type (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Function.%pattern_type (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_32: type = name_ref T, %T.loc4_13.2 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Function.%T.loc4_13.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc4_26: type = name_ref T, %T.loc4_13.2 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:     %x: @Function.%T.loc4_13.1 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Function.%T.loc4_13.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Function.%T.loc4_13.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @CallGeneric.%pattern_type (%pattern_type.7dc) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @CallGeneric.%pattern_type (%pattern_type.7dc) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @CallGeneric.%pattern_type (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @CallGeneric.%pattern_type (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_35: type = name_ref T, %T.loc8_16.2 [symbolic = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:     %T.loc8_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @CallGeneric.%T.loc8_16.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc8_29: type = name_ref T, %T.loc8_16.2 [symbolic = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:     %x: @CallGeneric.%T.loc8_16.1 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @CallGeneric.%T.loc8_16.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @CallGeneric.%T.loc8_16.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericPtr.decl: %CallGenericPtr.type = fn_decl @CallGenericPtr [concrete = constants.%CallGenericPtr] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @CallGenericPtr.%pattern_type (%pattern_type.afe) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @CallGenericPtr.%pattern_type (%pattern_type.afe) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @CallGenericPtr.%pattern_type (%pattern_type.afe) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @CallGenericPtr.%pattern_type (%pattern_type.afe) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12_39: type = name_ref T, %T.loc12_19.2 [symbolic = %T.loc12_19.1 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc12_40: type = ptr_type %T.ref.loc12_39 [symbolic = %ptr.loc12_33.1 (constants.%ptr.79f)]
// CHECK:STDOUT:     %T.loc12_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_19.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f) = value_param call_param0
// CHECK:STDOUT:     %.loc12: type = splice_block %ptr.loc12_33.2 [symbolic = %ptr.loc12_33.1 (constants.%ptr.79f)] {
// CHECK:STDOUT:       %T.ref.loc12_32: type = name_ref T, %T.loc12_19.2 [symbolic = %T.loc12_19.1 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc12_33.2: type = ptr_type %T.ref.loc12_32 [symbolic = %ptr.loc12_33.1 (constants.%ptr.79f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f) = out_param call_param1
// CHECK:STDOUT:     %return: ref @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %CallSpecific.decl: %CallSpecific.type = fn_decl @CallSpecific [concrete = constants.%CallSpecific] {
// CHECK:STDOUT:     %x.patt: %pattern_type.c48 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.c48 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param1 [concrete]
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
// CHECK:STDOUT: impl @C.as.Destroy.impl: constants.%C as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc16: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.edd]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Function(%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc4_13.1 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_13.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Function.%T.loc4_13.1 (%T)) -> @Function.%T.loc4_13.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Function.%T.loc4_13.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc8_16.2: type) {
// CHECK:STDOUT:   %T.loc8_16.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc8_16.1 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc8_16.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Function.specific_fn.loc9_10.2: <specific function> = specific_function constants.%Function, @Function(%T.loc8_16.1) [symbolic = %Function.specific_fn.loc9_10.2 (constants.%Function.specific_fn.46f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @CallGeneric.%T.loc8_16.1 (%T)) -> @CallGeneric.%T.loc8_16.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %T.ref.loc9: type = name_ref T, %T.loc8_16.2 [symbolic = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:     %x.ref: @CallGeneric.%T.loc8_16.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %Function.specific_fn.loc9_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%T) [symbolic = %Function.specific_fn.loc9_10.2 (constants.%Function.specific_fn.46f)]
// CHECK:STDOUT:     %Function.call: init @CallGeneric.%T.loc8_16.1 (%T) = call %Function.specific_fn.loc9_10.1(%x.ref)
// CHECK:STDOUT:     %.loc9_24.1: @CallGeneric.%T.loc8_16.1 (%T) = value_of_initializer %Function.call
// CHECK:STDOUT:     %.loc9_24.2: @CallGeneric.%T.loc8_16.1 (%T) = converted %Function.call, %.loc9_24.1
// CHECK:STDOUT:     return %.loc9_24.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericPtr(%T.loc12_19.2: type) {
// CHECK:STDOUT:   %T.loc12_19.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_19.1 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc12_33.1: type = ptr_type %T.loc12_19.1 [symbolic = %ptr.loc12_33.1 (constants.%ptr.79f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc12_33.1 [symbolic = %pattern_type (constants.%pattern_type.afe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.loc12_33.1 [symbolic = %require_complete (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %Function.specific_fn.loc13_10.2: <specific function> = specific_function constants.%Function, @Function(%ptr.loc12_33.1) [symbolic = %Function.specific_fn.loc13_10.2 (constants.%Function.specific_fn.4d7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f)) -> @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %T.ref.loc13: type = name_ref T, %T.loc12_19.2 [symbolic = %T.loc12_19.1 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc13: type = ptr_type %T.ref.loc13 [symbolic = %ptr.loc12_33.1 (constants.%ptr.79f)]
// CHECK:STDOUT:     %x.ref: @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f) = name_ref x, %x
// CHECK:STDOUT:     %Function.specific_fn.loc13_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%ptr.79f) [symbolic = %Function.specific_fn.loc13_10.2 (constants.%Function.specific_fn.4d7)]
// CHECK:STDOUT:     %Function.call: init @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f) = call %Function.specific_fn.loc13_10.1(%x.ref)
// CHECK:STDOUT:     %.loc13_25.1: @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f) = value_of_initializer %Function.call
// CHECK:STDOUT:     %.loc13_25.2: @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f) = converted %Function.call, %.loc13_25.1
// CHECK:STDOUT:     return %.loc13_25.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallSpecific(%x.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:   %C.ref.loc19: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %x.ref: %C = name_ref x, %x
// CHECK:STDOUT:   %Function.specific_fn: <specific function> = specific_function %Function.ref, @Function(constants.%C) [concrete = constants.%Function.specific_fn.1b5]
// CHECK:STDOUT:   %.loc18: ref %C = splice_block %return {}
// CHECK:STDOUT:   %Function.call: init %C = call %Function.specific_fn(%x.ref) to %.loc18
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc18, constants.%C.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc18
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call: init %empty_tuple.type = call %C.as.Destroy.impl.Op.bound(%addr)
// CHECK:STDOUT:   return %Function.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T) {
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericPtr(constants.%T) {
// CHECK:STDOUT:   %T.loc12_19.1 => constants.%T
// CHECK:STDOUT:   %ptr.loc12_33.1 => constants.%ptr.79f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.afe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%ptr.79f) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%ptr.79f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.afe
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6e5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%C) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c48
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Function.type: type = fn_type @Function [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Function: %Function.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %CallGeneric.type: type = fn_type @CallGeneric [concrete]
// CHECK:STDOUT:   %CallGeneric: %CallGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %Function.specific_fn.46f: <specific function> = specific_function %Function, @Function(%T) [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %CallGenericPtr.type: type = fn_type @CallGenericPtr [concrete]
// CHECK:STDOUT:   %CallGenericPtr: %CallGenericPtr.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %Function.specific_fn.4d7: <specific function> = specific_function %Function, @Function(%ptr.79f) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.edd: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %CallSpecific.type: type = fn_type @CallSpecific [concrete]
// CHECK:STDOUT:   %CallSpecific: %CallSpecific.type = struct_value () [concrete]
// CHECK:STDOUT:   %Function.specific_fn.1b5: <specific function> = specific_function %Function, @Function(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Function = %Function.decl
// CHECK:STDOUT:     .CallGeneric = %CallGeneric.decl
// CHECK:STDOUT:     .CallGenericPtr = %CallGenericPtr.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .CallSpecific = %CallSpecific.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Function.decl: %Function.type = fn_decl @Function [concrete = constants.%Function] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @Function.%pattern_type (%pattern_type.7dc) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @Function.%pattern_type (%pattern_type.7dc) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Function.%pattern_type (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Function.%pattern_type (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_32: type = name_ref T, %T.loc4_13.2 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Function.%T.loc4_13.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc4_26: type = name_ref T, %T.loc4_13.2 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:     %x: @Function.%T.loc4_13.1 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Function.%T.loc4_13.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Function.%T.loc4_13.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGeneric.decl: %CallGeneric.type = fn_decl @CallGeneric [concrete = constants.%CallGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @CallGeneric.%pattern_type (%pattern_type.7dc) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @CallGeneric.%pattern_type (%pattern_type.7dc) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @CallGeneric.%pattern_type (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @CallGeneric.%pattern_type (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_35: type = name_ref T, %T.loc8_16.2 [symbolic = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:     %T.loc8_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @CallGeneric.%T.loc8_16.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc8_29: type = name_ref T, %T.loc8_16.2 [symbolic = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:     %x: @CallGeneric.%T.loc8_16.1 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @CallGeneric.%T.loc8_16.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @CallGeneric.%T.loc8_16.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGenericPtr.decl: %CallGenericPtr.type = fn_decl @CallGenericPtr [concrete = constants.%CallGenericPtr] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @CallGenericPtr.%pattern_type (%pattern_type.afe) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @CallGenericPtr.%pattern_type (%pattern_type.afe) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @CallGenericPtr.%pattern_type (%pattern_type.afe) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @CallGenericPtr.%pattern_type (%pattern_type.afe) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12_39: type = name_ref T, %T.loc12_19.2 [symbolic = %T.loc12_19.1 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc12_40: type = ptr_type %T.ref.loc12_39 [symbolic = %ptr.loc12_33.1 (constants.%ptr.79f)]
// CHECK:STDOUT:     %T.loc12_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_19.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f) = value_param call_param0
// CHECK:STDOUT:     %.loc12: type = splice_block %ptr.loc12_33.2 [symbolic = %ptr.loc12_33.1 (constants.%ptr.79f)] {
// CHECK:STDOUT:       %T.ref.loc12_32: type = name_ref T, %T.loc12_19.2 [symbolic = %T.loc12_19.1 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc12_33.2: type = ptr_type %T.ref.loc12_32 [symbolic = %ptr.loc12_33.1 (constants.%ptr.79f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f) = out_param call_param1
// CHECK:STDOUT:     %return: ref @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %CallSpecific.decl: %CallSpecific.type = fn_decl @CallSpecific [concrete = constants.%CallSpecific] {
// CHECK:STDOUT:     %x.patt: %pattern_type.c48 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.c48 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param1 [concrete]
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
// CHECK:STDOUT: impl @C.as.Destroy.impl: constants.%C as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc16: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.edd]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Function(%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc4_13.1 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_13.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Function.%T.loc4_13.1 (%T)) -> @Function.%T.loc4_13.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Function.%T.loc4_13.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc8_16.2: type) {
// CHECK:STDOUT:   %T.loc8_16.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc8_16.1 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc8_16.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Function.specific_fn.loc9_10.2: <specific function> = specific_function constants.%Function, @Function(%T.loc8_16.1) [symbolic = %Function.specific_fn.loc9_10.2 (constants.%Function.specific_fn.46f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @CallGeneric.%T.loc8_16.1 (%T)) -> @CallGeneric.%T.loc8_16.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %x.ref: @CallGeneric.%T.loc8_16.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %Function.specific_fn.loc9_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%T) [symbolic = %Function.specific_fn.loc9_10.2 (constants.%Function.specific_fn.46f)]
// CHECK:STDOUT:     %Function.call: init @CallGeneric.%T.loc8_16.1 (%T) = call %Function.specific_fn.loc9_10.1(%x.ref)
// CHECK:STDOUT:     %.loc9_21.1: @CallGeneric.%T.loc8_16.1 (%T) = value_of_initializer %Function.call
// CHECK:STDOUT:     %.loc9_21.2: @CallGeneric.%T.loc8_16.1 (%T) = converted %Function.call, %.loc9_21.1
// CHECK:STDOUT:     return %.loc9_21.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericPtr(%T.loc12_19.2: type) {
// CHECK:STDOUT:   %T.loc12_19.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_19.1 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc12_33.1: type = ptr_type %T.loc12_19.1 [symbolic = %ptr.loc12_33.1 (constants.%ptr.79f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc12_33.1 [symbolic = %pattern_type (constants.%pattern_type.afe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.loc12_33.1 [symbolic = %require_complete (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %Function.specific_fn.loc13_10.2: <specific function> = specific_function constants.%Function, @Function(%ptr.loc12_33.1) [symbolic = %Function.specific_fn.loc13_10.2 (constants.%Function.specific_fn.4d7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f)) -> @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %x.ref: @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f) = name_ref x, %x
// CHECK:STDOUT:     %Function.specific_fn.loc13_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%ptr.79f) [symbolic = %Function.specific_fn.loc13_10.2 (constants.%Function.specific_fn.4d7)]
// CHECK:STDOUT:     %Function.call: init @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f) = call %Function.specific_fn.loc13_10.1(%x.ref)
// CHECK:STDOUT:     %.loc13_21.1: @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f) = value_of_initializer %Function.call
// CHECK:STDOUT:     %.loc13_21.2: @CallGenericPtr.%ptr.loc12_33.1 (%ptr.79f) = converted %Function.call, %.loc13_21.1
// CHECK:STDOUT:     return %.loc13_21.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallSpecific(%x.param: %C) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:   %x.ref: %C = name_ref x, %x
// CHECK:STDOUT:   %Function.specific_fn: <specific function> = specific_function %Function.ref, @Function(constants.%C) [concrete = constants.%Function.specific_fn.1b5]
// CHECK:STDOUT:   %.loc18: ref %C = splice_block %return {}
// CHECK:STDOUT:   %Function.call: init %C = call %Function.specific_fn(%x.ref) to %.loc18
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc18, constants.%C.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc18
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call: init %empty_tuple.type = call %C.as.Destroy.impl.Op.bound(%addr)
// CHECK:STDOUT:   return %Function.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T) {
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericPtr(constants.%T) {
// CHECK:STDOUT:   %T.loc12_19.1 => constants.%T
// CHECK:STDOUT:   %ptr.loc12_33.1 => constants.%ptr.79f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.afe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%ptr.79f) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%ptr.79f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.afe
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6e5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%C) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c48
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
