// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/type_operator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/type_operator.carbon

// --- pointer.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](p: T*) -> T { return F(p); }

fn G(p: C*) -> C {
  return F(p);
}

// --- const.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](p: const T*) -> T { return F(p); }

fn G(p: const C*) -> C {
  return F(p);
}

// --- nonconst_from_const.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](p: T*) -> T { return F(p); }

fn G(p: const C*) -> const C {
  return F(p);
}

// --- fail_const_from_nonconst.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](p: const T*) -> T { return F(p); }

fn G(p: C*) -> const C {
  // CHECK:STDERR: fail_const_from_nonconst.carbon:[[@LINE+7]]:10: error: cannot deduce value for generic parameter `T` [DeductionIncomplete]
  // CHECK:STDERR:   return F(p);
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR: fail_const_from_nonconst.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F[T:! type](p: const T*) -> T { return F(p); }
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(p);
}

// CHECK:STDOUT: --- pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %F.specific_fn.ef1: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.04a: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %complete_type.d05: <witness> = complete_type_witness %ptr.019 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %p.patt: @F.%ptr.loc6_20.2 (%ptr.79f) = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: @F.%ptr.loc6_20.2 (%ptr.79f) = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc6_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc6_6.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_26: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%ptr.loc6_20.2 (%ptr.79f) = value_param call_param0
// CHECK:STDOUT:     %.loc6_20: type = splice_block %ptr.loc6_20.1 [symbolic = %ptr.loc6_20.2 (constants.%ptr.79f)] {
// CHECK:STDOUT:       %T.ref.loc6_19: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc6_20.1: type = ptr_type %T [symbolic = %ptr.loc6_20.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%ptr.loc6_20.2 (%ptr.79f) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %ptr.019 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.019 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8_16: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %p.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc8_10: type = splice_block %ptr [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref.loc8_9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.019 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.1: type) {
// CHECK:STDOUT:   %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc6_20.2: type = ptr_type @F.%T.loc6_6.2 (%T) [symbolic = %ptr.loc6_20.2 (constants.%ptr.79f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_23: <witness> = require_complete_type @F.%T.loc6_6.2 (%T) [symbolic = %require_complete.loc6_23 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc6_17: <witness> = require_complete_type @F.%ptr.loc6_20.2 (%ptr.79f) [symbolic = %require_complete.loc6_17 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %F.specific_fn.loc6_37.2: <specific function> = specific_function constants.%F, @F(%T.loc6_6.2) [symbolic = %F.specific_fn.loc6_37.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc6_6.1: type](%p.param_patt: @F.%ptr.loc6_20.2 (%ptr.79f)) -> @F.%T.loc6_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%ptr.loc6_20.2 (%ptr.79f) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc6_37.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc6_37.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_6.2 (%T) = call %F.specific_fn.loc6_37.1(%p.ref)
// CHECK:STDOUT:     %.loc6_41.1: @F.%T.loc6_6.2 (%T) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc6_41.2: @F.%T.loc6_6.2 (%T) = converted %F.call, %.loc6_41.1
// CHECK:STDOUT:     return %.loc6_41.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param_patt: %ptr.019) -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %ptr.019 = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.04a]
// CHECK:STDOUT:   %.loc8_13: ref %C = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %C = call %F.specific_fn(%p.ref) to %.loc8_13
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T.patt
// CHECK:STDOUT:   %ptr.loc6_20.2 => constants.%ptr.79f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_23 => constants.%require_complete.4ae
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%require_complete.6e5
// CHECK:STDOUT:   %F.specific_fn.loc6_37.2 => constants.%F.specific_fn.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(%T.loc6_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T.patt
// CHECK:STDOUT:   %ptr.loc6_20.2 => constants.%ptr.019
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_23 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%complete_type.d05
// CHECK:STDOUT:   %F.specific_fn.loc6_37.2 => constants.%F.specific_fn.04a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %const.a1a: type = const_type %T [symbolic]
// CHECK:STDOUT:   %ptr.6d4: type = ptr_type %const.a1a [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.20b: <witness> = require_complete_type %ptr.6d4 [symbolic]
// CHECK:STDOUT:   %F.specific_fn.ef1: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %const.668: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.801: type = ptr_type %const.668 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.04a: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %complete_type.247: <witness> = complete_type_witness %ptr.801 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %p.patt: @F.%ptr.loc6_26.2 (%ptr.6d4) = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: @F.%ptr.loc6_26.2 (%ptr.6d4) = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc6_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc6_6.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_32: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%ptr.loc6_26.2 (%ptr.6d4) = value_param call_param0
// CHECK:STDOUT:     %.loc6_26: type = splice_block %ptr.loc6_26.1 [symbolic = %ptr.loc6_26.2 (constants.%ptr.6d4)] {
// CHECK:STDOUT:       %T.ref.loc6_25: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:       %const.loc6_19.1: type = const_type %T [symbolic = %const.loc6_19.2 (constants.%const.a1a)]
// CHECK:STDOUT:       %ptr.loc6_26.1: type = ptr_type %const.a1a [symbolic = %ptr.loc6_26.2 (constants.%ptr.6d4)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%ptr.loc6_26.2 (%ptr.6d4) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %ptr.801 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.801 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8_22: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %p.param: %ptr.801 = value_param call_param0
// CHECK:STDOUT:     %.loc8_16: type = splice_block %ptr [concrete = constants.%ptr.801] {
// CHECK:STDOUT:       %C.ref.loc8_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const: type = const_type %C [concrete = constants.%const.668]
// CHECK:STDOUT:       %ptr: type = ptr_type %const.668 [concrete = constants.%ptr.801]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.801 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.1: type) {
// CHECK:STDOUT:   %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %const.loc6_19.2: type = const_type @F.%T.loc6_6.2 (%T) [symbolic = %const.loc6_19.2 (constants.%const.a1a)]
// CHECK:STDOUT:   %ptr.loc6_26.2: type = ptr_type @F.%const.loc6_19.2 (%const.a1a) [symbolic = %ptr.loc6_26.2 (constants.%ptr.6d4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_29: <witness> = require_complete_type @F.%T.loc6_6.2 (%T) [symbolic = %require_complete.loc6_29 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc6_17: <witness> = require_complete_type @F.%ptr.loc6_26.2 (%ptr.6d4) [symbolic = %require_complete.loc6_17 (constants.%require_complete.20b)]
// CHECK:STDOUT:   %F.specific_fn.loc6_43.2: <specific function> = specific_function constants.%F, @F(%T.loc6_6.2) [symbolic = %F.specific_fn.loc6_43.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc6_6.1: type](%p.param_patt: @F.%ptr.loc6_26.2 (%ptr.6d4)) -> @F.%T.loc6_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%ptr.loc6_26.2 (%ptr.6d4) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc6_43.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc6_43.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_6.2 (%T) = call %F.specific_fn.loc6_43.1(%p.ref)
// CHECK:STDOUT:     %.loc6_47.1: @F.%T.loc6_6.2 (%T) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc6_47.2: @F.%T.loc6_6.2 (%T) = converted %F.call, %.loc6_47.1
// CHECK:STDOUT:     return %.loc6_47.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param_patt: %ptr.801) -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %ptr.801 = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.04a]
// CHECK:STDOUT:   %.loc8_19: ref %C = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %C = call %F.specific_fn(%p.ref) to %.loc8_19
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T.patt
// CHECK:STDOUT:   %const.loc6_19.2 => constants.%const.a1a
// CHECK:STDOUT:   %ptr.loc6_26.2 => constants.%ptr.6d4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_29 => constants.%require_complete.4ae
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%require_complete.20b
// CHECK:STDOUT:   %F.specific_fn.loc6_43.2 => constants.%F.specific_fn.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(%T.loc6_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T.patt
// CHECK:STDOUT:   %const.loc6_19.2 => constants.%const.668
// CHECK:STDOUT:   %ptr.loc6_26.2 => constants.%ptr.801
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_29 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%complete_type.247
// CHECK:STDOUT:   %F.specific_fn.loc6_43.2 => constants.%F.specific_fn.04a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nonconst_from_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %F.specific_fn.ef1: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %const: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.801: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.486: <specific function> = specific_function %F, @F(%const) [concrete]
// CHECK:STDOUT:   %complete_type.247: <witness> = complete_type_witness %ptr.801 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %p.patt: @F.%ptr.loc6_20.2 (%ptr.79f) = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: @F.%ptr.loc6_20.2 (%ptr.79f) = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc6_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc6_6.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_26: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%ptr.loc6_20.2 (%ptr.79f) = value_param call_param0
// CHECK:STDOUT:     %.loc6_20: type = splice_block %ptr.loc6_20.1 [symbolic = %ptr.loc6_20.2 (constants.%ptr.79f)] {
// CHECK:STDOUT:       %T.ref.loc6_19: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc6_20.1: type = ptr_type %T [symbolic = %ptr.loc6_20.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%ptr.loc6_20.2 (%ptr.79f) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %ptr.801 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.801 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %const = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %const = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8_28: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %const.loc8_22: type = const_type %C [concrete = constants.%const]
// CHECK:STDOUT:     %p.param: %ptr.801 = value_param call_param0
// CHECK:STDOUT:     %.loc8_16: type = splice_block %ptr [concrete = constants.%ptr.801] {
// CHECK:STDOUT:       %C.ref.loc8_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const.loc8_9: type = const_type %C [concrete = constants.%const]
// CHECK:STDOUT:       %ptr: type = ptr_type %const [concrete = constants.%ptr.801]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.801 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %const = out_param call_param1
// CHECK:STDOUT:     %return: ref %const = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.1: type) {
// CHECK:STDOUT:   %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc6_20.2: type = ptr_type @F.%T.loc6_6.2 (%T) [symbolic = %ptr.loc6_20.2 (constants.%ptr.79f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_23: <witness> = require_complete_type @F.%T.loc6_6.2 (%T) [symbolic = %require_complete.loc6_23 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc6_17: <witness> = require_complete_type @F.%ptr.loc6_20.2 (%ptr.79f) [symbolic = %require_complete.loc6_17 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %F.specific_fn.loc6_37.2: <specific function> = specific_function constants.%F, @F(%T.loc6_6.2) [symbolic = %F.specific_fn.loc6_37.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc6_6.1: type](%p.param_patt: @F.%ptr.loc6_20.2 (%ptr.79f)) -> @F.%T.loc6_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%ptr.loc6_20.2 (%ptr.79f) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc6_37.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc6_37.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_6.2 (%T) = call %F.specific_fn.loc6_37.1(%p.ref)
// CHECK:STDOUT:     %.loc6_41.1: @F.%T.loc6_6.2 (%T) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc6_41.2: @F.%T.loc6_6.2 (%T) = converted %F.call, %.loc6_41.1
// CHECK:STDOUT:     return %.loc6_41.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param_patt: %ptr.801) -> %return.param_patt: %const {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %ptr.801 = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%const) [concrete = constants.%F.specific_fn.486]
// CHECK:STDOUT:   %.loc8_19: ref %const = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %const = call %F.specific_fn(%p.ref) to %.loc8_19
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T.patt
// CHECK:STDOUT:   %ptr.loc6_20.2 => constants.%ptr.79f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_23 => constants.%require_complete.4ae
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%require_complete.6e5
// CHECK:STDOUT:   %F.specific_fn.loc6_37.2 => constants.%F.specific_fn.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(%T.loc6_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%const) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%const
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T.patt
// CHECK:STDOUT:   %ptr.loc6_20.2 => constants.%ptr.801
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_23 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%complete_type.247
// CHECK:STDOUT:   %F.specific_fn.loc6_37.2 => constants.%F.specific_fn.486
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_const_from_nonconst.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %const.a1a: type = const_type %T [symbolic]
// CHECK:STDOUT:   %ptr.6d4: type = ptr_type %const.a1a [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.20b: <witness> = require_complete_type %ptr.6d4 [symbolic]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %const.668: type = const_type %C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %p.patt: @F.%ptr.loc6_26.2 (%ptr.6d4) = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: @F.%ptr.loc6_26.2 (%ptr.6d4) = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc6_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc6_6.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_32: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%ptr.loc6_26.2 (%ptr.6d4) = value_param call_param0
// CHECK:STDOUT:     %.loc6_26: type = splice_block %ptr.loc6_26.1 [symbolic = %ptr.loc6_26.2 (constants.%ptr.6d4)] {
// CHECK:STDOUT:       %T.ref.loc6_25: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:       %const.loc6_19.1: type = const_type %T [symbolic = %const.loc6_19.2 (constants.%const.a1a)]
// CHECK:STDOUT:       %ptr.loc6_26.1: type = ptr_type %const.a1a [symbolic = %ptr.loc6_26.2 (constants.%ptr.6d4)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%ptr.loc6_26.2 (%ptr.6d4) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %ptr.019 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.019 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %const.668 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %const.668 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8_22: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %const: type = const_type %C [concrete = constants.%const.668]
// CHECK:STDOUT:     %p.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc8: type = splice_block %ptr [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref.loc8_9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.019 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %const.668 = out_param call_param1
// CHECK:STDOUT:     %return: ref %const.668 = return_slot %return.param
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
// CHECK:STDOUT: generic fn @F(%T.loc6_6.1: type) {
// CHECK:STDOUT:   %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %const.loc6_19.2: type = const_type @F.%T.loc6_6.2 (%T) [symbolic = %const.loc6_19.2 (constants.%const.a1a)]
// CHECK:STDOUT:   %ptr.loc6_26.2: type = ptr_type @F.%const.loc6_19.2 (%const.a1a) [symbolic = %ptr.loc6_26.2 (constants.%ptr.6d4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_29: <witness> = require_complete_type @F.%T.loc6_6.2 (%T) [symbolic = %require_complete.loc6_29 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc6_17: <witness> = require_complete_type @F.%ptr.loc6_26.2 (%ptr.6d4) [symbolic = %require_complete.loc6_17 (constants.%require_complete.20b)]
// CHECK:STDOUT:   %F.specific_fn.loc6_43.2: <specific function> = specific_function constants.%F, @F(%T.loc6_6.2) [symbolic = %F.specific_fn.loc6_43.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc6_6.1: type](%p.param_patt: @F.%ptr.loc6_26.2 (%ptr.6d4)) -> @F.%T.loc6_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%ptr.loc6_26.2 (%ptr.6d4) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc6_43.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc6_43.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_6.2 (%T) = call %F.specific_fn.loc6_43.1(%p.ref)
// CHECK:STDOUT:     %.loc6_47.1: @F.%T.loc6_6.2 (%T) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc6_47.2: @F.%T.loc6_6.2 (%T) = converted %F.call, %.loc6_47.1
// CHECK:STDOUT:     return %.loc6_47.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param_patt: %ptr.019) -> %return.param_patt: %const.668 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %ptr.019 = name_ref p, %p
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T.patt
// CHECK:STDOUT:   %const.loc6_19.2 => constants.%const.a1a
// CHECK:STDOUT:   %ptr.loc6_26.2 => constants.%ptr.6d4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_29 => constants.%require_complete.4ae
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%require_complete.20b
// CHECK:STDOUT:   %F.specific_fn.loc6_43.2 => constants.%F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(%T.loc6_6.2) {}
// CHECK:STDOUT:
