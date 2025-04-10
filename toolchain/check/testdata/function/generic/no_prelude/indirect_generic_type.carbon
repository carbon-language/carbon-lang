// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/no_prelude/indirect_generic_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/no_prelude/indirect_generic_type.carbon

fn F(T:! type, p: T**) -> T* {
  return *p;
}

// CHECK:STDOUT: --- indirect_generic_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %ptr.a13: type = ptr_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %require_complete.132: <witness> = require_complete_type %ptr.a13 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc11_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %p.patt: @F.%ptr.loc11_21.2 (%ptr.a13) = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: @F.%ptr.loc11_21.2 (%ptr.a13) = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.%ptr.loc11_20.2 (%ptr.79f) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%ptr.loc11_20.2 (%ptr.79f) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc11_27: type = name_ref T, %T.loc11_6.1 [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc11_28: type = ptr_type %T.ref.loc11_27 [symbolic = %ptr.loc11_20.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     %T.loc11_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%ptr.loc11_21.2 (%ptr.a13) = value_param call_param0
// CHECK:STDOUT:     %.loc11: type = splice_block %ptr.loc11_21.1 [symbolic = %ptr.loc11_21.2 (constants.%ptr.a13)] {
// CHECK:STDOUT:       %T.ref.loc11_19: type = name_ref T, %T.loc11_6.1 [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc11_20.1: type = ptr_type %T.ref.loc11_19 [symbolic = %ptr.loc11_20.2 (constants.%ptr.79f)]
// CHECK:STDOUT:       %ptr.loc11_21.1: type = ptr_type %ptr.loc11_20.1 [symbolic = %ptr.loc11_21.2 (constants.%ptr.a13)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%ptr.loc11_21.2 (%ptr.a13) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr.loc11_20.2 (%ptr.79f) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr.loc11_20.2 (%ptr.79f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc11_6.1: type) {
// CHECK:STDOUT:   %T.loc11_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc11_20.2: type = ptr_type %T.loc11_6.2 [symbolic = %ptr.loc11_20.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %ptr.loc11_21.2: type = ptr_type %ptr.loc11_20.2 [symbolic = %ptr.loc11_21.2 (constants.%ptr.a13)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_24: <witness> = require_complete_type @F.%ptr.loc11_20.2 (%ptr.79f) [symbolic = %require_complete.loc11_24 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %require_complete.loc11_17: <witness> = require_complete_type @F.%ptr.loc11_21.2 (%ptr.a13) [symbolic = %require_complete.loc11_17 (constants.%require_complete.132)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc11_6.1: type, %p.param_patt: @F.%ptr.loc11_21.2 (%ptr.a13)) -> @F.%ptr.loc11_20.2 (%ptr.79f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %p.ref: @F.%ptr.loc11_21.2 (%ptr.a13) = name_ref p, %p
// CHECK:STDOUT:     %.loc12_10.1: ref @F.%ptr.loc11_20.2 (%ptr.79f) = deref %p.ref
// CHECK:STDOUT:     %.loc12_10.2: @F.%ptr.loc11_20.2 (%ptr.79f) = bind_value %.loc12_10.1
// CHECK:STDOUT:     return %.loc12_10.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc11_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_6.2 => constants.%T.patt
// CHECK:STDOUT:   %ptr.loc11_20.2 => constants.%ptr.79f
// CHECK:STDOUT:   %ptr.loc11_21.2 => constants.%ptr.a13
// CHECK:STDOUT: }
// CHECK:STDOUT:
