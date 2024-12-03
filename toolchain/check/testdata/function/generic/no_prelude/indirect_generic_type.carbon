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
// CHECK:STDOUT:   %ptr.1: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %ptr.1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc11_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_6.1, runtime_param<invalid> [symbolic = %T.patt.loc11_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %p.patt: @F.%ptr.loc11_21.2 (%ptr.2) = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: @F.%ptr.loc11_21.2 (%ptr.2) = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @F.%ptr.loc11_20.2 (%ptr.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%ptr.loc11_20.2 (%ptr.1) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc11_19: type = name_ref T, %T.loc11_6.1 [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc11_20.1: type = ptr_type %T [symbolic = %ptr.loc11_20.2 (constants.%ptr.1)]
// CHECK:STDOUT:     %ptr.loc11_21.1: type = ptr_type %ptr.1 [symbolic = %ptr.loc11_21.2 (constants.%ptr.2)]
// CHECK:STDOUT:     %T.ref.loc11_27: type = name_ref T, %T.loc11_6.1 [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc11_28: type = ptr_type %T [symbolic = %ptr.loc11_20.2 (constants.%ptr.1)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%ptr.loc11_21.2 (%ptr.2) = value_param runtime_param0
// CHECK:STDOUT:     %p: @F.%ptr.loc11_21.2 (%ptr.2) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr.loc11_20.2 (%ptr.1) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @F.%ptr.loc11_20.2 (%ptr.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc11_6.1: type) {
// CHECK:STDOUT:   %T.loc11_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc11_20.2: type = ptr_type @F.%T.loc11_6.2 (%T) [symbolic = %ptr.loc11_20.2 (constants.%ptr.1)]
// CHECK:STDOUT:   %ptr.loc11_21.2: type = ptr_type @F.%ptr.loc11_20.2 (%ptr.1) [symbolic = %ptr.loc11_21.2 (constants.%ptr.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type, %p.param_patt: @F.%ptr.loc11_21.2 (%ptr.2)) -> @F.%ptr.loc11_20.2 (%ptr.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %p.ref: @F.%ptr.loc11_21.2 (%ptr.2) = name_ref p, %p
// CHECK:STDOUT:     %.loc12_10.1: ref @F.%ptr.loc11_20.2 (%ptr.1) = deref %p.ref
// CHECK:STDOUT:     %.loc12_10.2: @F.%ptr.loc11_20.2 (%ptr.1) = bind_value %.loc12_10.1
// CHECK:STDOUT:     return %.loc12_10.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc11_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_6.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc11_20.2 => constants.%ptr.1
// CHECK:STDOUT:   %ptr.loc11_21.2 => constants.%ptr.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
