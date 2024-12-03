// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/no_prelude/fail_type_param_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/no_prelude/fail_type_param_mismatch.carbon

fn F(T:! type, U:! type) {
  var p: T*;
  // CHECK:STDERR: fail_type_param_mismatch.carbon:[[@LINE+3]]:3: error: `Core.ImplicitAs` implicitly referenced here, but package `Core` not found [CoreNotFound]
  // CHECK:STDERR:   let n: U = *p;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  let n: U = *p;
}

// CHECK:STDOUT: --- fail_type_param_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc11_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_6.1, runtime_param<invalid> [symbolic = %T.patt.loc11_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc11_16.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc11_16.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc11_16.1, runtime_param<invalid> [symbolic = %U.patt.loc11_16.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:     %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc11_16.1: type = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc11_16.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc11_6.1: type, %U.loc11_16.1: type) {
// CHECK:STDOUT:   %T.loc11_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %U.loc11_16.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc11_16.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc11_16.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc11_16.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc12_11.2: type = ptr_type @F.%T.loc11_6.2 (%T) [symbolic = %ptr.loc12_11.2 (constants.%ptr)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type, %U.param_patt: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_6.1 [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc12_11.1: type = ptr_type %T [symbolic = %ptr.loc12_11.2 (constants.%ptr)]
// CHECK:STDOUT:     %p.var: ref @F.%ptr.loc12_11.2 (%ptr) = var p
// CHECK:STDOUT:     %p: ref @F.%ptr.loc12_11.2 (%ptr) = bind_name p, %p.var
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc11_16.1 [symbolic = %U.loc11_16.2 (constants.%U)]
// CHECK:STDOUT:     %p.ref: ref @F.%ptr.loc12_11.2 (%ptr) = name_ref p, %p
// CHECK:STDOUT:     %.loc16_15: @F.%ptr.loc12_11.2 (%ptr) = bind_value %p.ref
// CHECK:STDOUT:     %.loc16_14: ref @F.%T.loc11_6.2 (%T) = deref %.loc16_15
// CHECK:STDOUT:     %.loc16_16: @F.%U.loc11_16.2 (%U) = converted %.loc16_14, <error> [template = <error>]
// CHECK:STDOUT:     %n: @F.%U.loc11_16.2 (%U) = bind_name n, <error>
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc11_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_6.2 => constants.%T
// CHECK:STDOUT:   %U.loc11_16.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc11_16.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
