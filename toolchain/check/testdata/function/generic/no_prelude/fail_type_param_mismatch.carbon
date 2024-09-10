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
  // CHECK:STDERR: fail_type_param_mismatch.carbon:[[@LINE+3]]:3: ERROR: Package `Core` implicitly referenced here, but not found.
  // CHECK:STDERR:   let n: U = *p;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  let n: U = *p;
}

// CHECK:STDOUT: --- fail_type_param_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.loc11_6.1: type = param T
// CHECK:STDOUT:     @F.%T.loc11: type = bind_symbolic_name T 0, %T.loc11_6.1 [symbolic = @F.%T.1 (constants.%T)]
// CHECK:STDOUT:     %U.loc11_16.1: type = param U
// CHECK:STDOUT:     @F.%U.loc11: type = bind_symbolic_name U 1, %U.loc11_16.1 [symbolic = @F.%U.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc11: type, %U.loc11: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U 1 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = ptr_type @F.%T.1 (%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc11: type, %U.loc11: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc12: type = ptr_type %T [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:     %p.var: ref @F.%.1 (%.2) = var p
// CHECK:STDOUT:     %p: ref @F.%.1 (%.2) = bind_name p, %p.var
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc11 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:     %p.ref: ref @F.%.1 (%.2) = name_ref p, %p
// CHECK:STDOUT:     %.loc16_15: @F.%.1 (%.2) = bind_value %p.ref
// CHECK:STDOUT:     %.loc16_14: ref @F.%T.1 (%T) = deref %.loc16_15
// CHECK:STDOUT:     %.loc16_16: @F.%U.1 (%U) = converted %.loc16_14, <error> [template = <error>]
// CHECK:STDOUT:     %n: @F.%U.1 (%U) = bind_name n, <error>
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
