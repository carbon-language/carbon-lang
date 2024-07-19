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
  // CHECK:STDERR: fail_type_param_mismatch.carbon:[[@LINE+3]]:3: ERROR: Cannot implicitly convert from `T` to `U`.
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
// CHECK:STDOUT:     @F.%T: type = bind_symbolic_name T 0, %T.loc11_6.1 [symbolic = @F.%T (constants.%T)]
// CHECK:STDOUT:     %U.loc11_16.1: type = param U
// CHECK:STDOUT:     @F.%U: type = bind_symbolic_name U 1, %U.loc11_16.1 [symbolic = @F.%U (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%T: type, %U: type)
// CHECK:STDOUT:     generic [%T: type, %U: type] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %.loc12: type = ptr_type %T [symbolic = %.loc12 (constants.%.2)]
// CHECK:STDOUT:   %p.var: ref @F.%.loc12 (%.2) = var p
// CHECK:STDOUT:   %p: ref @F.%.loc12 (%.2) = bind_name p, %p.var
// CHECK:STDOUT:   %U.ref: type = name_ref U, %U [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %p.ref: ref @F.%.loc12 (%.2) = name_ref p, %p
// CHECK:STDOUT:   %.loc16_15: @F.%.loc12 (%.2) = bind_value %p.ref
// CHECK:STDOUT:   %.loc16_14: ref @F.%T (%T) = deref %.loc16_15
// CHECK:STDOUT:   %n: @F.%U (%U) = bind_name n, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific file.%F.decl(constants.%T, constants.%U) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   @F.%T => constants.%T
// CHECK:STDOUT:   @F.%U => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
