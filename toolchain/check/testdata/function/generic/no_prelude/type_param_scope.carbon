// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/no_prelude/type_param_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/no_prelude/type_param_scope.carbon

fn F(T:! type, n: T) -> T {
  let m: T = n;
  return m;
}

// CHECK:STDOUT: --- type_param_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.loc11_6.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     @F.%T.loc11: type = bind_symbolic_name T 0, %T.loc11_6.1 [symbolic = @F.%T.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc11_19: type = name_ref T, @F.%T.loc11 [symbolic = @F.%T.1 (constants.%T)]
// CHECK:STDOUT:     %n.loc11_16.1: @F.%T.1 (%T) = param n, runtime_param0
// CHECK:STDOUT:     @F.%n: @F.%T.1 (%T) = bind_name n, %n.loc11_16.1
// CHECK:STDOUT:     %T.ref.loc11_25: type = name_ref T, @F.%T.loc11 [symbolic = @F.%T.1 (constants.%T)]
// CHECK:STDOUT:     @F.%return: ref @F.%T.1 (%T) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc11: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc11: type, %n: @F.%T.1 (%T)) -> @F.%T.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %n.ref: @F.%T.1 (%T) = name_ref n, %n
// CHECK:STDOUT:     %m: @F.%T.1 (%T) = bind_name m, %n.ref
// CHECK:STDOUT:     %m.ref: @F.%T.1 (%T) = name_ref m, %m
// CHECK:STDOUT:     return %m.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
