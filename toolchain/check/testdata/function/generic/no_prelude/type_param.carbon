// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/no_prelude/type_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/no_prelude/type_param.carbon

fn F(T:! type) {
  var p: T*;
  let n: T = *p;
}

// CHECK:STDOUT: --- type_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
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
// CHECK:STDOUT:     %T.loc11_6.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     @F.%T.loc11: type = bind_symbolic_name T 0, %T.loc11_6.1 [symbolic = @F.%T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc11: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = ptr_type @F.%T.1 (%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc11: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc12: type = name_ref T, %T.loc11 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc12: type = ptr_type %T [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:     %p.var: ref @F.%.1 (%.2) = var p
// CHECK:STDOUT:     %p: ref @F.%.1 (%.2) = bind_name p, %p.var
// CHECK:STDOUT:     %T.ref.loc13: type = name_ref T, %T.loc11 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %p.ref: ref @F.%.1 (%.2) = name_ref p, %p
// CHECK:STDOUT:     %.loc13_15: @F.%.1 (%.2) = bind_value %p.ref
// CHECK:STDOUT:     %.loc13_14.1: ref @F.%T.1 (%T) = deref %.loc13_15
// CHECK:STDOUT:     %.loc13_14.2: @F.%T.1 (%T) = bind_value %.loc13_14.1
// CHECK:STDOUT:     %n: @F.%T.1 (%T) = bind_name n, %.loc13_14.2
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
