// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/eval/symbolic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/eval/symbolic.carbon

// Check that we propagate the `symbolic` tag through evaluations.
fn F(T:! type) {
  var u: (T*, const T);
  var v: {.a: T};
  var w: [T; 5];
}

// CHECK:STDOUT: --- symbolic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %.3: type = const_type %T [symbolic]
// CHECK:STDOUT:   %.4: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.5: type = tuple_type (%.2, %.3) [symbolic]
// CHECK:STDOUT:   %.6: type = tuple_type (%.2, %T) [symbolic]
// CHECK:STDOUT:   %.7: type = ptr_type %.6 [symbolic]
// CHECK:STDOUT:   %.8: type = struct_type {.a: %T} [symbolic]
// CHECK:STDOUT:   %.9: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.10: type = array_type %.9, %T [symbolic]
// CHECK:STDOUT:   %.11: type = ptr_type %.10 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Core: <namespace> = namespace %Core.import, [template] {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.loc12_6.1: type = param T
// CHECK:STDOUT:     @F.%T: type = bind_symbolic_name T 0, %T.loc12_6.1 [symbolic = @F.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%T: type)
// CHECK:STDOUT:     generic [%T: type] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %T.ref.loc13_11: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc13_12: type = ptr_type %T [symbolic = constants.%.2]
// CHECK:STDOUT:   %T.ref.loc13_21: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc13_15: type = const_type %T [symbolic = constants.%.3]
// CHECK:STDOUT:   %.loc13_22.1: %.4 = tuple_literal (%.loc13_12, %.loc13_15)
// CHECK:STDOUT:   %.loc13_22.2: type = converted %.loc13_22.1, constants.%.5 [symbolic = constants.%.5]
// CHECK:STDOUT:   %u.var: ref %.5 = var u
// CHECK:STDOUT:   %u: ref %.5 = bind_name u, %u.var
// CHECK:STDOUT:   %T.ref.loc14: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc14: type = struct_type {.a: %T} [symbolic = constants.%.8]
// CHECK:STDOUT:   %v.var: ref %.8 = var v
// CHECK:STDOUT:   %v: ref %.8 = bind_name v, %v.var
// CHECK:STDOUT:   %T.ref.loc15: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc15_14: i32 = int_literal 5 [template = constants.%.9]
// CHECK:STDOUT:   %.loc15_15: type = array_type %.loc15_14, %T [symbolic = constants.%.10]
// CHECK:STDOUT:   %w.var: ref %.10 = var w
// CHECK:STDOUT:   %w: ref %.10 = bind_name w, %w.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: instance file.%F.decl(constants.%T) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   @F.%T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
