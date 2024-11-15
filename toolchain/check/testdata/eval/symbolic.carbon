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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %.2: type = const_type %T [symbolic]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%.1, %.2) [symbolic]
// CHECK:STDOUT:   %.4: type = struct_type {.a: %T} [symbolic]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.6: type = array_type %.5, %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc12_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc12_6.1, runtime_param<invalid> [symbolic = %T.patt.loc12_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc12_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc12_6.1: type) {
// CHECK:STDOUT:   %T.loc12_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc12_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_6.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc13_12.2: type = ptr_type @F.%T.loc12_6.2 (%T) [symbolic = %.loc13_12.2 (constants.%.1)]
// CHECK:STDOUT:   %.loc13_15.2: type = const_type @F.%T.loc12_6.2 (%T) [symbolic = %.loc13_15.2 (constants.%.2)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@F.%.loc13_12.2 (%.1), @F.%.loc13_15.2 (%.2)) [symbolic = %tuple.type (constants.%tuple.type.2)]
// CHECK:STDOUT:   %.loc14_16.2: type = struct_type {.a: @F.%T.loc12_6.2 (%T)} [symbolic = %.loc14_16.2 (constants.%.4)]
// CHECK:STDOUT:   %.loc15_15.2: type = array_type constants.%.5, @F.%T.loc12_6.2 (%T) [symbolic = %.loc15_15.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc13_11: type = name_ref T, %T.loc12_6.1 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:     %.loc13_12.1: type = ptr_type %T [symbolic = %.loc13_12.2 (constants.%.1)]
// CHECK:STDOUT:     %T.ref.loc13_21: type = name_ref T, %T.loc12_6.1 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:     %.loc13_15.1: type = const_type %T [symbolic = %.loc13_15.2 (constants.%.2)]
// CHECK:STDOUT:     %.loc13_22.1: %tuple.type.1 = tuple_literal (%.loc13_12.1, %.loc13_15.1)
// CHECK:STDOUT:     %.loc13_22.2: type = converted %.loc13_22.1, constants.%tuple.type.2 [symbolic = %tuple.type (constants.%tuple.type.2)]
// CHECK:STDOUT:     %u.var: ref @F.%tuple.type (%tuple.type.2) = var u
// CHECK:STDOUT:     %u: ref @F.%tuple.type (%tuple.type.2) = bind_name u, %u.var
// CHECK:STDOUT:     %T.ref.loc14: type = name_ref T, %T.loc12_6.1 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:     %.loc14_16.1: type = struct_type {.a: %T} [symbolic = %.loc14_16.2 (constants.%.4)]
// CHECK:STDOUT:     %v.var: ref @F.%.loc14_16.2 (%.4) = var v
// CHECK:STDOUT:     %v: ref @F.%.loc14_16.2 (%.4) = bind_name v, %v.var
// CHECK:STDOUT:     %T.ref.loc15: type = name_ref T, %T.loc12_6.1 [symbolic = %T.loc12_6.2 (constants.%T)]
// CHECK:STDOUT:     %.loc15_14: Core.IntLiteral = int_value 5 [template = constants.%.5]
// CHECK:STDOUT:     %.loc15_15.1: type = array_type %.loc15_14, %T [symbolic = %.loc15_15.2 (constants.%.6)]
// CHECK:STDOUT:     %w.var: ref @F.%.loc15_15.2 (%.6) = var w
// CHECK:STDOUT:     %w: ref @F.%.loc15_15.2 (%.6) = bind_name w, %w.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc12_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc12_6.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
