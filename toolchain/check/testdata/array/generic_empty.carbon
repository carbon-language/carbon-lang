// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/generic_empty.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/generic_empty.carbon

fn G(T:! type) {
  // We can initialize this without knowing T.
  var arr: [T; 0] = ();
}

// CHECK:STDOUT: --- generic_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, %T [symbolic]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [symbolic]
// CHECK:STDOUT:   %array: %.3 = tuple_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Core: <namespace> = namespace %Core.import, [template] {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %T.loc11_6.1: type = param T
// CHECK:STDOUT:     @G.%T: type = bind_symbolic_name T 0, %T.loc11_6.1 [symbolic = @G.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%T: type)
// CHECK:STDOUT:     generic [%T: type] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc13_16: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_17: type = array_type %.loc13_16, %T [symbolic = constants.%.3]
// CHECK:STDOUT:   %arr.var: ref %.3 = var arr
// CHECK:STDOUT:   %arr: ref %.3 = bind_name arr, %arr.var
// CHECK:STDOUT:   %.loc13_22.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc13_22.2: init %.3 = array_init () to %arr.var [symbolic = constants.%array]
// CHECK:STDOUT:   %.loc13_23: init %.3 = converted %.loc13_22.1, %.loc13_22.2 [symbolic = constants.%array]
// CHECK:STDOUT:   assign %arr.var, %.loc13_23
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
