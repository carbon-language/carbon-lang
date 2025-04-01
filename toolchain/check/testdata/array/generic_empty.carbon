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
  var arr: array(T, 0) = ();
}

// CHECK:STDOUT: --- generic_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_0, %T [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type [symbolic]
// CHECK:STDOUT:   %array: %array_type = tuple_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc11_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc11_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc11_6.1: type) {
// CHECK:STDOUT:   %T.loc11_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_6.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %array_type.loc13_22.2: type = array_type constants.%int_0, @G.%T.loc11_6.2 (%T) [symbolic = %array_type.loc13_22.2 (constants.%array_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @G.%array_type.loc13_22.2 (%array_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %array: @G.%array_type.loc13_22.2 (%array_type) = tuple_value () [symbolic = %array (constants.%array)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc11_6.1: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %arr.patt: @G.%array_type.loc13_22.2 (%array_type) = binding_pattern arr
// CHECK:STDOUT:       %.loc13_3.1: @G.%array_type.loc13_22.2 (%array_type) = var_pattern %arr.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr.var: ref @G.%array_type.loc13_22.2 (%array_type) = var arr
// CHECK:STDOUT:     %.loc13_27.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc13_27.2: init @G.%array_type.loc13_22.2 (%array_type) = array_init () to %arr.var [symbolic = %array (constants.%array)]
// CHECK:STDOUT:     %.loc13_3.2: init @G.%array_type.loc13_22.2 (%array_type) = converted %.loc13_27.1, %.loc13_27.2 [symbolic = %array (constants.%array)]
// CHECK:STDOUT:     assign %arr.var, %.loc13_3.2
// CHECK:STDOUT:     %.loc13_22: type = splice_block %array_type.loc13_22.1 [symbolic = %array_type.loc13_22.2 (constants.%array_type)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc11_6.1 [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %array_type.loc13_22.1: type = array_type %int_0, %T [symbolic = %array_type.loc13_22.2 (constants.%array_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr: ref @G.%array_type.loc13_22.2 (%array_type) = bind_name arr, %arr.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc11_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_6.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
