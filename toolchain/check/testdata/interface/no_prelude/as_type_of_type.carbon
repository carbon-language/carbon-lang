// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/as_type_of_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/as_type_of_type.carbon

interface Empty {}

fn F(T:! Empty) {
  var x: T;
}

// CHECK:STDOUT: --- as_type_of_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty> [concrete]
// CHECK:STDOUT:   %Self: %Empty.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: %Empty.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %Empty.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Empty.decl: type = interface_decl @Empty [concrete = constants.%Empty.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc13_6.1: %Empty.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.type]
// CHECK:STDOUT:     %T.loc13_6.1: %Empty.type = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:   %Self: %Empty.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc13_6.1: %Empty.type) {
// CHECK:STDOUT:   %T.loc13_6.2: %Empty.type = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc13_6.2: %Empty.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_6.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc14_10.2: type = facet_access_type %T.loc13_6.2 [symbolic = %T.as_type.loc14_10.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%T.as_type.loc14_10.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc13_6.1: %Empty.type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %x.patt: @F.%T.as_type.loc14_10.2 (%T.as_type) = binding_pattern x
// CHECK:STDOUT:       %.loc14_3: @F.%T.as_type.loc14_10.2 (%T.as_type) = var_pattern %x.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.var: ref @F.%T.as_type.loc14_10.2 (%T.as_type) = var x
// CHECK:STDOUT:     %.loc14_10.1: type = splice_block %.loc14_10.2 [symbolic = %T.as_type.loc14_10.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %Empty.type = name_ref T, %T.loc13_6.1 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc14_10.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc14_10.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc14_10.2: type = converted %T.ref, %T.as_type.loc14_10.1 [symbolic = %T.as_type.loc14_10.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref @F.%T.as_type.loc14_10.2 (%T.as_type) = bind_name x, %x.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc13_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc13_6.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
