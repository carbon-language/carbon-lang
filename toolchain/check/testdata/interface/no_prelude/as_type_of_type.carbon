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
// CHECK:STDOUT:   %.1: type = interface_type @Empty [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %T: %.1 = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Empty.decl: type = interface_decl @Empty [template = constants.%.1] {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Empty.ref: type = name_ref Empty, %Empty.decl [template = constants.%.1]
// CHECK:STDOUT:     %T.loc13_6.1: %.1 = param T, runtime_param<invalid>
// CHECK:STDOUT:     @F.%T.loc13: %.1 = bind_symbolic_name T 0, %T.loc13_6.1 [symbolic = @F.%T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc13: %.1) {
// CHECK:STDOUT:   %T.1: %.1 = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc13: %.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %.1 = name_ref T, %T.loc13 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc14_10.1: type = facet_type_access %T.ref [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc14_10.2: type = converted %T.ref, %.loc14_10.1 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %x.var: ref @F.%T.1 (%T) = var x
// CHECK:STDOUT:     %x: ref @F.%T.1 (%T) = bind_name x, %x.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
