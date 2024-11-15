// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/struct.carbon

fn Main() -> {.a: i32} {
  return {.a = 3};
}

// CHECK:STDOUT: --- struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_value 3 [template]
// CHECK:STDOUT:   %struct: %.1 = struct_value (%.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {
// CHECK:STDOUT:     %return.patt: %.1 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.1 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_19.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc11_19.2: type = converted %int.make_type_32, %.loc11_19.1 [template = i32]
// CHECK:STDOUT:     %.loc11_22: type = struct_type {.a: i32} [template = constants.%.1]
// CHECK:STDOUT:     %return.param: ref %.1 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.1 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_16: i32 = int_value 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_17: %.1 = struct_literal (%.loc12_16)
// CHECK:STDOUT:   %struct: %.1 = struct_value (%.loc12_16) [template = constants.%struct]
// CHECK:STDOUT:   %.loc12_18: %.1 = converted %.loc12_17, %struct [template = constants.%struct]
// CHECK:STDOUT:   return %.loc12_18
// CHECK:STDOUT: }
// CHECK:STDOUT:
