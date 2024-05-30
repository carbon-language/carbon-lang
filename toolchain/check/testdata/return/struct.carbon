// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/return/struct.carbon

fn Main() -> {.a: i32} {
  return {.a = 3};
}

// CHECK:STDOUT: --- struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %Main: type = fn_type @Main [template]
// CHECK:STDOUT:   %struct.2: Main = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %struct.3: {.a: i32} = struct_value (%.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Main.decl: Main = fn_decl @Main [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc11_19.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc11_19.2: type = converted %int.make_type_32, %.loc11_19.1 [template = i32]
// CHECK:STDOUT:     %.loc11_22: type = struct_type {.a: i32} [template = constants.%.2]
// CHECK:STDOUT:     @Main.%return: ref {.a: i32} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> {.a: i32} {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_16: i32 = int_literal 3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_17: {.a: i32} = struct_literal (%.loc12_16)
// CHECK:STDOUT:   %struct: {.a: i32} = struct_value (%.loc12_16) [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc12_18: {.a: i32} = converted %.loc12_17, %struct [template = constants.%struct.3]
// CHECK:STDOUT:   return %.loc12_18
// CHECK:STDOUT: }
// CHECK:STDOUT:
