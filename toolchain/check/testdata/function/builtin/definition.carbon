// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/definition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/function/builtin/definition.carbon

fn Add(a: i32, b: i32) -> i32 = "int.sadd";

// CHECK:STDOUT: --- definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %Add: type = fn_type @Add [template]
// CHECK:STDOUT:   %struct.2: Add = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Add.decl: Add = fn_decl @Add [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc11_11: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %int.make_type_32.loc11_11 [template = i32]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %int.make_type_32.loc11_11, %.loc11_11.1 [template = i32]
// CHECK:STDOUT:     %a.loc11_8.1: i32 = param a
// CHECK:STDOUT:     @Add.%a: i32 = bind_name a, %a.loc11_8.1
// CHECK:STDOUT:     %int.make_type_32.loc11_19: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc11_19.1: type = value_of_initializer %int.make_type_32.loc11_19 [template = i32]
// CHECK:STDOUT:     %.loc11_19.2: type = converted %int.make_type_32.loc11_19, %.loc11_19.1 [template = i32]
// CHECK:STDOUT:     %b.loc11_16.1: i32 = param b
// CHECK:STDOUT:     @Add.%b: i32 = bind_name b, %b.loc11_16.1
// CHECK:STDOUT:     %int.make_type_32.loc11_27: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc11_27.1: type = value_of_initializer %int.make_type_32.loc11_27 [template = i32]
// CHECK:STDOUT:     %.loc11_27.2: type = converted %int.make_type_32.loc11_27, %.loc11_27.1 [template = i32]
// CHECK:STDOUT:     @Add.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add(%a: i32, %b: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
