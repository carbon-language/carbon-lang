// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/fail_member_access_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/struct/fail_member_access_type.carbon

var x: {.a: f64} = {.a = 4.0};
// CHECK:STDERR: fail_member_access_type.carbon:[[@LINE+3]]:14: ERROR: Type `{.a: f64}` does not have a member `b`.
// CHECK:STDERR: var y: i32 = x.b;
// CHECK:STDERR:              ^~~
var y: i32 = x.b;

// CHECK:STDOUT: --- fail_member_access_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %Float: type = fn_type @Float [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Float = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: f64} [template]
// CHECK:STDOUT:   %.4: f64 = float_literal 4 [template]
// CHECK:STDOUT:   %struct.2: {.a: f64} = struct_value (%.4) [template]
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %struct.3: Int32 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc11_13.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:   %import_ref.1: Float = import_ref ir3, inst+31, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%struct.1(%.loc11_13.1) [template = f64]
// CHECK:STDOUT:   %.loc11_13.2: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc11_13.3: type = converted %float.make_type, %.loc11_13.2 [template = f64]
// CHECK:STDOUT:   %.loc11_16: type = struct_type {.a: f64} [template = constants.%.3]
// CHECK:STDOUT:   %x.var: ref {.a: f64} = var x
// CHECK:STDOUT:   %x: ref {.a: f64} = bind_name x, %x.var
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.3]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%struct.3() [template = i32]
// CHECK:STDOUT:   %.loc15_8.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc15_8.2: type = converted %int.make_type_32, %.loc15_8.1 [template = i32]
// CHECK:STDOUT:   %y.var: ref i32 = var y
// CHECK:STDOUT:   %y: ref i32 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_26: f64 = float_literal 4 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_29.1: {.a: f64} = struct_literal (%.loc11_26)
// CHECK:STDOUT:   %.loc11_29.2: init {.a: f64} = struct_init (%.loc11_26) to file.%x.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc11_30: init {.a: f64} = converted %.loc11_29.1, %.loc11_29.2 [template = constants.%struct.2]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_30
// CHECK:STDOUT:   %x.ref: ref {.a: f64} = name_ref x, file.%x
// CHECK:STDOUT:   assign file.%y.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
