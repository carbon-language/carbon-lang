// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/tuple_element_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/index/tuple_element_access.carbon

var a: (i32,) = (12,);
var b: (i32,) = a;
var c: i32 = b[0];

// CHECK:STDOUT: --- tuple_element_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.4) [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc11_13.1: (type,) = tuple_literal (%int.make_type_32.loc11)
// CHECK:STDOUT:   %.loc11_13.2: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_13.3: type = converted %int.make_type_32.loc11, %.loc11_13.2 [template = i32]
// CHECK:STDOUT:   %.loc11_13.4: type = converted %.loc11_13.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref (i32,) = var a
// CHECK:STDOUT:   %a: ref (i32,) = bind_name a, %a.var
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc12_13.1: (type,) = tuple_literal (%int.make_type_32.loc12)
// CHECK:STDOUT:   %.loc12_13.2: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_13.3: type = converted %int.make_type_32.loc12, %.loc12_13.2 [template = i32]
// CHECK:STDOUT:   %.loc12_13.4: type = converted %.loc12_13.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %b.var: ref (i32,) = var b
// CHECK:STDOUT:   %b: ref (i32,) = bind_name b, %b.var
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc13_8.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_8.2: type = converted %int.make_type_32.loc13, %.loc13_8.1 [template = i32]
// CHECK:STDOUT:   %c.var: ref i32 = var c
// CHECK:STDOUT:   %c: ref i32 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_18: i32 = int_literal 12 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_21.1: (i32,) = tuple_literal (%.loc11_18)
// CHECK:STDOUT:   %.loc11_21.2: init (i32,) = tuple_init (%.loc11_18) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_22: init (i32,) = converted %.loc11_21.1, %.loc11_21.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_22
// CHECK:STDOUT:   %a.ref: ref (i32,) = name_ref a, file.%a
// CHECK:STDOUT:   %.loc12_17.1: ref i32 = tuple_access %a.ref, element0
// CHECK:STDOUT:   %.loc12_17.2: i32 = bind_value %.loc12_17.1
// CHECK:STDOUT:   %.loc12_17.3: init (i32,) = tuple_init (%.loc12_17.2) to file.%b.var
// CHECK:STDOUT:   %.loc12_18: init (i32,) = converted %a.ref, %.loc12_17.3
// CHECK:STDOUT:   assign file.%b.var, %.loc12_18
// CHECK:STDOUT:   %b.ref: ref (i32,) = name_ref b, file.%b
// CHECK:STDOUT:   %.loc13_16: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_17.1: ref i32 = tuple_index %b.ref, %.loc13_16
// CHECK:STDOUT:   %.loc13_17.2: i32 = bind_value %.loc13_17.1
// CHECK:STDOUT:   assign file.%c.var, %.loc13_17.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
