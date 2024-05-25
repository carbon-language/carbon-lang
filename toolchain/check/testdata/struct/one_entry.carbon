// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: {.a: i32} = {.a = 4};
var y: {.a: i32} = x;

// CHECK:STDOUT: --- one_entry.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %struct.2: {.a: i32} = struct_value (%.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc7: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc7_13.1: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:   %.loc7_13.2: type = converted %int.make_type_32.loc7, %.loc7_13.1 [template = i32]
// CHECK:STDOUT:   %.loc7_16: type = struct_type {.a: i32} [template = constants.%.2]
// CHECK:STDOUT:   %x.var: ref {.a: i32} = var x
// CHECK:STDOUT:   %x: ref {.a: i32} = bind_name x, %x.var
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc8: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc8_13.1: type = value_of_initializer %int.make_type_32.loc8 [template = i32]
// CHECK:STDOUT:   %.loc8_13.2: type = converted %int.make_type_32.loc8, %.loc8_13.1 [template = i32]
// CHECK:STDOUT:   %.loc8_16: type = struct_type {.a: i32} [template = constants.%.2]
// CHECK:STDOUT:   %y.var: ref {.a: i32} = var y
// CHECK:STDOUT:   %y: ref {.a: i32} = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_26: i32 = int_literal 4 [template = constants.%.3]
// CHECK:STDOUT:   %.loc7_27.1: {.a: i32} = struct_literal (%.loc7_26)
// CHECK:STDOUT:   %.loc7_27.2: init {.a: i32} = struct_init (%.loc7_26) to file.%x.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc7_28: init {.a: i32} = converted %.loc7_27.1, %.loc7_27.2 [template = constants.%struct.2]
// CHECK:STDOUT:   assign file.%x.var, %.loc7_28
// CHECK:STDOUT:   %x.ref: ref {.a: i32} = name_ref x, file.%x
// CHECK:STDOUT:   %.loc8_20.1: ref i32 = struct_access %x.ref, element0
// CHECK:STDOUT:   %.loc8_20.2: i32 = bind_value %.loc8_20.1
// CHECK:STDOUT:   %.loc8_20.3: init {.a: i32} = struct_init (%.loc8_20.2) to file.%y.var
// CHECK:STDOUT:   %.loc8_21: init {.a: i32} = converted %x.ref, %.loc8_20.3
// CHECK:STDOUT:   assign file.%y.var, %.loc8_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
