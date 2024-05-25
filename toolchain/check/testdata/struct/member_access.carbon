// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
var y: i32 = x.b;
var z: i32 = y;

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %Float: type = fn_type @Float [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Float = struct_value () [template]
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %struct.2: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: f64, .b: i32} [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.a: f64, .b: i32} [template]
// CHECK:STDOUT:   %.5: f64 = float_literal 0 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %struct.3: {.a: f64, .b: i32} = struct_value (%.5, %.6) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:     .z = %z
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc7_13.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:   %import_ref.1: Float = import_ref ir3, inst+31, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%struct.1(%.loc7_13.1) [template = f64]
// CHECK:STDOUT:   %.loc7_13.2: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc7_13.3: type = converted %float.make_type, %.loc7_13.2 [template = f64]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %int.make_type_32.loc7: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %.loc7_22.1: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:   %.loc7_22.2: type = converted %int.make_type_32.loc7, %.loc7_22.1 [template = i32]
// CHECK:STDOUT:   %.loc7_25: type = struct_type {.a: f64, .b: i32} [template = constants.%.3]
// CHECK:STDOUT:   %x.var: ref {.a: f64, .b: i32} = var x
// CHECK:STDOUT:   %x: ref {.a: f64, .b: i32} = bind_name x, %x.var
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %int.make_type_32.loc8: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %.loc8_8.1: type = value_of_initializer %int.make_type_32.loc8 [template = i32]
// CHECK:STDOUT:   %.loc8_8.2: type = converted %int.make_type_32.loc8, %.loc8_8.1 [template = i32]
// CHECK:STDOUT:   %y.var: ref i32 = var y
// CHECK:STDOUT:   %y: ref i32 = bind_name y, %y.var
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %int.make_type_32.loc9: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %.loc9_8.1: type = value_of_initializer %int.make_type_32.loc9 [template = i32]
// CHECK:STDOUT:   %.loc9_8.2: type = converted %int.make_type_32.loc9, %.loc9_8.1 [template = i32]
// CHECK:STDOUT:   %z.var: ref i32 = var z
// CHECK:STDOUT:   %z: ref i32 = bind_name z, %z.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_35: f64 = float_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_45: i32 = int_literal 1 [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_46.1: {.a: f64, .b: i32} = struct_literal (%.loc7_35, %.loc7_45)
// CHECK:STDOUT:   %.loc7_46.2: ref f64 = struct_access file.%x.var, element0
// CHECK:STDOUT:   %.loc7_46.3: init f64 = initialize_from %.loc7_35 to %.loc7_46.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_46.4: ref i32 = struct_access file.%x.var, element1
// CHECK:STDOUT:   %.loc7_46.5: init i32 = initialize_from %.loc7_45 to %.loc7_46.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_46.6: init {.a: f64, .b: i32} = struct_init (%.loc7_46.3, %.loc7_46.5) to file.%x.var [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc7_47: init {.a: f64, .b: i32} = converted %.loc7_46.1, %.loc7_46.6 [template = constants.%struct.3]
// CHECK:STDOUT:   assign file.%x.var, %.loc7_47
// CHECK:STDOUT:   %x.ref: ref {.a: f64, .b: i32} = name_ref x, file.%x
// CHECK:STDOUT:   %.loc8_15.1: ref i32 = struct_access %x.ref, element1
// CHECK:STDOUT:   %.loc8_15.2: i32 = bind_value %.loc8_15.1
// CHECK:STDOUT:   assign file.%y.var, %.loc8_15.2
// CHECK:STDOUT:   %y.ref: ref i32 = name_ref y, file.%y
// CHECK:STDOUT:   %.loc9: i32 = bind_value %y.ref
// CHECK:STDOUT:   assign file.%z.var, %.loc9
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
