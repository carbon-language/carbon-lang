// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: [i32; 2] = (12, 24);
var b: i32 = 1;
var c: i32 = a[0];
var d: i32 = a[b];

// CHECK:STDOUT: --- array_element_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type [i32; 2] [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 24 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %array: [i32; 2] = tuple_value (%.5, %.6) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc7: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc7_14: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_9.1: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:   %.loc7_9.2: type = converted %int.make_type_32.loc7, %.loc7_9.1 [template = i32]
// CHECK:STDOUT:   %.loc7_15: type = array_type %.loc7_14, i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref [i32; 2] = var a
// CHECK:STDOUT:   %a: ref [i32; 2] = bind_name a, %a.var
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc8: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc8_8.1: type = value_of_initializer %int.make_type_32.loc8 [template = i32]
// CHECK:STDOUT:   %.loc8_8.2: type = converted %int.make_type_32.loc8, %.loc8_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc9: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc9_8.1: type = value_of_initializer %int.make_type_32.loc9 [template = i32]
// CHECK:STDOUT:   %.loc9_8.2: type = converted %int.make_type_32.loc9, %.loc9_8.1 [template = i32]
// CHECK:STDOUT:   %c.var: ref i32 = var c
// CHECK:STDOUT:   %c: ref i32 = bind_name c, %c.var
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc10: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc10_8.1: type = value_of_initializer %int.make_type_32.loc10 [template = i32]
// CHECK:STDOUT:   %.loc10_8.2: type = converted %int.make_type_32.loc10, %.loc10_8.1 [template = i32]
// CHECK:STDOUT:   %d.var: ref i32 = var d
// CHECK:STDOUT:   %d: ref i32 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_20: i32 = int_literal 12 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_24: i32 = int_literal 24 [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_26.1: (i32, i32) = tuple_literal (%.loc7_20, %.loc7_24)
// CHECK:STDOUT:   %.loc7_26.2: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc7_26.3: ref i32 = array_index file.%a.var, %.loc7_26.2
// CHECK:STDOUT:   %.loc7_26.4: init i32 = initialize_from %.loc7_20 to %.loc7_26.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_26.5: i32 = int_literal 1 [template = constants.%.9]
// CHECK:STDOUT:   %.loc7_26.6: ref i32 = array_index file.%a.var, %.loc7_26.5
// CHECK:STDOUT:   %.loc7_26.7: init i32 = initialize_from %.loc7_24 to %.loc7_26.6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_26.8: init [i32; 2] = array_init (%.loc7_26.4, %.loc7_26.7) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc7_27: init [i32; 2] = converted %.loc7_26.1, %.loc7_26.8 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc7_27
// CHECK:STDOUT:   %.loc8: i32 = int_literal 1 [template = constants.%.9]
// CHECK:STDOUT:   assign file.%b.var, %.loc8
// CHECK:STDOUT:   %a.ref.loc9: ref [i32; 2] = name_ref a, file.%a
// CHECK:STDOUT:   %.loc9_16: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc9_17.1: ref i32 = array_index %a.ref.loc9, %.loc9_16
// CHECK:STDOUT:   %.loc9_17.2: i32 = bind_value %.loc9_17.1
// CHECK:STDOUT:   assign file.%c.var, %.loc9_17.2
// CHECK:STDOUT:   %a.ref.loc10: ref [i32; 2] = name_ref a, file.%a
// CHECK:STDOUT:   %b.ref: ref i32 = name_ref b, file.%b
// CHECK:STDOUT:   %.loc10_16: i32 = bind_value %b.ref
// CHECK:STDOUT:   %.loc10_17.1: ref i32 = array_index %a.ref.loc10, %.loc10_16
// CHECK:STDOUT:   %.loc10_17.2: i32 = bind_value %.loc10_17.1
// CHECK:STDOUT:   assign file.%d.var, %.loc10_17.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
