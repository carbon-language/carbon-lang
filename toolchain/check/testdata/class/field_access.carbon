// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  var j: i32;
  var k: i32;
}

fn Run() {
  var c: Class;
  c.j = 1;
  c.k = 2;
  var cj: i32 = c.j;
  var ck: i32 = c.k;
}

// CHECK:STDOUT: --- field_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type Class, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.j: i32, .k: i32} [template]
// CHECK:STDOUT:   %Run: type = fn_type @Run [template]
// CHECK:STDOUT:   %struct.2: Run = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.j: i32, .k: i32} [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Run.decl: Run = fn_decl @Run [template = constants.%struct.2] {}
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %int.make_type_32.loc8: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc8_10.1: type = value_of_initializer %int.make_type_32.loc8 [template = i32]
// CHECK:STDOUT:   %.loc8_10.2: type = converted %int.make_type_32.loc8, %.loc8_10.1 [template = i32]
// CHECK:STDOUT:   %.loc8_8: <unbound element of class Class> = field_decl j, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc9: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc9_10.1: type = value_of_initializer %int.make_type_32.loc9 [template = i32]
// CHECK:STDOUT:   %.loc9_10.2: type = converted %int.make_type_32.loc9, %.loc9_10.1 [template = i32]
// CHECK:STDOUT:   %.loc9_8: <unbound element of class Class> = field_decl k, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .j = %.loc8_8
// CHECK:STDOUT:   .k = %.loc9_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %c.var: ref Class = var c
// CHECK:STDOUT:   %c: ref Class = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref.loc14: ref Class = name_ref c, %c
// CHECK:STDOUT:   %j.ref.loc14: <unbound element of class Class> = name_ref j, @Class.%.loc8_8 [template = @Class.%.loc8_8]
// CHECK:STDOUT:   %.loc14_4: ref i32 = class_element_access %c.ref.loc14, element0
// CHECK:STDOUT:   %.loc14_9: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   assign %.loc14_4, %.loc14_9
// CHECK:STDOUT:   %c.ref.loc15: ref Class = name_ref c, %c
// CHECK:STDOUT:   %k.ref.loc15: <unbound element of class Class> = name_ref k, @Class.%.loc9_8 [template = @Class.%.loc9_8]
// CHECK:STDOUT:   %.loc15_4: ref i32 = class_element_access %c.ref.loc15, element1
// CHECK:STDOUT:   %.loc15_9: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   assign %.loc15_4, %.loc15_9
// CHECK:STDOUT:   %int.make_type_32.loc16: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc16_11.1: type = value_of_initializer %int.make_type_32.loc16 [template = i32]
// CHECK:STDOUT:   %.loc16_11.2: type = converted %int.make_type_32.loc16, %.loc16_11.1 [template = i32]
// CHECK:STDOUT:   %cj.var: ref i32 = var cj
// CHECK:STDOUT:   %cj: ref i32 = bind_name cj, %cj.var
// CHECK:STDOUT:   %c.ref.loc16: ref Class = name_ref c, %c
// CHECK:STDOUT:   %j.ref.loc16: <unbound element of class Class> = name_ref j, @Class.%.loc8_8 [template = @Class.%.loc8_8]
// CHECK:STDOUT:   %.loc16_18.1: ref i32 = class_element_access %c.ref.loc16, element0
// CHECK:STDOUT:   %.loc16_18.2: i32 = bind_value %.loc16_18.1
// CHECK:STDOUT:   assign %cj.var, %.loc16_18.2
// CHECK:STDOUT:   %int.make_type_32.loc17: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc17_11.1: type = value_of_initializer %int.make_type_32.loc17 [template = i32]
// CHECK:STDOUT:   %.loc17_11.2: type = converted %int.make_type_32.loc17, %.loc17_11.1 [template = i32]
// CHECK:STDOUT:   %ck.var: ref i32 = var ck
// CHECK:STDOUT:   %ck: ref i32 = bind_name ck, %ck.var
// CHECK:STDOUT:   %c.ref.loc17: ref Class = name_ref c, %c
// CHECK:STDOUT:   %k.ref.loc17: <unbound element of class Class> = name_ref k, @Class.%.loc9_8 [template = @Class.%.loc9_8]
// CHECK:STDOUT:   %.loc17_18.1: ref i32 = class_element_access %c.ref.loc17, element1
// CHECK:STDOUT:   %.loc17_18.2: i32 = bind_value %.loc17_18.1
// CHECK:STDOUT:   assign %ck.var, %.loc17_18.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
