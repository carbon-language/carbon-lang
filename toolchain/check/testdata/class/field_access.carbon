// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  var j: i32;
  var k: i32;
}

fn Run() -> i32 {
  var c: Class;
  c.j = 1;
  c.k = 2;
  return c.j + c.k;
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc10_1.1: type = struct_type {.j: i32, .k: i32}
// CHECK:STDOUT:   %.loc10_1.2: type = ptr_type {.j: i32, .k: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "field_access.carbon" {
// CHECK:STDOUT:   class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc8_8.1: type = unbound_field_type Class, i32
// CHECK:STDOUT:   %.loc8_8.2: <unbound field of class Class> = field j, member0
// CHECK:STDOUT:   %j: <unbound field of class Class> = bind_name j, %.loc8_8.2
// CHECK:STDOUT:   %.loc9_8.1: type = unbound_field_type Class, i32
// CHECK:STDOUT:   %.loc9_8.2: <unbound field of class Class> = field k, member1
// CHECK:STDOUT:   %k: <unbound field of class Class> = bind_name k, %.loc9_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .j = %j
// CHECK:STDOUT:   .k = %k
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class
// CHECK:STDOUT:   %c.var: ref Class = var c
// CHECK:STDOUT:   %c: ref Class = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref.loc14: ref Class = name_ref c, %c
// CHECK:STDOUT:   %.loc14_4: ref i32 = class_field_access %c.ref.loc14, member0
// CHECK:STDOUT:   %.loc14_9: i32 = int_literal 1
// CHECK:STDOUT:   assign %.loc14_4, %.loc14_9
// CHECK:STDOUT:   %c.ref.loc15: ref Class = name_ref c, %c
// CHECK:STDOUT:   %.loc15_4: ref i32 = class_field_access %c.ref.loc15, member1
// CHECK:STDOUT:   %.loc15_9: i32 = int_literal 2
// CHECK:STDOUT:   assign %.loc15_4, %.loc15_9
// CHECK:STDOUT:   %c.ref.loc16_10: ref Class = name_ref c, %c
// CHECK:STDOUT:   %.loc16_11.1: ref i32 = class_field_access %c.ref.loc16_10, member0
// CHECK:STDOUT:   %c.ref.loc16_16: ref Class = name_ref c, %c
// CHECK:STDOUT:   %.loc16_17.1: ref i32 = class_field_access %c.ref.loc16_16, member1
// CHECK:STDOUT:   %.loc16_11.2: i32 = bind_value %.loc16_11.1
// CHECK:STDOUT:   %.loc16_17.2: i32 = bind_value %.loc16_17.1
// CHECK:STDOUT:   %.loc16_14: i32 = add %.loc16_11.2, %.loc16_17.2
// CHECK:STDOUT:   return %.loc16_14
// CHECK:STDOUT: }
