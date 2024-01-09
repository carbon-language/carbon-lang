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
// CHECK:STDOUT:   %.loc10_1.1: type = struct_type {.j: i32, .k: i32}
// CHECK:STDOUT:   %.loc10_1.2: type = ptr_type {.j: i32, .k: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Class = %Class.decl, .Run = %Run}
// CHECK:STDOUT:   %Class.decl = class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class, const
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc8_8.1: type = unbound_element_type Class, i32, const
// CHECK:STDOUT:   %.loc8_8.2: <unbound element of class Class> = field_decl j, element0, const
// CHECK:STDOUT:   %j: <unbound element of class Class> = bind_name j, %.loc8_8.2, const = %.loc8_8.2
// CHECK:STDOUT:   %.loc9_8.1: type = unbound_element_type Class, i32, const
// CHECK:STDOUT:   %.loc9_8.2: <unbound element of class Class> = field_decl k, element1, const
// CHECK:STDOUT:   %k: <unbound element of class Class> = bind_name k, %.loc9_8.2, const = %.loc9_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .j = %j
// CHECK:STDOUT:   .k = %k
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class, const = file.%Class
// CHECK:STDOUT:   %c.var: ref Class = var c
// CHECK:STDOUT:   %c: ref Class = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref.loc14: ref Class = name_ref c, %c
// CHECK:STDOUT:   %.loc14_4: ref i32 = class_element_access %c.ref.loc14, element0
// CHECK:STDOUT:   %.loc14_9: i32 = int_literal 1, const
// CHECK:STDOUT:   assign %.loc14_4, %.loc14_9
// CHECK:STDOUT:   %c.ref.loc15: ref Class = name_ref c, %c
// CHECK:STDOUT:   %.loc15_4: ref i32 = class_element_access %c.ref.loc15, element1
// CHECK:STDOUT:   %.loc15_9: i32 = int_literal 2, const
// CHECK:STDOUT:   assign %.loc15_4, %.loc15_9
// CHECK:STDOUT:   %cj.var: ref i32 = var cj
// CHECK:STDOUT:   %cj: ref i32 = bind_name cj, %cj.var
// CHECK:STDOUT:   %c.ref.loc16: ref Class = name_ref c, %c
// CHECK:STDOUT:   %.loc16_18.1: ref i32 = class_element_access %c.ref.loc16, element0
// CHECK:STDOUT:   %.loc16_18.2: i32 = bind_value %.loc16_18.1
// CHECK:STDOUT:   assign %cj.var, %.loc16_18.2
// CHECK:STDOUT:   %ck.var: ref i32 = var ck
// CHECK:STDOUT:   %ck: ref i32 = bind_name ck, %ck.var
// CHECK:STDOUT:   %c.ref.loc17: ref Class = name_ref c, %c
// CHECK:STDOUT:   %.loc17_18.1: ref i32 = class_element_access %c.ref.loc17, element1
// CHECK:STDOUT:   %.loc17_18.2: i32 = bind_value %.loc17_18.1
// CHECK:STDOUT:   assign %ck.var, %.loc17_18.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
