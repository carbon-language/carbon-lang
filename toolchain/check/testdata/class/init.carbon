// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  var n: i32;
  var next: Class*;
}

fn Make(n: i32, next: Class*) -> Class {
  return {.n = n, .next = next};
}

fn MakeReorder(n: i32, next: Class*) -> Class {
  return {.next = next, .n = n};
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc10_1.1: type = struct_type {.n: i32, .next: Class*}
// CHECK:STDOUT:   %.loc10_1.2: type = ptr_type {.n: i32, .next: Class*}
// CHECK:STDOUT:   %.loc17: type = struct_type {.next: Class*, .n: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "init.carbon" {
// CHECK:STDOUT:   class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %Make: <function> = fn_decl @Make
// CHECK:STDOUT:   %MakeReorder: <function> = fn_decl @MakeReorder
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc8_8.1: type = unbound_field_type Class, i32
// CHECK:STDOUT:   %.loc8_8.2: <unbound field of class Class> = field n, member0
// CHECK:STDOUT:   %n: <unbound field of class Class> = bind_name n, %.loc8_8.2
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class
// CHECK:STDOUT:   %.loc9_18: type = ptr_type Class
// CHECK:STDOUT:   %.loc9_11.1: type = unbound_field_type Class, Class*
// CHECK:STDOUT:   %.loc9_11.2: <unbound field of class Class> = field next, member1
// CHECK:STDOUT:   %next: <unbound field of class Class> = bind_name next, %.loc9_11.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .n = %n
// CHECK:STDOUT:   .next = %next
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make(%n: i32, %next: Class*) -> %return: Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   %next.ref: Class* = name_ref next, %next
// CHECK:STDOUT:   %.loc13_31.1: {.n: i32, .next: Class*} = struct_literal (%n.ref, %next.ref)
// CHECK:STDOUT:   %.loc13_31.2: ref i32 = class_field_access %return, member0
// CHECK:STDOUT:   %.loc13_31.3: init i32 = initialize_from %n.ref to %.loc13_31.2
// CHECK:STDOUT:   %.loc13_31.4: ref Class* = class_field_access %return, member1
// CHECK:STDOUT:   %.loc13_31.5: init Class* = initialize_from %next.ref to %.loc13_31.4
// CHECK:STDOUT:   %.loc13_31.6: init Class = class_init (%.loc13_31.3, %.loc13_31.5), %return
// CHECK:STDOUT:   %.loc13_31.7: init Class = converted %.loc13_31.1, %.loc13_31.6
// CHECK:STDOUT:   return %.loc13_31.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeReorder(%n: i32, %next: Class*) -> %return: Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %next.ref: Class* = name_ref next, %next
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc17_31.1: {.next: Class*, .n: i32} = struct_literal (%next.ref, %n.ref)
// CHECK:STDOUT:   %.loc17_31.2: ref i32 = class_field_access %return, member1
// CHECK:STDOUT:   %.loc17_31.3: init i32 = initialize_from %n.ref to %.loc17_31.2
// CHECK:STDOUT:   %.loc17_31.4: ref Class* = class_field_access %return, member0
// CHECK:STDOUT:   %.loc17_31.5: init Class* = initialize_from %next.ref to %.loc17_31.4
// CHECK:STDOUT:   %.loc17_31.6: init Class = class_init (%.loc17_31.3, %.loc17_31.5), %return
// CHECK:STDOUT:   %.loc17_31.7: init Class = converted %.loc17_31.1, %.loc17_31.6
// CHECK:STDOUT:   return %.loc17_31.7
// CHECK:STDOUT: }
