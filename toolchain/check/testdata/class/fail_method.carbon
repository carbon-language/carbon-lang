// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F[self: Class]() -> i32;

  var k: i32;
}

fn Class.F[self: Class]() -> i32 {
  return self.k;
}

fn Call(c: Class) -> i32 {
  // CHECK:STDERR: fail_method.carbon:[[@LINE+3]]:13: ERROR: Value of type <bound method> is not callable.
  // CHECK:STDERR:   return c.F();
  // CHECK:STDERR:             ^
  return c.F();
}

// CHECK:STDOUT: file "fail_method.carbon" {
// CHECK:STDOUT:   class_declaration @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %.loc11: type = struct_type {.k: i32}
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %Call: <function> = fn_decl @Call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %.loc10_8.1: type = unbound_field_type Class, i32
// CHECK:STDOUT:   %.loc10_8.2: <unbound field of class Class> = field "k", member0
// CHECK:STDOUT:   %k: <unbound field of class Class> = bind_name "k", %.loc10_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .k = %k
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: Class]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: type = ptr_type {.k: i32}
// CHECK:STDOUT:   %self.ref: Class = name_reference "self", %self
// CHECK:STDOUT:   %.loc14_14.1: ref i32 = class_field_access %self.ref, member0
// CHECK:STDOUT:   %.loc14_14.2: i32 = bind_value %.loc14_14.1
// CHECK:STDOUT:   return %.loc14_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%c: Class) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: Class = name_reference "c", %c
// CHECK:STDOUT:   %.loc21: <bound method> = bound_method %c.ref, @Class.%F
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
