// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  var field: i32;
  fn F() -> i32 {
    // CHECK:STDERR: fail_unbound_field.carbon:[[@LINE+3]]:12: ERROR: Expression cannot be used as a value.
    // CHECK:STDERR:     return field;
    // CHECK:STDERR:            ^~~~~
    return field;
  }
}

fn G() -> i32 {
  // CHECK:STDERR: fail_unbound_field.carbon:[[@LINE+3]]:10: ERROR: Expression cannot be used as a value.
  // CHECK:STDERR:   return Class.field;
  // CHECK:STDERR:          ^~~~~~~
  return Class.field;
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc15: type = struct_type {.field: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_unbound_field.carbon" {
// CHECK:STDOUT:   class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc8_12.1: type = unbound_field_type Class, i32
// CHECK:STDOUT:   %.loc8_12.2: <unbound field of class Class> = field field, member0
// CHECK:STDOUT:   %field: <unbound field of class Class> = bind_name field, %.loc8_12.2
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .field = %field
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %field.ref: <unbound field of class Class> = name_ref field, @Class.%field
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class
// CHECK:STDOUT:   %field.ref: <unbound field of class Class> = name_ref field, @Class.%field
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
