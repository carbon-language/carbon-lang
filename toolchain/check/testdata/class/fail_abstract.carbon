// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

abstract class Class {
  var k: i32;
}

fn Make() -> Class {
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE+3]]:10: ERROR: Cannot construct instance of abstract class. Consider using `partial Class` instead.
  // CHECK:STDERR:   return {.k = 1};
  // CHECK:STDERR:          ^~~~~~~~
  return {.k = 1};
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc9: type = struct_type {.k: i32}
// CHECK:STDOUT:   %.loc7: type = ptr_type {.k: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_abstract.carbon" {
// CHECK:STDOUT:   class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %Make: <function> = fn_decl @Make
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc8_8.1: type = unbound_field_type Class, i32
// CHECK:STDOUT:   %.loc8_8.2: <unbound field of class Class> = field k, member0
// CHECK:STDOUT:   %k: <unbound field of class Class> = bind_name k, %.loc8_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .k = %k
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_16: i32 = int_literal 1
// CHECK:STDOUT:   %.loc15_17: {.k: i32} = struct_literal (%.loc15_16)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
