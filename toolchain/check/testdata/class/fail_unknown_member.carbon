// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  var n: i32;
}

fn G(c: Class) -> i32 {
  // TODO: Mention the scope in which we looked for the name.
  // CHECK:STDERR: fail_unknown_member.carbon:[[@LINE+3]]:10: ERROR: Name `something` not found.
  // CHECK:STDERR:   return c.something;
  // CHECK:STDERR:          ^~~~~~~~~~~
  return c.something;
}

// CHECK:STDOUT: --- fail_unknown_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc9: type = struct_type {.n: i32}, const
// CHECK:STDOUT:   %.loc7: type = ptr_type {.n: i32}, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.Class = %Class.decl, .G = %G}
// CHECK:STDOUT:   %Class.decl = class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class, const
// CHECK:STDOUT:   %G: <function> = fn_decl @G, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc8_8.1: type = unbound_element_type Class, i32, const
// CHECK:STDOUT:   %.loc8_8.2: <unbound element of class Class> = field_decl n, element0, const
// CHECK:STDOUT:   %n: <unbound element of class Class> = bind_name n, %.loc8_8.2, const = %.loc8_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .n = %n
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%c: Class) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: Class = name_ref c, %c
// CHECK:STDOUT:   %something.ref: <error> = name_ref something, <error>
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
