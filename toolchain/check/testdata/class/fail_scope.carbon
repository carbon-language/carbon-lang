// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F() -> i32 {
    return 1;
  }

  // TODO: This `F()` should find `Class.F`.
  fn G() -> i32 {
    // CHECK:STDERR: fail_scope.carbon:[[@LINE+6]]:12: ERROR: Name `F` not found.
    // CHECK:STDERR:     return F();
    // CHECK:STDERR:            ^
    // CHECK:STDERR: fail_scope.carbon:[[@LINE+3]]:14: ERROR: Semantics TODO: `Not a callable name`.
    // CHECK:STDERR:     return F();
    // CHECK:STDERR:              ^
    return F();
  }
}

// CHECK:STDOUT: file "fail_scope.carbon" {
// CHECK:STDOUT:   %Class: type = class_declaration @Class, ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9: i32 = int_literal 1
// CHECK:STDOUT:   return %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <error> = name_reference "F", <error>
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
