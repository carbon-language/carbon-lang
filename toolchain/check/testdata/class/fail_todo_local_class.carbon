// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// TODO: Support parsing classes at function scope.
class A {
  fn F() {
    // CHECK:STDERR: fail_todo_local_class.carbon:[[@LINE+7]]:5: ERROR: Expected expression.
    // CHECK:STDERR:     class B {
    // CHECK:STDERR:     ^~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_todo_local_class.carbon:[[@LINE+3]]:5: ERROR: Semantics TODO: `HandleInvalidParse`.
    // CHECK:STDERR:     class B {
    // CHECK:STDERR:     ^~~~~
    class B {
      fn G() {
        var b: B = {};
      }
    }
  }
}

// CHECK:STDOUT: --- fail_todo_local_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: F = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT: }
// CHECK:STDOUT:
