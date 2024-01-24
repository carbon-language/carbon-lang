// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn G() -> i32 {
    // TODO: This should find the member function `F` even though it's declared
    // later.
    // CHECK:STDERR: fail_reorder.carbon:[[@LINE+9]]:12: ERROR: Member access into incomplete class `Class`.
    // CHECK:STDERR:     return Class.F();
    // CHECK:STDERR:            ^~~~~
    // CHECK:STDERR: fail_reorder.carbon:[[@LINE-7]]:1: Class is incomplete within its definition.
    // CHECK:STDERR: class Class {
    // CHECK:STDERR: ^~~~~~~~~~~~~
    // CHECK:STDERR: fail_reorder.carbon:[[@LINE+3]]:12: ERROR: Name `F` not found.
    // CHECK:STDERR:     return Class.F();
    // CHECK:STDERR:            ^~~~~~~
    return Class.F();
  }

  fn F() -> i32 {
    return 1;
  }
}

// CHECK:STDOUT: --- fail_reorder.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Class = %Class.decl} [template]
// CHECK:STDOUT:   %Class.decl = class_decl @Class, ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   return %.loc24
// CHECK:STDOUT: }
// CHECK:STDOUT:
