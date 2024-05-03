// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F() -> i32 {
    return 1;
  }
}

fn G() -> i32 {
  // CHECK:STDERR: fail_scope.carbon:[[@LINE+3]]:10: ERROR: Name `F` not found.
  // CHECK:STDERR:   return F();
  // CHECK:STDERR:          ^
  return F();
}

// CHECK:STDOUT: --- fail_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.2: G = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.2] {
// CHECK:STDOUT:     @G.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   return %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
