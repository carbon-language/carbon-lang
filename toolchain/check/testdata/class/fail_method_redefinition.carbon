// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F() {}
  // CHECK:STDERR: fail_method_redefinition.carbon:[[@LINE+6]]:3: ERROR: Redefinition of `fn F`.
  // CHECK:STDERR:   fn F() {}
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_method_redefinition.carbon:[[@LINE-4]]:3: Previously defined here.
  // CHECK:STDERR:   fn F() {}
  // CHECK:STDERR:   ^~~~~~~~
  fn F() {}
}

// CHECK:STDOUT: --- fail_method_redefinition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: F = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl.loc8: F = fn_decl @F [template = constants.%struct] {}
// CHECK:STDOUT:   %F.decl.loc15: F = fn_decl @F [template = constants.%struct] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc15:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
