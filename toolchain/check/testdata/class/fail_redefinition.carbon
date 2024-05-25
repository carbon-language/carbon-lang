// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F();
  fn H();
  fn I() {}
}

// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+7]]:1: ERROR: Redefinition of `class Class`.
// CHECK:STDERR: class Class {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE-9]]:1: Previously defined here.
// CHECK:STDERR: class Class {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
class Class {
  fn G();
  fn H();
  fn I() {}
}

fn Class.F() {}
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+4]]:10: ERROR: Out-of-line declaration requires a declaration in scoped entity.
// CHECK:STDERR: fn Class.G() {}
// CHECK:STDERR:          ^
// CHECK:STDERR:
fn Class.G() {}
fn Class.H() {}
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+6]]:1: ERROR: Redefinition of `fn I`.
// CHECK:STDERR: fn Class.I() {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE-26]]:3: Previously defined here.
// CHECK:STDERR:   fn I() {}
// CHECK:STDERR:   ^~~~~~~~
fn Class.I() {}

// CHECK:STDOUT: --- fail_redefinition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %H.1: type = fn_type @H.1 [template]
// CHECK:STDOUT:   %struct.2: H = struct_value () [template]
// CHECK:STDOUT:   %I.1: type = fn_type @I.1 [template]
// CHECK:STDOUT:   %struct.3: I = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = class_type @.1 [template]
// CHECK:STDOUT:   %G.1: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %struct.4: G = struct_value () [template]
// CHECK:STDOUT:   %H.2: type = fn_type @H.2 [template]
// CHECK:STDOUT:   %struct.5: H = struct_value () [template]
// CHECK:STDOUT:   %I.2: type = fn_type @I.2 [template]
// CHECK:STDOUT:   %struct.6: I = struct_value () [template]
// CHECK:STDOUT:   %G.2: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %struct.7: G = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %.decl: type = class_decl @.1 [template = constants.%.3] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {}
// CHECK:STDOUT:   %G.decl: G = fn_decl @G.2 [template = constants.%struct.7] {}
// CHECK:STDOUT:   %H.decl: H = fn_decl @H.1 [template = constants.%struct.2] {}
// CHECK:STDOUT:   %I.decl: I = fn_decl @I.1 [template = constants.%struct.3] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {}
// CHECK:STDOUT:   %H.decl: H = fn_decl @H.1 [template = constants.%struct.2] {}
// CHECK:STDOUT:   %I.decl: I = fn_decl @I.1 [template = constants.%struct.3] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   .I = %I.decl
// CHECK:STDOUT:   .G = file.%G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT:   %G.decl: G = fn_decl @G.1 [template = constants.%struct.4] {}
// CHECK:STDOUT:   %H.decl: H = fn_decl @H.2 [template = constants.%struct.5] {}
// CHECK:STDOUT:   %I.decl: I = fn_decl @I.2 [template = constants.%struct.6] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.3
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   .I = %I.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc39:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.2();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
