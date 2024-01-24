// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F() {
    var r#Self: Self*;
    var p: Self* = r#Self;
  }
}

// CHECK:STDOUT: --- raw_self_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = ptr_type Class [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Class = %Class.decl} [template]
// CHECK:STDOUT:   %Class.decl = class_decl @Class, ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref.loc9: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %.loc9: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:   %Self.var: ref Class* = var r#Self
// CHECK:STDOUT:   %Self: ref Class* = bind_name r#Self, %Self.var
// CHECK:STDOUT:   %Self.ref.loc10_12: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %.loc10_16: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:   %p.var: ref Class* = var p
// CHECK:STDOUT:   %p: ref Class* = bind_name p, %p.var
// CHECK:STDOUT:   %Self.ref.loc10_20: ref Class* = name_ref r#Self, %Self
// CHECK:STDOUT:   %.loc10_20: Class* = bind_value %Self.ref.loc10_20
// CHECK:STDOUT:   assign %p.var, %.loc10_20
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
