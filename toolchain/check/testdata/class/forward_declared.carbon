// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class;

fn F(p: Class*) -> Class* { return p; }

// CHECK:STDOUT: --- forward_declared.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = ptr_type Class [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %Class.ref.loc9_9: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc9_14: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:     %p.loc9_6.1: Class* = param p
// CHECK:STDOUT:     @F.%p: Class* = bind_name p, %p.loc9_6.1
// CHECK:STDOUT:     %Class.ref.loc9_20: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc9_25: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:     %return.var: ref Class* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p: Class*) -> Class* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Class* = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
