// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class;

fn F(p: Class*) -> Class* { return p; }

// CHECK:STDOUT: file "forward_declared.carbon" {
// CHECK:STDOUT:   %Class: type = class_declaration @Class, ()
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p: Class*) -> Class* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Class* = name_reference "p", %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
