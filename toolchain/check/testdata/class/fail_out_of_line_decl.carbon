// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C {}

// CHECK:STDERR: fail_out_of_line_decl.carbon:[[@LINE+3]]:6: ERROR: Out-of-line declaration requires a declaration in scoped entity.
// CHECK:STDERR: fn C.F() {}
// CHECK:STDERR:      ^
fn C.F() {}

// CHECK:STDOUT: --- fail_out_of_line_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc7: type = struct_type {}, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.C = %C.decl}
// CHECK:STDOUT:   %C.decl = class_decl @C, ()
// CHECK:STDOUT:   %C: type = class_type @C, const
// CHECK:STDOUT:   %F: <function> = fn_decl @F, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = file.%F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
