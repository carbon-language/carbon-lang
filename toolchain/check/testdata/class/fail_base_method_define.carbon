// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class B {
  fn F();

  class C {
    fn F();
  }
}

class D {
  extend base: B;
}

// CHECK:STDERR: fail_base_method_define.carbon:[[@LINE+3]]:6: ERROR: Out-of-line declaration requires a declaration in scoped entity.
// CHECK:STDERR: fn D.F() {}
// CHECK:STDERR:      ^
fn D.F() {}

// CHECK:STDERR: fail_base_method_define.carbon:[[@LINE+3]]:6: ERROR: Name `C` not found.
// CHECK:STDERR: fn D.C.F() {}
// CHECK:STDERR:      ^
fn D.C.F() {}

// CHECK:STDOUT: --- fail_base_method_define.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type D, B [template]
// CHECK:STDOUT:   %.5: type = struct_type {.base: B} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.B = %B.decl, .D = %D.decl} [template]
// CHECK:STDOUT:   %B.decl = class_decl @B, ()
// CHECK:STDOUT:   %D.decl = class_decl @D, ()
// CHECK:STDOUT:   %F: <function> = fn_decl @F.3 [template]
// CHECK:STDOUT:   %.loc27: <function> = fn_decl @.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 [template]
// CHECK:STDOUT:   %C.decl = class_decl @C, ()
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .C = %C.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %B.ref: type = name_ref B, constants.%B [template = constants.%B]
// CHECK:STDOUT:   %.loc16: <unbound element of class D> = base_decl B, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   .F = file.%F
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
