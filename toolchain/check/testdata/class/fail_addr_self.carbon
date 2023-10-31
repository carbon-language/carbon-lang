// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F[addr self: Class*]();
  fn G[addr self: Class]();
}

fn F(c: Class) {
  // TODO: This is the wrong reason to reject this. We should complain that `c` is not an lvalue.
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE+6]]:6: ERROR: Cannot implicitly convert from `Class` to `Class*`.
  // CHECK:STDERR:   c.F();
  // CHECK:STDERR:      ^
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE-9]]:13: Initializing self parameter of method declared here.
  // CHECK:STDERR:   fn F[addr self: Class*]();
  // CHECK:STDERR:             ^
  c.F();

  // TODO: This is invalid and should be rejected.
  c.G();
}

// CHECK:STDOUT: file "fail_addr_self.carbon" {
// CHECK:STDOUT:   class_declaration @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %.loc10: type = struct_type {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[%self.addr: Class*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[%self.addr: Class]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%c: Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10: type = tuple_type ()
// CHECK:STDOUT:   %.loc7: type = ptr_type {}
// CHECK:STDOUT:   %c.ref.loc20: Class = name_reference "c", %c
// CHECK:STDOUT:   %.loc20_4: <bound method> = bound_method %c.ref.loc20, @Class.%F
// CHECK:STDOUT:   %.loc20_6: init () = call %.loc20_4(<invalid>)
// CHECK:STDOUT:   %c.ref.loc23: Class = name_reference "c", %c
// CHECK:STDOUT:   %.loc23_4: <bound method> = bound_method %c.ref.loc23, @Class.%G
// CHECK:STDOUT:   %.loc23_6: init () = call %.loc23_4(%c.ref.loc23)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
