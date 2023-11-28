// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F[addr self: Class*]();
  fn G[addr self: Class]();
}

fn F(c: Class, p: Class*) {
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE+6]]:3: ERROR: `addr self` method cannot be invoked on a value.
  // CHECK:STDERR:   c.F();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE-8]]:13: Initializing `addr self` parameter of method declared here.
  // CHECK:STDERR:   fn F[addr self: Class*]();
  // CHECK:STDERR:             ^~~~
  c.F();

  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE+6]]:3: ERROR: `addr self` method cannot be invoked on a value.
  // CHECK:STDERR:   c.G();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE-15]]:13: Initializing `addr self` parameter of method declared here.
  // CHECK:STDERR:   fn G[addr self: Class]();
  // CHECK:STDERR:             ^~~~
  c.G();

  // This call is OK.
  (*p).F();

  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE+6]]:3: ERROR: Cannot implicitly convert from `Class*` to `Class`.
  // CHECK:STDERR:   (*p).G();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE-26]]:13: Initializing `addr self` parameter of method declared here.
  // CHECK:STDERR:   fn G[addr self: Class]();
  // CHECK:STDERR:             ^~~~
  (*p).G();
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc10_1.1: type = struct_type {}
// CHECK:STDOUT:   %.loc10_1.2: type = tuple_type ()
// CHECK:STDOUT:   %.loc7: type = ptr_type {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_addr_self.carbon" {
// CHECK:STDOUT:   class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
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
// CHECK:STDOUT: fn @F.2(%c: Class, %p: Class*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref.loc19: Class = name_ref c, %c
// CHECK:STDOUT:   %.loc19_4: <bound method> = bound_method %c.ref.loc19, @Class.%F
// CHECK:STDOUT:   %.loc19_6: init () = call %.loc19_4(<invalid>)
// CHECK:STDOUT:   %c.ref.loc27: Class = name_ref c, %c
// CHECK:STDOUT:   %.loc27_4: <bound method> = bound_method %c.ref.loc27, @Class.%G
// CHECK:STDOUT:   %.loc27_6: init () = call %.loc27_4(<invalid>)
// CHECK:STDOUT:   %p.ref.loc30: Class* = name_ref p, %p
// CHECK:STDOUT:   %.loc30_4.1: ref Class = deref %p.ref.loc30
// CHECK:STDOUT:   %.loc30_7: <bound method> = bound_method %.loc30_4.1, @Class.%F
// CHECK:STDOUT:   %.loc30_4.2: Class* = address_of %.loc30_4.1
// CHECK:STDOUT:   %.loc30_9: init () = call %.loc30_7(%.loc30_4.2)
// CHECK:STDOUT:   %p.ref.loc38: Class* = name_ref p, %p
// CHECK:STDOUT:   %.loc38_4.1: ref Class = deref %p.ref.loc38
// CHECK:STDOUT:   %.loc38_7: <bound method> = bound_method %.loc38_4.1, @Class.%G
// CHECK:STDOUT:   %.loc38_4.2: Class* = address_of %.loc38_4.1
// CHECK:STDOUT:   %.loc38_9: init () = call %.loc38_7(<invalid>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
