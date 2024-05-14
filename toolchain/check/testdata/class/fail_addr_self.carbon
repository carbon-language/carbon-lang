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
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE+7]]:6: ERROR: `addr self` method cannot be invoked on a value.
  // CHECK:STDERR:   c.F();
  // CHECK:STDERR:      ^
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE-8]]:13: Initializing `addr self` parameter of method declared here.
  // CHECK:STDERR:   fn F[addr self: Class*]();
  // CHECK:STDERR:             ^~~~
  // CHECK:STDERR:
  c.F();

  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE+7]]:6: ERROR: `addr self` method cannot be invoked on a value.
  // CHECK:STDERR:   c.G();
  // CHECK:STDERR:      ^
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE-16]]:13: Initializing `addr self` parameter of method declared here.
  // CHECK:STDERR:   fn G[addr self: Class]();
  // CHECK:STDERR:             ^~~~
  // CHECK:STDERR:
  c.G();

  // This call is OK.
  (*p).F();

  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE+6]]:3: ERROR: Cannot implicitly convert from `Class*` to `Class`.
  // CHECK:STDERR:   (*p).G();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE-28]]:13: Initializing `addr self` parameter of method declared here.
  // CHECK:STDERR:   fn G[addr self: Class]();
  // CHECK:STDERR:             ^~~~
  (*p).G();
}

// CHECK:STDOUT: --- fail_addr_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = ptr_type Class [template]
// CHECK:STDOUT:   %F.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.2: G = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %F.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %struct.3: F = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.2 [template = constants.%struct.3] {
// CHECK:STDOUT:     %Class.ref.loc12_9: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %c.loc12_6.1: Class = param c
// CHECK:STDOUT:     @F.2.%c: Class = bind_name c, %c.loc12_6.1
// CHECK:STDOUT:     %Class.ref.loc12_19: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc12: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:     %p.loc12_16.1: Class* = param p
// CHECK:STDOUT:     @F.2.%p: Class* = bind_name p, %p.loc12_16.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.1 [template = constants.%struct.1] {
// CHECK:STDOUT:     %Class.ref.loc8: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc8_24: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:     %self.loc8_13.1: Class* = param self
// CHECK:STDOUT:     %self.loc8_13.3: Class* = bind_name self, %self.loc8_13.1
// CHECK:STDOUT:     %.loc8_8: Class* = addr_pattern %self.loc8_13.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.2] {
// CHECK:STDOUT:     %Class.ref.loc9: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %self.loc9_13.1: Class = param self
// CHECK:STDOUT:     %self.loc9_13.3: Class = bind_name self, %self.loc9_13.1
// CHECK:STDOUT:     %.loc9: Class = addr_pattern %self.loc9_13.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[addr @Class.%self.loc8_13.3: Class*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[addr @Class.%self.loc9_13.3: Class]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%c: Class, %p: Class*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref.loc20: Class = name_ref c, %c
// CHECK:STDOUT:   %F.ref.loc20: F = name_ref F, @Class.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc20: <bound method> = bound_method %c.ref.loc20, %F.ref.loc20
// CHECK:STDOUT:   %F.call.loc20: init () = call %.loc20(<invalid>) [template = <error>]
// CHECK:STDOUT:   %c.ref.loc29: Class = name_ref c, %c
// CHECK:STDOUT:   %G.ref.loc29: G = name_ref G, @Class.%G.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc29: <bound method> = bound_method %c.ref.loc29, %G.ref.loc29
// CHECK:STDOUT:   %G.call.loc29: init () = call %.loc29(<invalid>) [template = <error>]
// CHECK:STDOUT:   %p.ref.loc32: Class* = name_ref p, %p
// CHECK:STDOUT:   %.loc32_4.1: ref Class = deref %p.ref.loc32
// CHECK:STDOUT:   %F.ref.loc32: F = name_ref F, @Class.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc32_7: <bound method> = bound_method %.loc32_4.1, %F.ref.loc32
// CHECK:STDOUT:   %.loc32_4.2: Class* = addr_of %.loc32_4.1
// CHECK:STDOUT:   %F.call.loc32: init () = call %.loc32_7(%.loc32_4.2)
// CHECK:STDOUT:   %p.ref.loc40: Class* = name_ref p, %p
// CHECK:STDOUT:   %.loc40_4.1: ref Class = deref %p.ref.loc40
// CHECK:STDOUT:   %G.ref.loc40: G = name_ref G, @Class.%G.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc40_7: <bound method> = bound_method %.loc40_4.1, %G.ref.loc40
// CHECK:STDOUT:   %.loc40_4.2: Class* = addr_of %.loc40_4.1
// CHECK:STDOUT:   %G.call.loc40: init () = call %.loc40_7(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
