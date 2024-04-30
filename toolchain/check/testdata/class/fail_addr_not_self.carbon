// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  // CHECK:STDERR: fail_addr_not_self.carbon:[[@LINE+4]]:8: ERROR: `addr` can only be applied to a `self` parameter.
  // CHECK:STDERR:   fn F[addr a: Class*]();
  // CHECK:STDERR:        ^~~~
  // CHECK:STDERR:
  fn F[addr a: Class*]();

  // CHECK:STDERR: fail_addr_not_self.carbon:[[@LINE+3]]:8: ERROR: `addr` can only be applied to a `self` parameter.
  // CHECK:STDERR:   fn G(addr b: Class*);
  // CHECK:STDERR:        ^~~~
  fn G(addr b: Class*);
}

// CHECK:STDOUT: --- fail_addr_not_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = ptr_type Class [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.2: G = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
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
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {
// CHECK:STDOUT:     %Class.ref.loc12: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc12: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:     %a.loc12_13.1: Class* = param a
// CHECK:STDOUT:     %a.loc12_13.2: Class* = bind_name a, %a.loc12_13.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.2] {
// CHECK:STDOUT:     %Class.ref.loc17: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc17: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:     %b.loc17_13.1: Class* = param b
// CHECK:STDOUT:     %b.loc17_13.2: Class* = bind_name b, %b.loc17_13.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[@Class.%a.loc12_13.2: Class*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(@Class.%b.loc17_13.2: Class*);
// CHECK:STDOUT:
