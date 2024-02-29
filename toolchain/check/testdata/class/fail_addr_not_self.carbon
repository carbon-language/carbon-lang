// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  // CHECK:STDERR: fail_addr_not_self.carbon:[[@LINE+3]]:8: ERROR: `addr` can only be applied to a `self` parameter.
  // CHECK:STDERR:   fn F[addr a: Class*]();
  // CHECK:STDERR:        ^~~~
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
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.decl = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %Class.ref.loc11: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc11: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:     %a.loc11_13.1: Class* = param a
// CHECK:STDOUT:     %a.loc11_13.2: Class* = bind_name a, %a.loc11_13.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {
// CHECK:STDOUT:     %Class.ref.loc16: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc16: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:     %b.loc16_13.1: Class* = param b
// CHECK:STDOUT:     %b.loc16_13.2: Class* = bind_name b, %b.loc16_13.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[@Class.%a.loc11_13.2: Class*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(@Class.%b.loc16_13.2: Class*);
// CHECK:STDOUT:
