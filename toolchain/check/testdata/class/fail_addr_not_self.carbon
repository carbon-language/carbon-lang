// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  // CHECK:STDERR: fail_addr_not_self.carbon:[[@LINE+3]]:8: ERROR: `addr` can only be applied to a `self` parameter.
  // CHECK:STDERR:   fn F[addr a: Class*]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~~
  fn F[addr a: Class*]();

  // CHECK:STDERR: fail_addr_not_self.carbon:[[@LINE+3]]:8: ERROR: `addr` can only be applied to a `self` parameter.
  // CHECK:STDERR:   fn G(addr b: Class*);
  // CHECK:STDERR:        ^~~~~~~~~~~~~~
  fn G(addr b: Class*);
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc17: type = struct_type {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_addr_not_self.carbon" {
// CHECK:STDOUT:   class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%a: Class*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b: Class*);
