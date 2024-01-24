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
// CHECK:STDOUT:   package: <namespace> = namespace {.Class = %Class.decl} [template]
// CHECK:STDOUT:   %Class.decl = class_decl @Class, ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%a: Class*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b: Class*);
// CHECK:STDOUT:
