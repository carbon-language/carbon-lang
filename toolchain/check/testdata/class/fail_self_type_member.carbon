// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  var b: bool;
}

fn F() -> bool {
  var c1: Class = {.b = true};
  // CHECK:STDERR: fail_self_type_member.carbon:[[@LINE+7]]:17: ERROR: Expected identifier after `.`.
  // CHECK:STDERR:   var c2: Class.Self = c1;
  // CHECK:STDERR:                 ^~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_self_type_member.carbon:[[@LINE+3]]:17: ERROR: Semantics TODO: `Error recovery from keyword name.`.
  // CHECK:STDERR:   var c2: Class.Self = c1;
  // CHECK:STDERR:                 ^~~~
  var c2: Class.Self = c1;
  return c2.b;
}

// CHECK:STDOUT: --- fail_self_type_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type Class, bool [template]
// CHECK:STDOUT:   %.2: type = struct_type {.b: bool} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.b: bool} [template]
// CHECK:STDOUT:   %.4: bool = bool_literal true [template]
// CHECK:STDOUT:   %.5: Class = struct_value (%.4) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc8: <unbound element of class Class> = field_decl b, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .b = %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT: }
// CHECK:STDOUT:
