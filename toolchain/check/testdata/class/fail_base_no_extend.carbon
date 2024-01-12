// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class B {}

class C {
  // CHECK:STDERR: fail_base_no_extend.carbon:[[@LINE+3]]:3: ERROR: Missing `extend` before `base` declaration in class.
  // CHECK:STDERR:   base: B;
  // CHECK:STDERR:   ^~~~~~~~
  base: B;
}

// CHECK:STDOUT: --- fail_base_no_extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc7_15.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.loc7_15.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.loc7_14: type = ptr_type {} [template]
// CHECK:STDOUT:   %.loc14: type = struct_type {.base: B} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.B = %B.decl, .C = %C.decl}
// CHECK:STDOUT:   %B.decl = class_decl @B, ()
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %C.decl = class_decl @C, ()
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B [template = file.%B]
// CHECK:STDOUT:   %.loc13_10.1: type = unbound_element_type C, B [template]
// CHECK:STDOUT:   %.loc13_10.2: <unbound element of class C> = base_decl B, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc13_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
