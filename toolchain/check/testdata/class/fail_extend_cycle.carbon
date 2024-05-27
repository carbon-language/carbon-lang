// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class A {
}

base class B {
  // This ensures that the compiler treats A as complete.
  extend base: A;
}

// CHECK:STDERR: fail_extend_cycle.carbon:[[@LINE+7]]:1: ERROR: Redefinition of `class A`.
// CHECK:STDERR: base class A {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_extend_cycle.carbon:[[@LINE-11]]:1: Previously defined here.
// CHECK:STDERR: base class A {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
base class A {
  extend base: A;
  // CHECK:STDERR: fail_extend_cycle.carbon:[[@LINE+3]]:10: ERROR: Name `C` not found.
  // CHECK:STDERR:   var c: C;
  // CHECK:STDERR:          ^
  var c: C;
}

// CHECK:STDOUT: --- fail_extend_cycle.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type B, A [template]
// CHECK:STDOUT:   %.5: type = struct_type {.base: A} [template]
// CHECK:STDOUT:   %.6: type = class_type @.1 [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type <invalid>, A [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %.decl: type = class_decl @.1 [template = constants.%.6] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc12: <unbound element of class B> = base_decl A, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .base = %.loc12
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc23: <unbound element of class <invalid>> = base_decl A, element0 [template]
// CHECK:STDOUT:   %C.ref: <error> = name_ref C, <error> [template = <error>]
// CHECK:STDOUT:   %.loc27: <error> = field_decl c, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.6
// CHECK:STDOUT:   .base = %.loc23
// CHECK:STDOUT:   .c = %.loc27
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
