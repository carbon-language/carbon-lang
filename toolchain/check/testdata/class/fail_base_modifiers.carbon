// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class B {}

class C1 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+3]]:3: ERROR: `private` not allowed on `base` declaration.
  // CHECK:STDERR:   private extend base: B;
  // CHECK:STDERR:   ^~~~~~~
  private extend base: B;
}

class C2 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+6]]:3: ERROR: `abstract` not allowed on `base` declaration.
  // CHECK:STDERR:   abstract base: B;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+3]]:3: ERROR: Missing `extend` before `base` declaration in class.
  // CHECK:STDERR:   abstract base: B;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  abstract base: B;
}

class C3 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+6]]:10: ERROR: `default` not allowed on declaration with `extend`.
  // CHECK:STDERR:   extend default base: B;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+3]]:3: `extend` previously appeared here.
  // CHECK:STDERR:   extend default base: B;
  // CHECK:STDERR:   ^~~~~~
  extend default base: B;
}

class C4 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+6]]:10: ERROR: `extend` repeated on declaration.
  // CHECK:STDERR:   extend extend base: B;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+3]]:3: `extend` previously appeared here.
  // CHECK:STDERR:   extend extend base: B;
  // CHECK:STDERR:   ^~~~~~
  extend extend base: B;
}

// CHECK:STDOUT: --- fail_base_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %C1: type = class_type @C1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type C1, B [template]
// CHECK:STDOUT:   %.5: type = struct_type {.base: B} [template]
// CHECK:STDOUT:   %C2: type = class_type @C2 [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type C2, B [template]
// CHECK:STDOUT:   %C3: type = class_type @C3 [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type C3, B [template]
// CHECK:STDOUT:   %C4: type = class_type @C4 [template]
// CHECK:STDOUT:   %.8: type = unbound_element_type C4, B [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.B = %B.decl, .C1 = %C1.decl, .C2 = %C2.decl, .C3 = %C3.decl, .C4 = %C4.decl} [template]
// CHECK:STDOUT:   %B.decl = class_decl @B, ()
// CHECK:STDOUT:   %C1.decl = class_decl @C1, ()
// CHECK:STDOUT:   %C2.decl = class_decl @C2, ()
// CHECK:STDOUT:   %C3.decl = class_decl @C3, ()
// CHECK:STDOUT:   %C4.decl = class_decl @C4, ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1 {
// CHECK:STDOUT:   %B.ref: type = name_ref B, constants.%B [template = constants.%B]
// CHECK:STDOUT:   %.loc13: <unbound element of class C1> = base_decl B, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2 {
// CHECK:STDOUT:   %B.ref: type = name_ref B, constants.%B [template = constants.%B]
// CHECK:STDOUT:   %.loc23: <unbound element of class C2> = base_decl B, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C3 {
// CHECK:STDOUT:   %B.ref: type = name_ref B, constants.%B [template = constants.%B]
// CHECK:STDOUT:   %.loc33: <unbound element of class C3> = base_decl B, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc33
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C4 {
// CHECK:STDOUT:   %B.ref: type = name_ref B, constants.%B [template = constants.%B]
// CHECK:STDOUT:   %.loc43: <unbound element of class C4> = base_decl B, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc43
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
