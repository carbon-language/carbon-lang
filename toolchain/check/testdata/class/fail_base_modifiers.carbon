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
// CHECK:STDOUT:   %.loc7_15.1: type = struct_type {}
// CHECK:STDOUT:   %.loc7_15.2: type = tuple_type ()
// CHECK:STDOUT:   %.loc7_14: type = ptr_type {}
// CHECK:STDOUT:   %.loc14: type = struct_type {.base: B}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.B = %B.decl, .C1 = %C1.decl, .C2 = %C2.decl, .C3 = %C3.decl, .C4 = %C4.decl}
// CHECK:STDOUT:   %B.decl = class_decl @B, ()
// CHECK:STDOUT:   %B: type = class_type @B
// CHECK:STDOUT:   %C1.decl = class_decl @C1, ()
// CHECK:STDOUT:   %C1: type = class_type @C1
// CHECK:STDOUT:   %C2.decl = class_decl @C2, ()
// CHECK:STDOUT:   %C2: type = class_type @C2
// CHECK:STDOUT:   %C3.decl = class_decl @C3, ()
// CHECK:STDOUT:   %C3: type = class_type @C3
// CHECK:STDOUT:   %C4.decl = class_decl @C4, ()
// CHECK:STDOUT:   %C4: type = class_type @C4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1 {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B
// CHECK:STDOUT:   %.loc13_25.1: type = unbound_element_type C1, B
// CHECK:STDOUT:   %.loc13_25.2: <unbound element of class C1> = base_decl B, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc13_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2 {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B
// CHECK:STDOUT:   %.loc23_19.1: type = unbound_element_type C2, B
// CHECK:STDOUT:   %.loc23_19.2: <unbound element of class C2> = base_decl B, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc23_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C3 {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B
// CHECK:STDOUT:   %.loc33_25.1: type = unbound_element_type C3, B
// CHECK:STDOUT:   %.loc33_25.2: <unbound element of class C3> = base_decl B, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc33_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C4 {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B
// CHECK:STDOUT:   %.loc43_24.1: type = unbound_element_type C4, B
// CHECK:STDOUT:   %.loc43_24.2: <unbound element of class C4> = base_decl B, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc43_24.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
