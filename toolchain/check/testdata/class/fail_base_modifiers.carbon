// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class B {}

class C1 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+3]]:3: ERROR: `private` not allowed on `base` declaration.
  // CHECK:STDERR:   private base: B;
  // CHECK:STDERR:   ^~~~~~~
  private base: B;
}

class C2 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+3]]:3: ERROR: `abstract` not allowed on `base` declaration.
  // CHECK:STDERR:   abstract base: B;
  // CHECK:STDERR:   ^~~~~~~~
  abstract base: B;
}

// CHECK:STDOUT: --- fail_base_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc7_15.1: type = struct_type {}
// CHECK:STDOUT:   %.loc7_15.2: type = tuple_type ()
// CHECK:STDOUT:   %.loc7_1: type = ptr_type {}
// CHECK:STDOUT:   %.loc15: type = struct_type {.base: B}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.B = %B.decl, .B2 = %B2.decl, .C1 = %C1.decl, .C2 = %C2.decl}
// CHECK:STDOUT:   %B.decl = class_decl @B, ()
// CHECK:STDOUT:   %B: type = class_type @B
// CHECK:STDOUT:   %B2.decl = class_decl @B2, ()
// CHECK:STDOUT:   %B2: type = class_type @B2
// CHECK:STDOUT:   %C1.decl = class_decl @C1, ()
// CHECK:STDOUT:   %C1: type = class_type @C1
// CHECK:STDOUT:   %C2.decl = class_decl @C2, ()
// CHECK:STDOUT:   %C2: type = class_type @C2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B2 {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1 {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B
// CHECK:STDOUT:   %.loc14_18.1: type = unbound_element_type C1, B
// CHECK:STDOUT:   %.loc14_18.2: <unbound element of class C1> = base_decl B, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc14_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2 {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B
// CHECK:STDOUT:   %.loc21_19.1: type = unbound_element_type C2, B
// CHECK:STDOUT:   %.loc21_19.2: <unbound element of class C2> = base_decl B, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc21_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
