// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class B {}

class C {

  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+3]]:3: ERROR: `private` not allowed on `base` declaration.
  // CHECK:STDERR:   private base: B;
  // CHECK:STDERR:   ^~~~~~~
  private base: B;

  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+9]]:3: ERROR: `abstract` not allowed on `base` declaration.
  // CHECK:STDERR:   abstract base: B;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+6]]:19: ERROR: Multiple `base` declarations in class. Multiple inheritance is not permitted.
  // CHECK:STDERR:   abstract base: B;
  // CHECK:STDERR:                   ^
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE-8]]:18: Previous `base` declaration is here.
  // CHECK:STDERR:   private base: B;
  // CHECK:STDERR:                  ^
  abstract base: B;
}

// CHECK:STDOUT: --- fail_base_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc7_15.1: type = struct_type {}
// CHECK:STDOUT:   %.loc7_15.2: type = tuple_type ()
// CHECK:STDOUT:   %.loc7_1: type = ptr_type {}
// CHECK:STDOUT:   %.loc26: type = struct_type {.base: B}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.B = %B.decl, .C = %C.decl}
// CHECK:STDOUT:   %B.decl = class_decl @B, ()
// CHECK:STDOUT:   %B: type = class_type @B
// CHECK:STDOUT:   %C.decl = class_decl @C, ()
// CHECK:STDOUT:   %C: type = class_type @C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref.loc14: type = name_ref B, file.%B
// CHECK:STDOUT:   %.loc14_18.1: type = unbound_element_type C, B
// CHECK:STDOUT:   %.loc14_18.2: <unbound element of class C> = base_decl B, element0
// CHECK:STDOUT:   %B.ref.loc25: type = name_ref B, file.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc14_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
