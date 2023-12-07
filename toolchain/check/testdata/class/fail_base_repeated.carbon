// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class B1 {}
base class B2 {}

class C {
  extend base: B1;
  // CHECK:STDERR: fail_base_repeated.carbon:[[@LINE+6]]:18: ERROR: Multiple `base` declarations in class. Multiple inheritance is not permitted.
  // CHECK:STDERR:   extend base: B2;
  // CHECK:STDERR:                  ^
  // CHECK:STDERR: fail_base_repeated.carbon:[[@LINE-4]]:18: Previous `base` declaration is here.
  // CHECK:STDERR:   extend base: B1;
  // CHECK:STDERR:                  ^
  extend base: B2;
}

class D {
  // TODO: Consider adding a custom diagnostic for this case.
  extend base: B1;
  // CHECK:STDERR: fail_base_repeated.carbon:[[@LINE+6]]:18: ERROR: Multiple `base` declarations in class. Multiple inheritance is not permitted.
  // CHECK:STDERR:   extend base: B1;
  // CHECK:STDERR:                  ^
  // CHECK:STDERR: fail_base_repeated.carbon:[[@LINE-4]]:18: Previous `base` declaration is here.
  // CHECK:STDERR:   extend base: B1;
  // CHECK:STDERR:                  ^
  extend base: B1;
}

// CHECK:STDOUT: --- fail_base_repeated.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc7_16.1: type = struct_type {}
// CHECK:STDOUT:   %.loc7_16.2: type = tuple_type ()
// CHECK:STDOUT:   %.loc7_1: type = ptr_type {}
// CHECK:STDOUT:   %.loc19: type = struct_type {.base: B1}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.B1 = %B1.decl, .B2 = %B2.decl, .C = %C.decl, .D = %D.decl}
// CHECK:STDOUT:   %B1.decl = class_decl @B1, ()
// CHECK:STDOUT:   %B1: type = class_type @B1
// CHECK:STDOUT:   %B2.decl = class_decl @B2, ()
// CHECK:STDOUT:   %B2: type = class_type @B2
// CHECK:STDOUT:   %C.decl = class_decl @C, ()
// CHECK:STDOUT:   %C: type = class_type @C
// CHECK:STDOUT:   %D.decl = class_decl @D, ()
// CHECK:STDOUT:   %D: type = class_type @D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B1 {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B2 {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B1.ref: type = name_ref B1, file.%B1
// CHECK:STDOUT:   %.loc11_18.1: type = unbound_element_type C, B1
// CHECK:STDOUT:   %.loc11_18.2: <unbound element of class C> = base_decl B1, element0
// CHECK:STDOUT:   %B2.ref: type = name_ref B2, file.%B2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc11_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %B1.ref.loc23: type = name_ref B1, file.%B1
// CHECK:STDOUT:   %.loc23_18.1: type = unbound_element_type D, B1
// CHECK:STDOUT:   %.loc23_18.2: <unbound element of class D> = base_decl B1, element0
// CHECK:STDOUT:   %B1.ref.loc30: type = name_ref B1, file.%B1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc23_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
