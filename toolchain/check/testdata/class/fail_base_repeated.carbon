// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class B1 {}
base class B2 {}

class C {
  base: B1;
  // CHECK:STDERR: fail_base_repeated.carbon:[[@LINE+6]]:11: ERROR: Repeated `base` declaration. Multiple inheritance is not permitted.
  // CHECK:STDERR:   base: B2;
  // CHECK:STDERR:           ^
  // CHECK:STDERR: fail_base_repeated.carbon:[[@LINE-4]]:11: Previous `base` declaration is here.
  // CHECK:STDERR:   base: B1;
  // CHECK:STDERR:           ^
  base: B2;
}

class D {
  // TODO: Consider adding a custom diagnostic for this case.
  base: B1;
  // CHECK:STDERR: fail_base_repeated.carbon:[[@LINE+6]]:11: ERROR: Repeated `base` declaration. Multiple inheritance is not permitted.
  // CHECK:STDERR:   base: B1;
  // CHECK:STDERR:           ^
  // CHECK:STDERR: fail_base_repeated.carbon:[[@LINE-4]]:11: Previous `base` declaration is here.
  // CHECK:STDERR:   base: B1;
  // CHECK:STDERR:           ^
  base: B1;
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
// CHECK:STDOUT:   %.loc11_11.1: type = unbound_element_type C, B1
// CHECK:STDOUT:   %.loc11_11.2: <unbound field of class C> = base B1, element0
// CHECK:STDOUT:   %B2.ref: type = name_ref B2, file.%B2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc11_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %B1.ref.loc23: type = name_ref B1, file.%B1
// CHECK:STDOUT:   %.loc23_11.1: type = unbound_element_type D, B1
// CHECK:STDOUT:   %.loc23_11.2: <unbound field of class D> = base B1, element0
// CHECK:STDOUT:   %B1.ref.loc30: type = name_ref B1, file.%B1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc23_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
