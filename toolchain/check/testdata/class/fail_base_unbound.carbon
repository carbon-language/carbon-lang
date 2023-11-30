// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class B {}

class C {
  base: B;
}

// CHECK:STDERR: fail_base_unbound.carbon:[[@LINE+3]]:13: ERROR: Expression cannot be used as a value.
// CHECK:STDERR: let b: B = C.base;
// CHECK:STDERR:             ^
let b: B = C.base;

// CHECK:STDOUT: --- fail_base_unbound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc7_15.1: type = struct_type {}
// CHECK:STDOUT:   %.loc7_15.2: type = tuple_type ()
// CHECK:STDOUT:   %.loc7_1: type = ptr_type {}
// CHECK:STDOUT:   %.loc11: type = struct_type {.base: B}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %B.decl = class_decl @B, ()
// CHECK:STDOUT:   %B: type = class_type @B
// CHECK:STDOUT:   %C.decl = class_decl @C, ()
// CHECK:STDOUT:   %C: type = class_type @C
// CHECK:STDOUT:   %B.ref: type = name_ref B, %B
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C
// CHECK:STDOUT:   %base.ref: <unbound field of class C> = name_ref base, @C.%.loc10_10.2
// CHECK:STDOUT:   %b: B = bind_name b, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B
// CHECK:STDOUT:   %.loc10_10.1: type = unbound_element_type C, B
// CHECK:STDOUT:   %.loc10_10.2: <unbound field of class C> = base B, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc10_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
