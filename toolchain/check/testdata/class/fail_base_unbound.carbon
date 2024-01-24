// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class B {}

class C {
  extend base: B;
}

// CHECK:STDERR: fail_base_unbound.carbon:[[@LINE+3]]:12: ERROR: Expression cannot be used as a value.
// CHECK:STDERR: let b: B = C.base;
// CHECK:STDERR:            ^~~~~~
let b: B = C.base;

// CHECK:STDOUT: --- fail_base_unbound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type C, B [template]
// CHECK:STDOUT:   %.5: type = struct_type {.base: B} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.B = %B.decl, .C = %C.decl} [template]
// CHECK:STDOUT:   %B.decl = class_decl @B, ()
// CHECK:STDOUT:   %C.decl = class_decl @C, ()
// CHECK:STDOUT:   %B.ref: type = name_ref B, constants.%B [template = constants.%B]
// CHECK:STDOUT:   %C.ref: type = name_ref C, constants.%C [template = constants.%C]
// CHECK:STDOUT:   %base.ref: <unbound element of class C> = name_ref base, @C.%.loc10 [template = @C.%.loc10]
// CHECK:STDOUT:   %b: B = bind_name b, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, constants.%B [template = constants.%B]
// CHECK:STDOUT:   %.loc10: <unbound element of class C> = base_decl B, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc10
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
