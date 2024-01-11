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
// CHECK:STDOUT:   %.loc7_15.1: type = struct_type {}, const
// CHECK:STDOUT:   %.loc7_15.2: type = tuple_type (), const
// CHECK:STDOUT:   %.loc7_14: type = ptr_type {}, const
// CHECK:STDOUT:   %.loc11: type = struct_type {.base: B}, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.B = %B.decl, .C = %C.decl}
// CHECK:STDOUT:   %B.decl = class_decl @B, ()
// CHECK:STDOUT:   %B: type = class_type @B, const
// CHECK:STDOUT:   %C.decl = class_decl @C, ()
// CHECK:STDOUT:   %C: type = class_type @C, const
// CHECK:STDOUT:   %B.ref: type = name_ref B, %B, const = %B
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C, const = %C
// CHECK:STDOUT:   %base.ref: <unbound element of class C> = name_ref base, @C.%.loc10_17.2, const = @C.%.loc10_17.2
// CHECK:STDOUT:   %b: B = bind_name b, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B, const = file.%B
// CHECK:STDOUT:   %.loc10_17.1: type = unbound_element_type C, B, const
// CHECK:STDOUT:   %.loc10_17.2: <unbound element of class C> = base_decl B, element0, const
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc10_17.2
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
