// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C { var v: i32; }
alias b = C;
// CHECK:STDERR: fail_aliased_name_in_diag.carbon:[[@LINE+3]]:1: ERROR: Cannot implicitly convert from `i32` to `C`.
// CHECK:STDERR: let c: b = 2;
// CHECK:STDERR: ^~~~~~~~~~~~~
let c: b = 2;

// CHECK:STDOUT: --- fail_aliased_name_in_diag.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type C, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.v: i32} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.v: i32} [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %b: type = bind_alias b, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %b.ref: type = name_ref b, %b [template = constants.%C]
// CHECK:STDOUT:   %.loc12: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   %c: C = bind_name c, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc7: <unbound element of class C> = field_decl v, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .v = %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
