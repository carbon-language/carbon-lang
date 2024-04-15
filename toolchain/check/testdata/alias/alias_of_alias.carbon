// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C { var v: i32; }
alias a = C;
alias b = a;
alias c = b;
let d: c = {.v = 0};

// CHECK:STDOUT: --- alias_of_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type C, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.v: i32} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.v: i32} [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.5: C = struct_value (%.4) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %a: type = bind_alias a, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %a.ref: type = name_ref a, %a [template = constants.%C]
// CHECK:STDOUT:   %b: type = bind_alias b, %a [template = constants.%C]
// CHECK:STDOUT:   %b.ref: type = name_ref b, %b [template = constants.%C]
// CHECK:STDOUT:   %c: type = bind_alias c, %b [template = constants.%C]
// CHECK:STDOUT:   %c.ref: type = name_ref c, %c [template = constants.%C]
// CHECK:STDOUT:   %.loc11_18: i32 = int_literal 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_19.1: {.v: i32} = struct_literal (%.loc11_18)
// CHECK:STDOUT:   %.loc11_19.2: ref C = temporary_storage
// CHECK:STDOUT:   %.loc11_19.3: ref i32 = class_element_access %.loc11_19.2, element0
// CHECK:STDOUT:   %.loc11_19.4: init i32 = initialize_from %.loc11_18 to %.loc11_19.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_19.5: init C = class_init (%.loc11_19.4), %.loc11_19.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_19.6: ref C = temporary %.loc11_19.2, %.loc11_19.5
// CHECK:STDOUT:   %.loc11_20.1: ref C = converted %.loc11_19.1, %.loc11_19.6
// CHECK:STDOUT:   %.loc11_20.2: C = bind_value %.loc11_20.1
// CHECK:STDOUT:   %d: C = bind_name d, %.loc11_20.2
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
