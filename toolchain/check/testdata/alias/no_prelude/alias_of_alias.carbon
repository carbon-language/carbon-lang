// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C {}
alias a = C;
alias b = a;
alias c = b;
let d: c = {};

// CHECK:STDOUT: --- alias_of_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %struct: C = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %a: type = bind_alias a, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %a.ref: type = name_ref a, %a [template = constants.%C]
// CHECK:STDOUT:   %b: type = bind_alias b, %a [template = constants.%C]
// CHECK:STDOUT:   %b.ref: type = name_ref b, %b [template = constants.%C]
// CHECK:STDOUT:   %c: type = bind_alias c, %b [template = constants.%C]
// CHECK:STDOUT:   %c.ref: type = name_ref c, %c [template = constants.%C]
// CHECK:STDOUT:   %.loc11_13.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc11_13.2: ref C = temporary_storage
// CHECK:STDOUT:   %.loc11_13.3: init C = class_init (), %.loc11_13.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc11_13.4: ref C = temporary %.loc11_13.2, %.loc11_13.3
// CHECK:STDOUT:   %.loc11_14.1: ref C = converted %.loc11_13.1, %.loc11_13.4
// CHECK:STDOUT:   %.loc11_14.2: C = bind_value %.loc11_14.1
// CHECK:STDOUT:   %d: C = bind_name d, %.loc11_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
