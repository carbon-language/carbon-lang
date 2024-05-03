// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C {
  fn F(c: C) -> i32 { return c.a; }

  var a: i32;
}

// CHECK:STDOUT: --- complete_in_member_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: F = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type C, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.a: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %c.loc8_8.1: C = param c
// CHECK:STDOUT:     %c.loc8_8.2: C = bind_name c, %c.loc8_8.1
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: <unbound element of class C> = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .a = %.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(@C.%c.loc8_8.2: C) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: C = name_ref c, @C.%c.loc8_8.2
// CHECK:STDOUT:   %a.ref: <unbound element of class C> = name_ref a, @C.%.loc10 [template = @C.%.loc10]
// CHECK:STDOUT:   %.loc8_31.1: ref i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc8_31.2: i32 = bind_value %.loc8_31.1
// CHECK:STDOUT:   return %.loc8_31.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
