// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C { var v: (); }

namespace NS;
alias NS.a = C;

let b: NS.a = {.v = ()};

fn F() -> NS.a {
  return {.v = ()};
}

// CHECK:STDOUT: --- in_namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type C, () [template]
// CHECK:STDOUT:   %.3: type = struct_type {.v: ()} [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.v: ()} [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %struct.1: C = struct_value (%tuple) [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %a: type = bind_alias a, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %NS.ref.loc12: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:   %a.ref.loc12: type = name_ref a, %a [template = constants.%C]
// CHECK:STDOUT:   %.loc12_22.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc12_23.1: {.v: ()} = struct_literal (%.loc12_22.1)
// CHECK:STDOUT:   %.loc12_23.2: ref C = temporary_storage
// CHECK:STDOUT:   %.loc12_23.3: ref () = class_element_access %.loc12_23.2, element0
// CHECK:STDOUT:   %.loc12_22.2: init () = tuple_init () to %.loc12_23.3 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc12_23.4: init () = converted %.loc12_22.1, %.loc12_22.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc12_23.5: init C = class_init (%.loc12_23.4), %.loc12_23.2 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc12_23.6: ref C = temporary %.loc12_23.2, %.loc12_23.5
// CHECK:STDOUT:   %.loc12_24.1: ref C = converted %.loc12_23.1, %.loc12_23.6
// CHECK:STDOUT:   %.loc12_24.2: C = bind_value %.loc12_24.1
// CHECK:STDOUT:   %b: C = bind_name b, %.loc12_24.2
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %NS.ref.loc14: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:     %a.ref.loc14: type = name_ref a, %a [template = constants.%C]
// CHECK:STDOUT:     @F.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc7_19.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc7_19.2: type = converted %.loc7_19.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_16: <unbound element of class C> = field_decl v, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .v = %.loc7_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_17.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc15_18.1: {.v: ()} = struct_literal (%.loc15_17.1)
// CHECK:STDOUT:   %.loc15_18.2: ref () = class_element_access %return, element0
// CHECK:STDOUT:   %.loc15_17.2: init () = tuple_init () to %.loc15_18.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc15_18.3: init () = converted %.loc15_17.1, %.loc15_17.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc15_18.4: init C = class_init (%.loc15_18.3), %return [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc15_19: init C = converted %.loc15_18.1, %.loc15_18.4 [template = constants.%struct.1]
// CHECK:STDOUT:   return %.loc15_19 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
