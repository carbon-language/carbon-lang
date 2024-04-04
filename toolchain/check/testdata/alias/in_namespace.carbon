// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C { var v: i32; }

namespace NS;
alias NS.a = C;

let b: NS.a = {.v = 0};

fn F() -> NS.a {
  return {.v = 0};
}

// CHECK:STDOUT: --- in_namespace.carbon
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
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %a: type = bind_alias a, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %NS.ref.loc12: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:   %a.ref.loc12: type = name_ref a, %a [template = constants.%C]
// CHECK:STDOUT:   %.loc12_21: i32 = int_literal 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_22.1: {.v: i32} = struct_literal (%.loc12_21)
// CHECK:STDOUT:   %.loc12_22.2: ref C = temporary_storage
// CHECK:STDOUT:   %.loc12_22.3: ref i32 = class_element_access %.loc12_22.2, element0
// CHECK:STDOUT:   %.loc12_22.4: init i32 = initialize_from %.loc12_21 to %.loc12_22.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_22.5: init C = class_init (%.loc12_22.4), %.loc12_22.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_22.6: ref C = temporary %.loc12_22.2, %.loc12_22.5
// CHECK:STDOUT:   %.loc12_22.7: ref C = converted %.loc12_22.1, %.loc12_22.6
// CHECK:STDOUT:   %.loc12_22.8: C = bind_value %.loc12_22.7
// CHECK:STDOUT:   %b: C = bind_name b, %.loc12_22.8
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %NS.ref.loc14: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:     %a.ref.loc14: type = name_ref a, %a [template = constants.%C]
// CHECK:STDOUT:     @F.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: fn @F() -> %return: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_16: i32 = int_literal 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc15_17.1: {.v: i32} = struct_literal (%.loc15_16)
// CHECK:STDOUT:   %.loc15_17.2: ref i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc15_17.3: init i32 = initialize_from %.loc15_16 to %.loc15_17.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc15_17.4: init C = class_init (%.loc15_17.3), %return [template = constants.%.5]
// CHECK:STDOUT:   %.loc15_17.5: init C = converted %.loc15_17.1, %.loc15_17.4 [template = constants.%.5]
// CHECK:STDOUT:   return %.loc15_17.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
