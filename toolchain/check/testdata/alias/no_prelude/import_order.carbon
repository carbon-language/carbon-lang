// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon

library "a" api;

class C { var v: (); }
alias a = C;
alias b = a;
alias c = b;
alias d = c;

// --- b.carbon

library "b" api;

import library "a";

// Access imports in reverse order of export.
var d_val: d = {.v = ()};
var c_val: c = {.v = d_val.v};
var b_val: b = {.v = c_val.v};
var a_val: a = {.v = b_val.v};

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type C, () [template]
// CHECK:STDOUT:   %.3: type = struct_type {.v: ()} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %a: type = bind_alias a, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %a.ref: type = name_ref a, %a [template = constants.%C]
// CHECK:STDOUT:   %b: type = bind_alias b, %a [template = constants.%C]
// CHECK:STDOUT:   %b.ref: type = name_ref b, %b [template = constants.%C]
// CHECK:STDOUT:   %c: type = bind_alias c, %b [template = constants.%C]
// CHECK:STDOUT:   %c.ref: type = name_ref c, %c [template = constants.%C]
// CHECK:STDOUT:   %d: type = bind_alias d, %c [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4_19.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc4_19.2: type = converted %.loc4_19.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc4_16: <unbound element of class C> = field_decl v, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .v = %.loc4_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.v: ()} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.v: ()} [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %struct: C = struct_value (%tuple) [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type C, () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .a = %import_ref.2
// CHECK:STDOUT:     .b = %import_ref.3
// CHECK:STDOUT:     .c = %import_ref.4
// CHECK:STDOUT:     .d = %import_ref.5
// CHECK:STDOUT:     .d_val = %d_val
// CHECK:STDOUT:     .c_val = %c_val
// CHECK:STDOUT:     .b_val = %b_val
// CHECK:STDOUT:     .a_val = %a_val
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+1, unloaded
// CHECK:STDOUT:   %import_ref.2: type = import_ref ir1, inst+12, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.3: type = import_ref ir1, inst+14, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.4: type = import_ref ir1, inst+16, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.5: type = import_ref ir1, inst+18, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.6: <unbound element of class C> = import_ref ir1, inst+7, loaded [template = imports.%.1]
// CHECK:STDOUT:   %import_ref.7 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %d.ref: type = name_ref d, %import_ref.5 [template = constants.%C]
// CHECK:STDOUT:   %d_val.var: ref C = var d_val
// CHECK:STDOUT:   %d_val: ref C = bind_name d_val, %d_val.var
// CHECK:STDOUT:   %c.ref: type = name_ref c, %import_ref.4 [template = constants.%C]
// CHECK:STDOUT:   %c_val.var: ref C = var c_val
// CHECK:STDOUT:   %c_val: ref C = bind_name c_val, %c_val.var
// CHECK:STDOUT:   %b.ref: type = name_ref b, %import_ref.3 [template = constants.%C]
// CHECK:STDOUT:   %b_val.var: ref C = var b_val
// CHECK:STDOUT:   %b_val: ref C = bind_name b_val, %b_val.var
// CHECK:STDOUT:   %a.ref: type = name_ref a, %import_ref.2 [template = constants.%C]
// CHECK:STDOUT:   %a_val.var: ref C = var a_val
// CHECK:STDOUT:   %a_val: ref C = bind_name a_val, %a_val.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .v = file.%import_ref.6
// CHECK:STDOUT:   .Self = file.%import_ref.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_23.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc7_24.1: {.v: ()} = struct_literal (%.loc7_23.1)
// CHECK:STDOUT:   %.loc7_24.2: ref () = class_element_access file.%d_val.var, element0
// CHECK:STDOUT:   %.loc7_23.2: init () = tuple_init () to %.loc7_24.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_24.3: init () = converted %.loc7_23.1, %.loc7_23.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_24.4: init C = class_init (%.loc7_24.3), file.%d_val.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc7_25: init C = converted %.loc7_24.1, %.loc7_24.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%d_val.var, %.loc7_25
// CHECK:STDOUT:   %d_val.ref: ref C = name_ref d_val, file.%d_val
// CHECK:STDOUT:   %v.ref.loc8: <unbound element of class C> = name_ref v, file.%import_ref.6 [template = imports.%.1]
// CHECK:STDOUT:   %.loc8_27.1: ref () = class_element_access %d_val.ref, element0
// CHECK:STDOUT:   %.loc8_29.1: {.v: ()} = struct_literal (%.loc8_27.1)
// CHECK:STDOUT:   %.loc8_29.2: ref () = class_element_access file.%c_val.var, element0
// CHECK:STDOUT:   %.loc8_27.2: init () = tuple_init () to %.loc8_29.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc8_29.3: init () = converted %.loc8_27.1, %.loc8_27.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc8_29.4: init C = class_init (%.loc8_29.3), file.%c_val.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc8_30: init C = converted %.loc8_29.1, %.loc8_29.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c_val.var, %.loc8_30
// CHECK:STDOUT:   %c_val.ref: ref C = name_ref c_val, file.%c_val
// CHECK:STDOUT:   %v.ref.loc9: <unbound element of class C> = name_ref v, file.%import_ref.6 [template = imports.%.1]
// CHECK:STDOUT:   %.loc9_27.1: ref () = class_element_access %c_val.ref, element0
// CHECK:STDOUT:   %.loc9_29.1: {.v: ()} = struct_literal (%.loc9_27.1)
// CHECK:STDOUT:   %.loc9_29.2: ref () = class_element_access file.%b_val.var, element0
// CHECK:STDOUT:   %.loc9_27.2: init () = tuple_init () to %.loc9_29.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc9_29.3: init () = converted %.loc9_27.1, %.loc9_27.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc9_29.4: init C = class_init (%.loc9_29.3), file.%b_val.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_30: init C = converted %.loc9_29.1, %.loc9_29.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%b_val.var, %.loc9_30
// CHECK:STDOUT:   %b_val.ref: ref C = name_ref b_val, file.%b_val
// CHECK:STDOUT:   %v.ref.loc10: <unbound element of class C> = name_ref v, file.%import_ref.6 [template = imports.%.1]
// CHECK:STDOUT:   %.loc10_27.1: ref () = class_element_access %b_val.ref, element0
// CHECK:STDOUT:   %.loc10_29.1: {.v: ()} = struct_literal (%.loc10_27.1)
// CHECK:STDOUT:   %.loc10_29.2: ref () = class_element_access file.%a_val.var, element0
// CHECK:STDOUT:   %.loc10_27.2: init () = tuple_init () to %.loc10_29.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc10_29.3: init () = converted %.loc10_27.1, %.loc10_27.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc10_29.4: init C = class_init (%.loc10_29.3), file.%a_val.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc10_30: init C = converted %.loc10_29.1, %.loc10_29.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%a_val.var, %.loc10_30
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
