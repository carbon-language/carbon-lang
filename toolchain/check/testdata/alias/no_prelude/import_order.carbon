// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/no_prelude/import_order.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/no_prelude/import_order.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

class C { var v: (); }
alias a = C;
alias b = a;
alias c = b;
alias d = c;

// --- b.carbon

library "[[@TEST_NAME]]";

import library "a";

// Access imports in reverse order of export.
var d_val: d = {.v = ()};
var c_val: c = {.v = d_val.v};
var b_val: b = {.v = c_val.v};
var a_val: a = {.v = b_val.v};

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.v [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %a: type = bind_alias a, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %a.ref: type = name_ref a, %a [concrete = constants.%C]
// CHECK:STDOUT:   %b: type = bind_alias b, %a [concrete = constants.%C]
// CHECK:STDOUT:   %b.ref: type = name_ref b, %b [concrete = constants.%C]
// CHECK:STDOUT:   %c: type = bind_alias c, %b [concrete = constants.%C]
// CHECK:STDOUT:   %c.ref: type = name_ref c, %c [concrete = constants.%C]
// CHECK:STDOUT:   %d: type = bind_alias d, %c [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4_16: %C.elem = field_decl v, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc4_11: %C.elem = var_pattern %.loc4_16
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %C.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.v [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .v = %.loc4_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.v [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %empty_tuple.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C = import_ref Main//a, C, unloaded
// CHECK:STDOUT:   %Main.a: type = import_ref Main//a, a, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.b: type = import_ref Main//a, b, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.c: type = import_ref Main//a, c, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.d: type = import_ref Main//a, d, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.146: <witness> = import_ref Main//a, loc4_22, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//a, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.f99: %C.elem = import_ref Main//a, loc4_16, loaded [concrete = %.2fc]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .a = imports.%Main.a
// CHECK:STDOUT:     .b = imports.%Main.b
// CHECK:STDOUT:     .c = imports.%Main.c
// CHECK:STDOUT:     .d = imports.%Main.d
// CHECK:STDOUT:     .d_val = %d_val
// CHECK:STDOUT:     .c_val = %c_val
// CHECK:STDOUT:     .b_val = %b_val
// CHECK:STDOUT:     .a_val = %a_val
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d_val.patt: %C = binding_pattern d_val
// CHECK:STDOUT:     %.loc7: %C = var_pattern %d_val.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d_val.var: ref %C = var d_val
// CHECK:STDOUT:   %d.ref: type = name_ref d, imports.%Main.d [concrete = constants.%C]
// CHECK:STDOUT:   %d_val: ref %C = bind_name d_val, %d_val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c_val.patt: %C = binding_pattern c_val
// CHECK:STDOUT:     %.loc8: %C = var_pattern %c_val.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_val.var: ref %C = var c_val
// CHECK:STDOUT:   %c.ref: type = name_ref c, imports.%Main.c [concrete = constants.%C]
// CHECK:STDOUT:   %c_val: ref %C = bind_name c_val, %c_val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_val.patt: %C = binding_pattern b_val
// CHECK:STDOUT:     %.loc9: %C = var_pattern %b_val.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_val.var: ref %C = var b_val
// CHECK:STDOUT:   %b.ref: type = name_ref b, imports.%Main.b [concrete = constants.%C]
// CHECK:STDOUT:   %b_val: ref %C = bind_name b_val, %b_val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_val.patt: %C = binding_pattern a_val
// CHECK:STDOUT:     %.loc10: %C = var_pattern %a_val.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_val.var: ref %C = var a_val
// CHECK:STDOUT:   %a.ref: type = name_ref a, imports.%Main.a [concrete = constants.%C]
// CHECK:STDOUT:   %a_val: ref %C = bind_name a_val, %a_val.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.146
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT:   .v = imports.%Main.import_ref.f99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_23.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc7_24.1: %struct_type.v = struct_literal (%.loc7_23.1)
// CHECK:STDOUT:   %.loc7_24.2: ref %empty_tuple.type = class_element_access file.%d_val.var, element0
// CHECK:STDOUT:   %.loc7_23.2: init %empty_tuple.type = tuple_init () to %.loc7_24.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_24.3: init %empty_tuple.type = converted %.loc7_23.1, %.loc7_23.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_24.4: init %C = class_init (%.loc7_24.3), file.%d_val.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc7_1: init %C = converted %.loc7_24.1, %.loc7_24.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%d_val.var, %.loc7_1
// CHECK:STDOUT:   %d_val.ref: ref %C = name_ref d_val, file.%d_val
// CHECK:STDOUT:   %v.ref.loc8: %C.elem = name_ref v, imports.%Main.import_ref.f99 [concrete = imports.%.2fc]
// CHECK:STDOUT:   %.loc8_27.1: ref %empty_tuple.type = class_element_access %d_val.ref, element0
// CHECK:STDOUT:   %.loc8_29.1: %struct_type.v = struct_literal (%.loc8_27.1)
// CHECK:STDOUT:   %.loc8_29.2: ref %empty_tuple.type = class_element_access file.%c_val.var, element0
// CHECK:STDOUT:   %.loc8_27.2: init %empty_tuple.type = tuple_init () to %.loc8_29.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_29.3: init %empty_tuple.type = converted %.loc8_27.1, %.loc8_27.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_29.4: init %C = class_init (%.loc8_29.3), file.%c_val.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_1: init %C = converted %.loc8_29.1, %.loc8_29.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c_val.var, %.loc8_1
// CHECK:STDOUT:   %c_val.ref: ref %C = name_ref c_val, file.%c_val
// CHECK:STDOUT:   %v.ref.loc9: %C.elem = name_ref v, imports.%Main.import_ref.f99 [concrete = imports.%.2fc]
// CHECK:STDOUT:   %.loc9_27.1: ref %empty_tuple.type = class_element_access %c_val.ref, element0
// CHECK:STDOUT:   %.loc9_29.1: %struct_type.v = struct_literal (%.loc9_27.1)
// CHECK:STDOUT:   %.loc9_29.2: ref %empty_tuple.type = class_element_access file.%b_val.var, element0
// CHECK:STDOUT:   %.loc9_27.2: init %empty_tuple.type = tuple_init () to %.loc9_29.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_29.3: init %empty_tuple.type = converted %.loc9_27.1, %.loc9_27.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_29.4: init %C = class_init (%.loc9_29.3), file.%b_val.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_1: init %C = converted %.loc9_29.1, %.loc9_29.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%b_val.var, %.loc9_1
// CHECK:STDOUT:   %b_val.ref: ref %C = name_ref b_val, file.%b_val
// CHECK:STDOUT:   %v.ref.loc10: %C.elem = name_ref v, imports.%Main.import_ref.f99 [concrete = imports.%.2fc]
// CHECK:STDOUT:   %.loc10_27.1: ref %empty_tuple.type = class_element_access %b_val.ref, element0
// CHECK:STDOUT:   %.loc10_29.1: %struct_type.v = struct_literal (%.loc10_27.1)
// CHECK:STDOUT:   %.loc10_29.2: ref %empty_tuple.type = class_element_access file.%a_val.var, element0
// CHECK:STDOUT:   %.loc10_27.2: init %empty_tuple.type = tuple_init () to %.loc10_29.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_29.3: init %empty_tuple.type = converted %.loc10_27.1, %.loc10_27.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_29.4: init %C = class_init (%.loc10_29.3), file.%a_val.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc10_1: init %C = converted %.loc10_29.1, %.loc10_29.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%a_val.var, %.loc10_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
