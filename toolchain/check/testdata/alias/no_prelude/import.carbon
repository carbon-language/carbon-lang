// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- class1.carbon

library "class1" api;

class C {}

alias c_alias = C;

var a: C*;

// --- class2.carbon

library "class2" api;

import library "class1";

alias c_alias_alias = c_alias;

var b: c_alias*;

// --- class3.carbon

library "class3" api;

import library "class2";

var c: c_alias_alias*;

// --- var1.carbon

library "var1" api;

var a: () = ();

alias a_alias = a;

// --- var2.carbon

library "var2" api;

import library "var1";

alias a_alias_alias = a_alias;

var b: () = a_alias;

// --- fail_var3.carbon

library "var3" api;

// CHECK:STDERR: fail_var3.carbon:[[@LINE+6]]:1: In import.
// CHECK:STDERR: import library "var2";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: var2.carbon:8:5: ERROR: Semantics TODO: `Non-constant ImportRefLoaded (comes up with var)`.
// CHECK:STDERR: var b: () = a_alias;
// CHECK:STDERR:     ^
import library "var2";

var c: () = a_alias_alias;

// CHECK:STDOUT: --- class1.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = ptr_type C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .c_alias = %c_alias
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %C.ref.loc6: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %c_alias: type = bind_alias c_alias, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc8: type = ptr_type C [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref C* = var a
// CHECK:STDOUT:   %a: ref C* = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- class2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = ptr_type C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .c_alias = %import_ref.2
// CHECK:STDOUT:     .a = %import_ref.3
// CHECK:STDOUT:     .c_alias_alias = %c_alias_alias
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+1, unloaded
// CHECK:STDOUT:   %import_ref.2: type = import_ref ir1, inst+5, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+10, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %c_alias.ref.loc6: type = name_ref c_alias, %import_ref.2 [template = constants.%C]
// CHECK:STDOUT:   %c_alias_alias: type = bind_alias c_alias_alias, %import_ref.2 [template = constants.%C]
// CHECK:STDOUT:   %c_alias.ref.loc8: type = name_ref c_alias, %import_ref.2 [template = constants.%C]
// CHECK:STDOUT:   %.loc8: type = ptr_type C [template = constants.%.2]
// CHECK:STDOUT:   %b.var: ref C* = var b
// CHECK:STDOUT:   %b: ref C* = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- class3.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = ptr_type C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .c_alias_alias = %import_ref.1
// CHECK:STDOUT:     .b = %import_ref.2
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+9, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+14, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+7, unloaded
// CHECK:STDOUT:   %c_alias_alias.ref: type = name_ref c_alias_alias, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %.loc6: type = ptr_type C [template = constants.%.2]
// CHECK:STDOUT:   %c.var: ref C* = var c
// CHECK:STDOUT:   %c: ref C* = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var1.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .a_alias = %a_alias
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc4_9.2: type = converted %.loc4_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %a.ref: ref () = name_ref a, %a
// CHECK:STDOUT:   %a_alias: ref () = bind_alias a_alias, %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_14.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc4_14.2: init () = tuple_init () to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc4_15: init () = converted %.loc4_14.1, %.loc4_14.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc4_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %import_ref.1
// CHECK:STDOUT:     .a_alias = %import_ref.2
// CHECK:STDOUT:     .a_alias_alias = %a_alias_alias
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.2: ref () = import_ref ir1, inst+12, loaded
// CHECK:STDOUT:   %a_alias.ref: ref () = name_ref a_alias, %import_ref.2
// CHECK:STDOUT:   %a_alias_alias: ref () = bind_alias a_alias_alias, %import_ref.2
// CHECK:STDOUT:   %.loc8_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc8_9.2: type = converted %.loc8_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %b.var: ref () = var b
// CHECK:STDOUT:   %b: ref () = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_alias.ref: ref () = name_ref a_alias, file.%import_ref.2
// CHECK:STDOUT:   %.loc8_13: init () = tuple_init () to file.%b.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc8_20: init () = converted %a_alias.ref, %.loc8_13 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%b.var, %.loc8_20
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_var3.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_alias_alias = %import_ref.1
// CHECK:STDOUT:     .b = %import_ref.2
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: ref () = import_ref ir1, inst+5, loaded [template = <error>]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+9, unloaded
// CHECK:STDOUT:   %.loc12_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc12_9.2: type = converted %.loc12_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %c.var: ref () = var c
// CHECK:STDOUT:   %c: ref () = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_alias_alias.ref: ref () = name_ref a_alias_alias, file.%import_ref.1 [template = <error>]
// CHECK:STDOUT:   %.loc12_13: init () = tuple_init () to file.%c.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc12_26: init () = converted %a_alias_alias.ref, %.loc12_13 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%c.var, %.loc12_26
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
