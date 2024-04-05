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

var a: i32 = 0;

alias a_alias = a;

// --- var2.carbon

library "var2" api;

import library "var1";

alias a_alias_alias = a_alias;

var b: i32 = a_alias;

// --- fail_var3.carbon

library "var3" api;

// CHECK:STDERR: fail_var3.carbon:[[@LINE+6]]:1: In import.
// CHECK:STDERR: import library "var2";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: var2.carbon:8:5: ERROR: Semantics TODO: `Non-constant ImportRefUsed (comes up with var)`.
// CHECK:STDERR: var b: i32 = a_alias;
// CHECK:STDERR:     ^
import library "var2";

var c: i32 = a_alias_alias;

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
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .c_alias = %c_alias
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
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
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .c_alias_alias = %c_alias_alias
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.2: type = import_ref ir2, inst+6, loc_12 [template = constants.%C]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+11, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: invalid = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %import_ref.4 = import_ref ir2, inst+3, unloaded
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
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir2, inst+10, loc_11 [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir2, inst+15, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: invalid = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+8, unloaded
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
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .a_alias = %a_alias
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %a.ref: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %a_alias: ref i32 = bind_alias a_alias, %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign file.%a.var, %.loc4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %import_ref.1
// CHECK:STDOUT:     .a_alias = %import_ref.2
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a_alias_alias = %a_alias_alias
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir2, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.2: ref i32 = import_ref ir2, inst+8, loc_12
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %a_alias.ref: ref i32 = name_ref a_alias, %import_ref.2
// CHECK:STDOUT:   %a_alias_alias: ref i32 = bind_alias a_alias_alias, %import_ref.2
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_alias.ref: ref i32 = name_ref a_alias, file.%import_ref.2
// CHECK:STDOUT:   %.loc8: i32 = bind_value %a_alias.ref
// CHECK:STDOUT:   assign file.%b.var, %.loc8
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_var3.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_alias_alias = %import_ref.1
// CHECK:STDOUT:     .b = %import_ref.2
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: ref i32 = import_ref ir2, inst+5, loc_14 [template = <error>]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir2, inst+7, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %c.var: ref i32 = var c
// CHECK:STDOUT:   %c: ref i32 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_alias_alias.ref: ref i32 = name_ref a_alias_alias, file.%import_ref.1 [template = <error>]
// CHECK:STDOUT:   %.loc12: i32 = bind_value %a_alias_alias.ref
// CHECK:STDOUT:   assign file.%c.var, %.loc12
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
