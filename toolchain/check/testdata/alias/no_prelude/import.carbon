// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/no_prelude/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/no_prelude/import.carbon

// --- class1.carbon

library "[[@TEST_NAME]]";

class C {}

alias c_alias = C;

var a: C*;

// --- class2.carbon

library "[[@TEST_NAME]]";

import library "class1";

alias c_alias_alias = c_alias;

var b: c_alias*;

// --- class3.carbon

library "[[@TEST_NAME]]";

import library "class2";

var c: c_alias_alias*;

// --- var1.carbon

library "[[@TEST_NAME]]";

var a: () = ();

alias a_alias = a;

// --- var2.carbon

library "[[@TEST_NAME]]";

import library "var1";

alias a_alias_alias = a_alias;

var b: () = a_alias;

// --- fail_var3.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_var3.carbon:[[@LINE+4]]:1: in import [InImport]
// CHECK:STDERR: var2.carbon:8:5: error: semantics TODO: `Non-constant ImportRefLoaded (comes up with var)` [SemanticsTodo]
// CHECK:STDERR: var b: () = a_alias;
// CHECK:STDERR:     ^
import library "var2";

var c: () = a_alias_alias;

// CHECK:STDOUT: --- class1.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %.3: type = ptr_type %C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .c_alias = %c_alias
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref.loc6: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %c_alias: type = bind_alias c_alias, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc8: type = ptr_type %C [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- class2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.3: type = ptr_type %C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//class1, inst+1, unloaded
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//class1, inst+7, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//class1, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//class1, inst+2, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .c_alias = imports.%import_ref.2
// CHECK:STDOUT:     .a = imports.%import_ref.3
// CHECK:STDOUT:     .c_alias_alias = %c_alias_alias
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %c_alias.ref.loc6: type = name_ref c_alias, imports.%import_ref.2 [template = constants.%C]
// CHECK:STDOUT:   %c_alias_alias: type = bind_alias c_alias_alias, imports.%import_ref.2 [template = constants.%C]
// CHECK:STDOUT:   %c_alias.ref.loc8: type = name_ref c_alias, imports.%import_ref.2 [template = constants.%C]
// CHECK:STDOUT:   %.loc8: type = ptr_type %C [template = constants.%.3]
// CHECK:STDOUT:   %b.var: ref %.3 = var b
// CHECK:STDOUT:   %b: ref %.3 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- class3.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.3: type = ptr_type %C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//class2, inst+11, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//class2, inst+16, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//class2, inst+9, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .c_alias_alias = imports.%import_ref.1
// CHECK:STDOUT:     .b = imports.%import_ref.2
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %c_alias_alias.ref: type = name_ref c_alias_alias, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %.loc6: type = ptr_type %C [template = constants.%.3]
// CHECK:STDOUT:   %c.var: ref %.3 = var c
// CHECK:STDOUT:   %c: ref %.3 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var1.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .a_alias = %a_alias
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_9.2: type = converted %.loc4_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   %a.ref: ref %empty_tuple.type = name_ref a, %a
// CHECK:STDOUT:   %a_alias: ref %empty_tuple.type = bind_alias a_alias, %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_14.2: init %empty_tuple.type = tuple_init () to file.%a.var [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_15: init %empty_tuple.type = converted %.loc4_14.1, %.loc4_14.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc4_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//var1, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.2: ref %empty_tuple.type = import_ref Main//var1, inst+12, loaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = imports.%import_ref.1
// CHECK:STDOUT:     .a_alias = imports.%import_ref.2
// CHECK:STDOUT:     .a_alias_alias = %a_alias_alias
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %a_alias.ref: ref %empty_tuple.type = name_ref a_alias, imports.%import_ref.2
// CHECK:STDOUT:   %a_alias_alias: ref %empty_tuple.type = bind_alias a_alias_alias, imports.%import_ref.2
// CHECK:STDOUT:   %.loc8_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc8_9.2: type = converted %.loc8_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %b.var: ref %empty_tuple.type = var b
// CHECK:STDOUT:   %b: ref %empty_tuple.type = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_alias.ref: ref %empty_tuple.type = name_ref a_alias, imports.%import_ref.2
// CHECK:STDOUT:   %.loc8_13: init %empty_tuple.type = tuple_init () to file.%b.var [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_20: init %empty_tuple.type = converted %a_alias.ref, %.loc8_13 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%b.var, %.loc8_20
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_var3.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: ref %empty_tuple.type = import_ref Main//var2, inst+6, loaded [template = <error>]
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//var2, inst+10, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_alias_alias = imports.%import_ref.1
// CHECK:STDOUT:     .b = imports.%import_ref.2
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %.loc10_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc10_9.2: type = converted %.loc10_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %c.var: ref %empty_tuple.type = var c
// CHECK:STDOUT:   %c: ref %empty_tuple.type = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_alias_alias.ref: ref %empty_tuple.type = name_ref a_alias_alias, imports.%import_ref.1 [template = <error>]
// CHECK:STDOUT:   %.loc10_13: init %empty_tuple.type = tuple_init () to file.%c.var [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_26: init %empty_tuple.type = converted %a_alias_alias.ref, %.loc10_13 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%c.var, %.loc10_26
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
