// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/no_prelude/export_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/no_prelude/export_name.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- base.carbon

library "[[@TEST_NAME]]";

class C {
  var x: ();
};

namespace NS;
class NS.NSC {
  var y: ();
};

// --- export.carbon

library "[[@TEST_NAME]]";

import library "base";
export C;
export NS.NSC;

// --- not_reexporting.carbon

library "[[@TEST_NAME]]";

import library "export";

// --- export_export.carbon

library "[[@TEST_NAME]]";

import library "export";
export C;
export NS.NSC;

// --- export_in_impl.carbon

// This is just providing an API for the implicit import.
library "[[@TEST_NAME]]";

// ============================================================================
// Test files
// ============================================================================

// --- use_export.carbon

library "[[@TEST_NAME]]";

import library "export";

var c: C = {.x = ()};
var nsc: NS.NSC = {.y = ()};

// --- export_export.impl.carbon

impl library "[[@TEST_NAME]]";

var c: C = {.x = ()};
var nsc: NS.NSC = {.y = ()};

// --- use_export_export.carbon

library "[[@TEST_NAME]]";

import library "export_export";

var c: C = {.x = ()};
var nsc: NS.NSC = {.y = ()};

// --- fail_export_ns.carbon

library "[[@TEST_NAME]]";

import library "base";

// CHECK:STDERR: fail_export_ns.carbon:[[@LINE+8]]:1: error: only imported entities are valid for `export` [ExportNotImportedEntity]
// CHECK:STDERR: export NS;
// CHECK:STDERR: ^~~~~~~~~~
// CHECK:STDERR: fail_export_ns.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: base.carbon:8:1: note: name is declared here [ExportNotImportedEntitySource]
// CHECK:STDERR: namespace NS;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
export NS;

// --- fail_export_decl.carbon

library "[[@TEST_NAME]]";

class Local {}

// CHECK:STDERR: fail_export_decl.carbon:[[@LINE+7]]:1: error: only imported entities are valid for `export` [ExportNotImportedEntity]
// CHECK:STDERR: export Local;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_export_decl.carbon:[[@LINE-5]]:1: note: name is declared here [ExportNotImportedEntitySource]
// CHECK:STDERR: class Local {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
export Local;

// --- fail_export_member.carbon

library "[[@TEST_NAME]]";

import library "base";

// CHECK:STDERR: fail_export_member.carbon:[[@LINE+8]]:8: error: name qualifiers are only allowed for entities that provide a scope [QualifiedNameInNonScope]
// CHECK:STDERR: export C.x;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_export_member.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: base.carbon:4:1: note: referenced non-scope entity declared here [QualifiedNameNonScopeEntity]
// CHECK:STDERR: class C {
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
export C.x;

// --- import_both.carbon

library "[[@TEST_NAME]]";

import library "base";
import library "export";

var c: C = {.x = ()};
var nsc: NS.NSC = {.y = ()};

// --- import_both_reversed.carbon

library "[[@TEST_NAME]]";

import library "export";
import library "base";

var c: C = {.x = ()};
var nsc: NS.NSC = {.y = ()};

// --- fail_use_not_reexporting.carbon

library "[[@TEST_NAME]]";

import library "not_reexporting";

// CHECK:STDERR: fail_use_not_reexporting.carbon:[[@LINE+4]]:15: error: name `C` not found [NameNotFound]
// CHECK:STDERR: alias Local = C;
// CHECK:STDERR:               ^
// CHECK:STDERR:
alias Local = C;
// CHECK:STDERR: fail_use_not_reexporting.carbon:[[@LINE+4]]:17: error: name `NS` not found [NameNotFound]
// CHECK:STDERR: alias NSLocal = NS.NSC;
// CHECK:STDERR:                 ^~
// CHECK:STDERR:
alias NSLocal = NS.NSC;

// --- repeat_export.carbon

library "[[@TEST_NAME]]";

import library "base";

export C;
// CHECK:STDERR: repeat_export.carbon:[[@LINE+7]]:1: warning: `export` matches previous `export` [ExportRedundant]
// CHECK:STDERR: export C;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: repeat_export.carbon:[[@LINE-4]]:1: note: previous `export` here [ExportPrevious]
// CHECK:STDERR: export C;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
export C;

// --- use_repeat_export.carbon

library "[[@TEST_NAME]]";

import library "repeat_export";

var c: C = {.x = ()};

// --- fail_modifiers.carbon

library "[[@TEST_NAME]]";

import library "base";

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `private` not allowed on `export` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: private export C;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
private export C;

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9be: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [concrete]
// CHECK:STDOUT:   %NSC.elem: type = unbound_element_type %NSC, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9f4: <witness> = complete_type_witness %struct_type.y [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NSC = %NSC.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NSC.decl: type = class_decl @NSC [concrete = constants.%NSC] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc5_8: %C.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc5_3: %C.elem = var_pattern %.loc5_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %C.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [concrete = constants.%complete_type.9be]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .x = %.loc5_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC {
// CHECK:STDOUT:   %.loc10_8: %NSC.elem = field_decl y, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc10_3: %NSC.elem = var_pattern %.loc10_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %NSC.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.y [concrete = constants.%complete_type.9f4]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NSC
// CHECK:STDOUT:   .y = %.loc10_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9be: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [concrete]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9f4: <witness> = complete_type_witness %struct_type.y [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//base, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//base, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .NSC = file.%NSC
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.NSC: type = import_ref Main//base, NSC, loaded [concrete = constants.%NSC]
// CHECK:STDOUT:   %Main.import_ref.56d: <witness> = import_ref Main//base, loc6_1, loaded [concrete = constants.%complete_type.9be]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//base, inst14 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.276 = import_ref Main//base, loc5_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab: <witness> = import_ref Main//base, loc11_1, loaded [concrete = constants.%complete_type.9f4]
// CHECK:STDOUT:   %Main.import_ref.31c = import_ref Main//base, inst28 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.be6 = import_ref Main//base, loc10_8, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C: type = export C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %NSC: type = export NSC, imports.%Main.NSC [concrete = constants.%NSC]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.56d
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT:   .x = imports.%Main.import_ref.276
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC [from "base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.5ab
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.31c
// CHECK:STDOUT:   .y = imports.%Main.import_ref.be6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- not_reexporting.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C = import_ref Main//export, C, unloaded
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//export, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .NSC = %Main.NSC
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9be: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [concrete]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9f4: <witness> = complete_type_witness %struct_type.y [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//export, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//export, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .NSC = file.%NSC
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.NSC: type = import_ref Main//export, NSC, loaded [concrete = constants.%NSC]
// CHECK:STDOUT:   %Main.import_ref.ad3: <witness> = import_ref Main//export, inst23 [indirect], loaded [concrete = constants.%complete_type.9be]
// CHECK:STDOUT:   %Main.import_ref.6a9 = import_ref Main//export, inst24 [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.f67 = import_ref Main//export, inst25 [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.63c: <witness> = import_ref Main//export, inst31 [indirect], loaded [concrete = constants.%complete_type.9f4]
// CHECK:STDOUT:   %Main.import_ref.f0b = import_ref Main//export, inst32 [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.ebc = import_ref Main//export, inst33 [indirect], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C: type = export C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %NSC: type = export NSC, imports.%Main.NSC [concrete = constants.%NSC]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.ad3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.6a9
// CHECK:STDOUT:   .x = imports.%Main.import_ref.f67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC [from "export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.63c
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.f0b
// CHECK:STDOUT:   .y = imports.%Main.import_ref.ebc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9be: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [concrete]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9f4: <witness> = complete_type_witness %struct_type.y [concrete]
// CHECK:STDOUT:   %NSC.val: %NSC = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//export, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//export, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .NSC = %Main.NSC
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.NSC: type = import_ref Main//export, NSC, loaded [concrete = constants.%NSC]
// CHECK:STDOUT:   %Main.import_ref.ad3: <witness> = import_ref Main//export, inst23 [indirect], loaded [concrete = constants.%complete_type.9be]
// CHECK:STDOUT:   %Main.import_ref.6a9 = import_ref Main//export, inst24 [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.f67 = import_ref Main//export, inst25 [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.63c: <witness> = import_ref Main//export, inst31 [indirect], loaded [concrete = constants.%complete_type.9f4]
// CHECK:STDOUT:   %Main.import_ref.f0b = import_ref Main//export, inst32 [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.ebc = import_ref Main//export, inst33 [indirect], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .nsc = %nsc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %.loc6: %C = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %nsc.patt: %NSC = binding_pattern nsc
// CHECK:STDOUT:     %.loc7_1: %NSC = var_pattern %nsc.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc.var: ref %NSC = var nsc
// CHECK:STDOUT:   %.loc7_12: type = splice_block %NSC.ref [concrete = constants.%NSC] {
// CHECK:STDOUT:     %NS.ref: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:     %NSC.ref: type = name_ref NSC, imports.%Main.NSC [concrete = constants.%NSC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc: ref %NSC = bind_name nsc, %nsc.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.ad3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.6a9
// CHECK:STDOUT:   .x = imports.%Main.import_ref.f67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC [from "export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.63c
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.f0b
// CHECK:STDOUT:   .y = imports.%Main.import_ref.ebc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_20.1: %struct_type.x = struct_literal (%.loc6_19.1)
// CHECK:STDOUT:   %.loc6_20.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_19.2: init %empty_tuple.type = tuple_init () to %.loc6_20.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_20.3: init %empty_tuple.type = converted %.loc6_19.1, %.loc6_19.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_20.4: init %C = class_init (%.loc6_20.3), file.%c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc6_1: init %C = converted %.loc6_20.1, %.loc6_20.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_1
// CHECK:STDOUT:   %.loc7_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc7_27.1: %struct_type.y = struct_literal (%.loc7_26.1)
// CHECK:STDOUT:   %.loc7_27.2: ref %empty_tuple.type = class_element_access file.%nsc.var, element0
// CHECK:STDOUT:   %.loc7_26.2: init %empty_tuple.type = tuple_init () to %.loc7_27.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_27.3: init %empty_tuple.type = converted %.loc7_26.1, %.loc7_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_27.4: init %NSC = class_init (%.loc7_27.3), file.%nsc.var [concrete = constants.%NSC.val]
// CHECK:STDOUT:   %.loc7_1: init %NSC = converted %.loc7_27.1, %.loc7_27.4 [concrete = constants.%NSC.val]
// CHECK:STDOUT:   assign file.%nsc.var, %.loc7_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_export.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9be: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [concrete]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9f4: <witness> = complete_type_witness %struct_type.y [concrete]
// CHECK:STDOUT:   %NSC.val: %NSC = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//export_export, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//export_export, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .NSC = %Main.NSC
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.NSC: type = import_ref Main//export_export, NSC, loaded [concrete = constants.%NSC]
// CHECK:STDOUT:   %Main.import_ref.328: <witness> = import_ref Main//export_export, inst23 [indirect], loaded [concrete = constants.%complete_type.9be]
// CHECK:STDOUT:   %Main.import_ref.db8 = import_ref Main//export_export, inst24 [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.3ef = import_ref Main//export_export, inst25 [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.c12: <witness> = import_ref Main//export_export, inst31 [indirect], loaded [concrete = constants.%complete_type.9f4]
// CHECK:STDOUT:   %Main.import_ref.1cb = import_ref Main//export_export, inst32 [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.b18 = import_ref Main//export_export, inst33 [indirect], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .nsc = %nsc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_29.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_29.2 = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %.loc4: %C = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %nsc.patt: %NSC = binding_pattern nsc
// CHECK:STDOUT:     %.loc5_1: %NSC = var_pattern %nsc.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc.var: ref %NSC = var nsc
// CHECK:STDOUT:   %.loc5_12: type = splice_block %NSC.ref [concrete = constants.%NSC] {
// CHECK:STDOUT:     %NS.ref: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:     %NSC.ref: type = name_ref NSC, imports.%Main.NSC [concrete = constants.%NSC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc: ref %NSC = bind_name nsc, %nsc.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "export_export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.328
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.db8
// CHECK:STDOUT:   .x = imports.%Main.import_ref.3ef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC [from "export_export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.c12
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.1cb
// CHECK:STDOUT:   .y = imports.%Main.import_ref.b18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_20.1: %struct_type.x = struct_literal (%.loc4_19.1)
// CHECK:STDOUT:   %.loc4_20.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc4_19.2: init %empty_tuple.type = tuple_init () to %.loc4_20.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_20.3: init %empty_tuple.type = converted %.loc4_19.1, %.loc4_19.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_20.4: init %C = class_init (%.loc4_20.3), file.%c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc4_1: init %C = converted %.loc4_20.1, %.loc4_20.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc4_1
// CHECK:STDOUT:   %.loc5_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc5_27.1: %struct_type.y = struct_literal (%.loc5_26.1)
// CHECK:STDOUT:   %.loc5_27.2: ref %empty_tuple.type = class_element_access file.%nsc.var, element0
// CHECK:STDOUT:   %.loc5_26.2: init %empty_tuple.type = tuple_init () to %.loc5_27.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_27.3: init %empty_tuple.type = converted %.loc5_26.1, %.loc5_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_27.4: init %NSC = class_init (%.loc5_27.3), file.%nsc.var [concrete = constants.%NSC.val]
// CHECK:STDOUT:   %.loc5_1: init %NSC = converted %.loc5_27.1, %.loc5_27.4 [concrete = constants.%NSC.val]
// CHECK:STDOUT:   assign file.%nsc.var, %.loc5_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9be: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [concrete]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9f4: <witness> = complete_type_witness %struct_type.y [concrete]
// CHECK:STDOUT:   %NSC.val: %NSC = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//export_export, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//export_export, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .NSC = %Main.NSC
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.NSC: type = import_ref Main//export_export, NSC, loaded [concrete = constants.%NSC]
// CHECK:STDOUT:   %Main.import_ref.328: <witness> = import_ref Main//export_export, inst23 [indirect], loaded [concrete = constants.%complete_type.9be]
// CHECK:STDOUT:   %Main.import_ref.db8 = import_ref Main//export_export, inst24 [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.3ef = import_ref Main//export_export, inst25 [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.c12: <witness> = import_ref Main//export_export, inst31 [indirect], loaded [concrete = constants.%complete_type.9f4]
// CHECK:STDOUT:   %Main.import_ref.1cb = import_ref Main//export_export, inst32 [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.b18 = import_ref Main//export_export, inst33 [indirect], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .nsc = %nsc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %.loc6: %C = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %nsc.patt: %NSC = binding_pattern nsc
// CHECK:STDOUT:     %.loc7_1: %NSC = var_pattern %nsc.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc.var: ref %NSC = var nsc
// CHECK:STDOUT:   %.loc7_12: type = splice_block %NSC.ref [concrete = constants.%NSC] {
// CHECK:STDOUT:     %NS.ref: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:     %NSC.ref: type = name_ref NSC, imports.%Main.NSC [concrete = constants.%NSC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc: ref %NSC = bind_name nsc, %nsc.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "export_export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.328
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.db8
// CHECK:STDOUT:   .x = imports.%Main.import_ref.3ef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC [from "export_export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.c12
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.1cb
// CHECK:STDOUT:   .y = imports.%Main.import_ref.b18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_20.1: %struct_type.x = struct_literal (%.loc6_19.1)
// CHECK:STDOUT:   %.loc6_20.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_19.2: init %empty_tuple.type = tuple_init () to %.loc6_20.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_20.3: init %empty_tuple.type = converted %.loc6_19.1, %.loc6_19.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_20.4: init %C = class_init (%.loc6_20.3), file.%c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc6_1: init %C = converted %.loc6_20.1, %.loc6_20.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_1
// CHECK:STDOUT:   %.loc7_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc7_27.1: %struct_type.y = struct_literal (%.loc7_26.1)
// CHECK:STDOUT:   %.loc7_27.2: ref %empty_tuple.type = class_element_access file.%nsc.var, element0
// CHECK:STDOUT:   %.loc7_26.2: init %empty_tuple.type = tuple_init () to %.loc7_27.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_27.3: init %empty_tuple.type = converted %.loc7_26.1, %.loc7_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_27.4: init %NSC = class_init (%.loc7_27.3), file.%nsc.var [concrete = constants.%NSC.val]
// CHECK:STDOUT:   %.loc7_1: init %NSC = converted %.loc7_27.1, %.loc7_27.4 [concrete = constants.%NSC.val]
// CHECK:STDOUT:   assign file.%nsc.var, %.loc7_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_export_ns.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C = import_ref Main//base, C, unloaded
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//base, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .NSC = %Main.NSC
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_export_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Local: type = class_type @Local [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Local = %Local.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Local.decl: type = class_decl @Local [concrete = constants.%Local] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Local {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Local
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_export_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//base, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//base, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .NSC = %Main.NSC
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.56d: <witness> = import_ref Main//base, loc6_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//base, inst14 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.276 = import_ref Main//base, loc5_8, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.56d
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT:   .x = imports.%Main.import_ref.276
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_both.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9be: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [concrete]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9f4: <witness> = complete_type_witness %struct_type.y [concrete]
// CHECK:STDOUT:   %NSC.val: %NSC = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//base, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//base, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .NSC = %Main.NSC
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.NSC: type = import_ref Main//base, NSC, loaded [concrete = constants.%NSC]
// CHECK:STDOUT:   %Main.import_ref.56d: <witness> = import_ref Main//base, loc6_1, loaded [concrete = constants.%complete_type.9be]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//base, inst14 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.276 = import_ref Main//base, loc5_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab: <witness> = import_ref Main//base, loc11_1, loaded [concrete = constants.%complete_type.9f4]
// CHECK:STDOUT:   %Main.import_ref.31c = import_ref Main//base, inst28 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.be6 = import_ref Main//base, loc10_8, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .nsc = %nsc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %.loc7: %C = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %nsc.patt: %NSC = binding_pattern nsc
// CHECK:STDOUT:     %.loc8_1: %NSC = var_pattern %nsc.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc.var: ref %NSC = var nsc
// CHECK:STDOUT:   %.loc8_12: type = splice_block %NSC.ref [concrete = constants.%NSC] {
// CHECK:STDOUT:     %NS.ref: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:     %NSC.ref: type = name_ref NSC, imports.%Main.NSC [concrete = constants.%NSC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc: ref %NSC = bind_name nsc, %nsc.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.56d
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT:   .x = imports.%Main.import_ref.276
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC [from "base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.5ab
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.31c
// CHECK:STDOUT:   .y = imports.%Main.import_ref.be6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc7_20.1: %struct_type.x = struct_literal (%.loc7_19.1)
// CHECK:STDOUT:   %.loc7_20.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc7_19.2: init %empty_tuple.type = tuple_init () to %.loc7_20.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_20.3: init %empty_tuple.type = converted %.loc7_19.1, %.loc7_19.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_20.4: init %C = class_init (%.loc7_20.3), file.%c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc7_1: init %C = converted %.loc7_20.1, %.loc7_20.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc7_1
// CHECK:STDOUT:   %.loc8_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc8_27.1: %struct_type.y = struct_literal (%.loc8_26.1)
// CHECK:STDOUT:   %.loc8_27.2: ref %empty_tuple.type = class_element_access file.%nsc.var, element0
// CHECK:STDOUT:   %.loc8_26.2: init %empty_tuple.type = tuple_init () to %.loc8_27.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_27.3: init %empty_tuple.type = converted %.loc8_26.1, %.loc8_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_27.4: init %NSC = class_init (%.loc8_27.3), file.%nsc.var [concrete = constants.%NSC.val]
// CHECK:STDOUT:   %.loc8_1: init %NSC = converted %.loc8_27.1, %.loc8_27.4 [concrete = constants.%NSC.val]
// CHECK:STDOUT:   assign file.%nsc.var, %.loc8_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_both_reversed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9be: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [concrete]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9f4: <witness> = complete_type_witness %struct_type.y [concrete]
// CHECK:STDOUT:   %NSC.val: %NSC = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//export, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//export, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .NSC = %Main.NSC
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.NSC: type = import_ref Main//export, NSC, loaded [concrete = constants.%NSC]
// CHECK:STDOUT:   %Main.import_ref.ad3: <witness> = import_ref Main//export, inst23 [indirect], loaded [concrete = constants.%complete_type.9be]
// CHECK:STDOUT:   %Main.import_ref.6a9 = import_ref Main//export, inst24 [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.f67 = import_ref Main//export, inst25 [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.63c: <witness> = import_ref Main//export, inst31 [indirect], loaded [concrete = constants.%complete_type.9f4]
// CHECK:STDOUT:   %Main.import_ref.f0b = import_ref Main//export, inst32 [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.ebc = import_ref Main//export, inst33 [indirect], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .nsc = %nsc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %.loc7: %C = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %nsc.patt: %NSC = binding_pattern nsc
// CHECK:STDOUT:     %.loc8_1: %NSC = var_pattern %nsc.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc.var: ref %NSC = var nsc
// CHECK:STDOUT:   %.loc8_12: type = splice_block %NSC.ref [concrete = constants.%NSC] {
// CHECK:STDOUT:     %NS.ref: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:     %NSC.ref: type = name_ref NSC, imports.%Main.NSC [concrete = constants.%NSC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc: ref %NSC = bind_name nsc, %nsc.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.ad3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.6a9
// CHECK:STDOUT:   .x = imports.%Main.import_ref.f67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC [from "export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.63c
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.f0b
// CHECK:STDOUT:   .y = imports.%Main.import_ref.ebc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc7_20.1: %struct_type.x = struct_literal (%.loc7_19.1)
// CHECK:STDOUT:   %.loc7_20.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc7_19.2: init %empty_tuple.type = tuple_init () to %.loc7_20.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_20.3: init %empty_tuple.type = converted %.loc7_19.1, %.loc7_19.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_20.4: init %C = class_init (%.loc7_20.3), file.%c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc7_1: init %C = converted %.loc7_20.1, %.loc7_20.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc7_1
// CHECK:STDOUT:   %.loc8_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc8_27.1: %struct_type.y = struct_literal (%.loc8_26.1)
// CHECK:STDOUT:   %.loc8_27.2: ref %empty_tuple.type = class_element_access file.%nsc.var, element0
// CHECK:STDOUT:   %.loc8_26.2: init %empty_tuple.type = tuple_init () to %.loc8_27.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_27.3: init %empty_tuple.type = converted %.loc8_26.1, %.loc8_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_27.4: init %NSC = class_init (%.loc8_27.3), file.%nsc.var [concrete = constants.%NSC.val]
// CHECK:STDOUT:   %.loc8_1: init %NSC = converted %.loc8_27.1, %.loc8_27.4 [concrete = constants.%NSC.val]
// CHECK:STDOUT:   assign file.%nsc.var, %.loc8_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_not_reexporting.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .Local = %Local
// CHECK:STDOUT:     .NS = <poisoned>
// CHECK:STDOUT:     .NSLocal = %NSLocal
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C.ref: <error> = name_ref C, <error> [concrete = <error>]
// CHECK:STDOUT:   %Local: <error> = bind_alias Local, <error> [concrete = <error>]
// CHECK:STDOUT:   %NS.ref: <error> = name_ref NS, <error> [concrete = <error>]
// CHECK:STDOUT:   %NSC.ref: <error> = name_ref NSC, <error> [concrete = <error>]
// CHECK:STDOUT:   %NSLocal: <error> = bind_alias NSLocal, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- repeat_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//base, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//base, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .NSC = %Main.NSC
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.56d: <witness> = import_ref Main//base, loc6_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//base, inst14 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.276 = import_ref Main//base, loc5_8, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C: type = export C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.56d
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT:   .x = imports.%Main.import_ref.276
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_repeat_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//repeat_export, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.ad3: <witness> = import_ref Main//repeat_export, inst23 [indirect], loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.6a9 = import_ref Main//repeat_export, inst24 [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.f67 = import_ref Main//repeat_export, inst25 [indirect], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %.loc6: %C = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "repeat_export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.ad3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.6a9
// CHECK:STDOUT:   .x = imports.%Main.import_ref.f67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_20.1: %struct_type.x = struct_literal (%.loc6_19.1)
// CHECK:STDOUT:   %.loc6_20.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_19.2: init %empty_tuple.type = tuple_init () to %.loc6_20.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_20.3: init %empty_tuple.type = converted %.loc6_19.1, %.loc6_19.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_20.4: init %C = class_init (%.loc6_20.3), file.%c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc6_1: init %C = converted %.loc6_20.1, %.loc6_20.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//base, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//base, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .NSC = %Main.NSC
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.56d: <witness> = import_ref Main//base, loc6_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//base, inst14 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.276 = import_ref Main//base, loc5_8, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C: type = export C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.56d
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT:   .x = imports.%Main.import_ref.276
// CHECK:STDOUT: }
// CHECK:STDOUT:
