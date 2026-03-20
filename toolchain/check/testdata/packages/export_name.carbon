// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/export_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/export_name.carbon

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
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
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
// CHECK:STDOUT:   %.loc11.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11.2: type = converted %.loc11.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc9: %C.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x [concrete = constants.%complete_type.9be]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .x = %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC {
// CHECK:STDOUT:   %.loc27.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc27.2: type = converted %.loc27.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc25: %NSC.elem = field_decl y, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.y [concrete = constants.%complete_type.9f4]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NSC
// CHECK:STDOUT:   .y = %.loc25
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
// CHECK:STDOUT:   %Main.import_ref.56d: <witness> = import_ref Main//base, loc13, loaded [concrete = constants.%complete_type.9be]
// CHECK:STDOUT:   %Main.import_ref.743 = import_ref Main//base, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.fec = import_ref Main//base, loc9, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab: <witness> = import_ref Main//base, loc29, loaded [concrete = constants.%complete_type.9f4]
// CHECK:STDOUT:   %Main.import_ref.7da = import_ref Main//base, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.045 = import_ref Main//base, loc25, unloaded
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
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.743
// CHECK:STDOUT:   .x = imports.%Main.import_ref.fec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC [from "base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.5ab
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.7da
// CHECK:STDOUT:   .y = imports.%Main.import_ref.045
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
// CHECK:STDOUT:   %Main.NSC = import_ref Main//export, NSC, unloaded
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
// CHECK:STDOUT:   %Main.import_ref.ad3: <witness> = import_ref Main//export, inst{{[0-9A-F]+}} [indirect], loaded [concrete = constants.%complete_type.9be]
// CHECK:STDOUT:   %Main.import_ref.a60 = import_ref Main//export, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.e6b = import_ref Main//export, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.63c: <witness> = import_ref Main//export, inst{{[0-9A-F]+}} [indirect], loaded [concrete = constants.%complete_type.9f4]
// CHECK:STDOUT:   %Main.import_ref.8ad = import_ref Main//export, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.be6 = import_ref Main//export, inst{{[0-9A-F]+}} [indirect], unloaded
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
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.a60
// CHECK:STDOUT:   .x = imports.%Main.import_ref.e6b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC [from "export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.63c
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.8ad
// CHECK:STDOUT:   .y = imports.%Main.import_ref.be6
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
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct.948: %struct_type.x = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %.1d2: ref %empty_tuple.type = class_element_access file.%c.var, element0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [concrete]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9f4: <witness> = complete_type_witness %struct_type.y [concrete]
// CHECK:STDOUT:   %pattern_type.ccb: type = pattern_type %NSC [concrete]
// CHECK:STDOUT:   %struct.8ce: %struct_type.y = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %.160: ref %empty_tuple.type = class_element_access file.%nsc.var, element0 [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.ad3: <witness> = import_ref Main//export, inst{{[0-9A-F]+}} [indirect], loaded [concrete = constants.%complete_type.9be]
// CHECK:STDOUT:   %Main.import_ref.a60 = import_ref Main//export, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.e6b = import_ref Main//export, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.63c: <witness> = import_ref Main//export, inst{{[0-9A-F]+}} [indirect], loaded [concrete = constants.%complete_type.9f4]
// CHECK:STDOUT:   %Main.import_ref.8ad = import_ref Main//export, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.be6 = import_ref Main//export, inst{{[0-9A-F]+}} [indirect], unloaded
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
// CHECK:STDOUT:     %c.patt: %pattern_type.7c7 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.7c7 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt [concrete]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %nsc.patt: %pattern_type.ccb = ref_binding_pattern nsc [concrete]
// CHECK:STDOUT:     %nsc.var_patt: %pattern_type.ccb = var_pattern %nsc.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc.var: ref %NSC = var %nsc.var_patt [concrete]
// CHECK:STDOUT:   %.loc25: type = splice_block %NSC.ref [concrete = constants.%NSC] {
// CHECK:STDOUT:     %NS.ref: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:     %NSC.ref: type = name_ref NSC, imports.%Main.NSC [concrete = constants.%NSC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc: ref %NSC = ref_binding nsc, %nsc.var [concrete = %nsc.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.ad3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.a60
// CHECK:STDOUT:   .x = imports.%Main.import_ref.e6b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC [from "export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.63c
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.8ad
// CHECK:STDOUT:   .y = imports.%Main.import_ref.be6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19.1: %struct_type.x = struct_literal (%.loc18.1) [concrete = constants.%struct.948]
// CHECK:STDOUT:   %.loc19.2: ref %empty_tuple.type = class_element_access file.%c.var, element0 [concrete = constants.%.1d2]
// CHECK:STDOUT:   %.loc18.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19.3: init %empty_tuple.type = converted %.loc18.1, %.loc18.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19.4: init %C to file.%c.var = class_init (%.loc19.3) [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8: init %C = converted %.loc19.1, %.loc19.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc8
// CHECK:STDOUT:   %.loc33.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc34.1: %struct_type.y = struct_literal (%.loc33.1) [concrete = constants.%struct.8ce]
// CHECK:STDOUT:   %.loc34.2: ref %empty_tuple.type = class_element_access file.%nsc.var, element0 [concrete = constants.%.160]
// CHECK:STDOUT:   %.loc33.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc34.3: init %empty_tuple.type = converted %.loc33.1, %.loc33.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc34.4: init %NSC to file.%nsc.var = class_init (%.loc34.3) [concrete = constants.%NSC.val]
// CHECK:STDOUT:   %.loc21: init %NSC = converted %.loc34.1, %.loc34.4 [concrete = constants.%NSC.val]
// CHECK:STDOUT:   assign file.%nsc.var, %.loc21
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
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct.948: %struct_type.x = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %.1d2: ref %empty_tuple.type = class_element_access file.%c.var, element0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [concrete]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9f4: <witness> = complete_type_witness %struct_type.y [concrete]
// CHECK:STDOUT:   %pattern_type.fd4: type = pattern_type %NSC [concrete]
// CHECK:STDOUT:   %struct.8ce: %struct_type.y = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %.2e1: ref %empty_tuple.type = class_element_access file.%nsc.var, element0 [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.328: <witness> = import_ref Main//export_export, inst{{[0-9A-F]+}} [indirect], loaded [concrete = constants.%complete_type.9be]
// CHECK:STDOUT:   %Main.import_ref.76e = import_ref Main//export_export, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.d5d = import_ref Main//export_export, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.c12: <witness> = import_ref Main//export_export, inst{{[0-9A-F]+}} [indirect], loaded [concrete = constants.%complete_type.9f4]
// CHECK:STDOUT:   %Main.import_ref.89a = import_ref Main//export_export, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.64d = import_ref Main//export_export, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .nsc = %nsc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc4.1 = import <none>
// CHECK:STDOUT:   %default.import.loc4.2 = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.7c7 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.7c7 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt [concrete]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %nsc.patt: %pattern_type.fd4 = ref_binding_pattern nsc [concrete]
// CHECK:STDOUT:     %nsc.var_patt: %pattern_type.fd4 = var_pattern %nsc.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc.var: ref %NSC = var %nsc.var_patt [concrete]
// CHECK:STDOUT:   %.loc22: type = splice_block %NSC.ref [concrete = constants.%NSC] {
// CHECK:STDOUT:     %NS.ref: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:     %NSC.ref: type = name_ref NSC, imports.%Main.NSC [concrete = constants.%NSC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc: ref %NSC = ref_binding nsc, %nsc.var [concrete = %nsc.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "export_export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.328
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.76e
// CHECK:STDOUT:   .x = imports.%Main.import_ref.d5d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC [from "export_export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.c12
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.89a
// CHECK:STDOUT:   .y = imports.%Main.import_ref.64d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16.1: %struct_type.x = struct_literal (%.loc15.1) [concrete = constants.%struct.948]
// CHECK:STDOUT:   %.loc16.2: ref %empty_tuple.type = class_element_access file.%c.var, element0 [concrete = constants.%.1d2]
// CHECK:STDOUT:   %.loc15.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16.3: init %empty_tuple.type = converted %.loc15.1, %.loc15.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16.4: init %C to file.%c.var = class_init (%.loc16.3) [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc5: init %C = converted %.loc16.1, %.loc16.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc5
// CHECK:STDOUT:   %.loc30.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc31.1: %struct_type.y = struct_literal (%.loc30.1) [concrete = constants.%struct.8ce]
// CHECK:STDOUT:   %.loc31.2: ref %empty_tuple.type = class_element_access file.%nsc.var, element0 [concrete = constants.%.2e1]
// CHECK:STDOUT:   %.loc30.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc31.3: init %empty_tuple.type = converted %.loc30.1, %.loc30.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc31.4: init %NSC to file.%nsc.var = class_init (%.loc31.3) [concrete = constants.%NSC.val]
// CHECK:STDOUT:   %.loc18: init %NSC = converted %.loc31.1, %.loc31.4 [concrete = constants.%NSC.val]
// CHECK:STDOUT:   assign file.%nsc.var, %.loc18
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
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct.948: %struct_type.x = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %.1d2: ref %empty_tuple.type = class_element_access file.%c.var, element0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [concrete]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9f4: <witness> = complete_type_witness %struct_type.y [concrete]
// CHECK:STDOUT:   %pattern_type.fd4: type = pattern_type %NSC [concrete]
// CHECK:STDOUT:   %struct.8ce: %struct_type.y = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %.2e1: ref %empty_tuple.type = class_element_access file.%nsc.var, element0 [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.328: <witness> = import_ref Main//export_export, inst{{[0-9A-F]+}} [indirect], loaded [concrete = constants.%complete_type.9be]
// CHECK:STDOUT:   %Main.import_ref.76e = import_ref Main//export_export, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.d5d = import_ref Main//export_export, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.c12: <witness> = import_ref Main//export_export, inst{{[0-9A-F]+}} [indirect], loaded [concrete = constants.%complete_type.9f4]
// CHECK:STDOUT:   %Main.import_ref.89a = import_ref Main//export_export, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.64d = import_ref Main//export_export, inst{{[0-9A-F]+}} [indirect], unloaded
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
// CHECK:STDOUT:     %c.patt: %pattern_type.7c7 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.7c7 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt [concrete]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %nsc.patt: %pattern_type.fd4 = ref_binding_pattern nsc [concrete]
// CHECK:STDOUT:     %nsc.var_patt: %pattern_type.fd4 = var_pattern %nsc.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc.var: ref %NSC = var %nsc.var_patt [concrete]
// CHECK:STDOUT:   %.loc25: type = splice_block %NSC.ref [concrete = constants.%NSC] {
// CHECK:STDOUT:     %NS.ref: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:     %NSC.ref: type = name_ref NSC, imports.%Main.NSC [concrete = constants.%NSC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc: ref %NSC = ref_binding nsc, %nsc.var [concrete = %nsc.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "export_export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.328
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.76e
// CHECK:STDOUT:   .x = imports.%Main.import_ref.d5d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC [from "export_export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.c12
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.89a
// CHECK:STDOUT:   .y = imports.%Main.import_ref.64d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19.1: %struct_type.x = struct_literal (%.loc18.1) [concrete = constants.%struct.948]
// CHECK:STDOUT:   %.loc19.2: ref %empty_tuple.type = class_element_access file.%c.var, element0 [concrete = constants.%.1d2]
// CHECK:STDOUT:   %.loc18.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19.3: init %empty_tuple.type = converted %.loc18.1, %.loc18.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19.4: init %C to file.%c.var = class_init (%.loc19.3) [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8: init %C = converted %.loc19.1, %.loc19.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc8
// CHECK:STDOUT:   %.loc33.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc34.1: %struct_type.y = struct_literal (%.loc33.1) [concrete = constants.%struct.8ce]
// CHECK:STDOUT:   %.loc34.2: ref %empty_tuple.type = class_element_access file.%nsc.var, element0 [concrete = constants.%.2e1]
// CHECK:STDOUT:   %.loc33.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc34.3: init %empty_tuple.type = converted %.loc33.1, %.loc33.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc34.4: init %NSC to file.%nsc.var = class_init (%.loc34.3) [concrete = constants.%NSC.val]
// CHECK:STDOUT:   %.loc21: init %NSC = converted %.loc34.1, %.loc34.4 [concrete = constants.%NSC.val]
// CHECK:STDOUT:   assign file.%nsc.var, %.loc21
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
// CHECK:STDOUT:   %Main.NSC = import_ref Main//base, NSC, unloaded
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
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
// CHECK:STDOUT:   %Main.NSC = import_ref Main//base, NSC, unloaded
// CHECK:STDOUT:   %Main.import_ref.56d: <witness> = import_ref Main//base, loc13, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.743 = import_ref Main//base, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.fec = import_ref Main//base, loc9, unloaded
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
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.743
// CHECK:STDOUT:   .x = imports.%Main.import_ref.fec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_both.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9be: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct.948: %struct_type.x = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %.1d2: ref %empty_tuple.type = class_element_access file.%c.var, element0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [concrete]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9f4: <witness> = complete_type_witness %struct_type.y [concrete]
// CHECK:STDOUT:   %pattern_type.b52: type = pattern_type %NSC [concrete]
// CHECK:STDOUT:   %struct.8ce: %struct_type.y = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %.b2f: ref %empty_tuple.type = class_element_access file.%nsc.var, element0 [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.56d: <witness> = import_ref Main//base, loc13, loaded [concrete = constants.%complete_type.9be]
// CHECK:STDOUT:   %Main.import_ref.743 = import_ref Main//base, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.fec = import_ref Main//base, loc9, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab: <witness> = import_ref Main//base, loc29, loaded [concrete = constants.%complete_type.9f4]
// CHECK:STDOUT:   %Main.import_ref.7da = import_ref Main//base, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.045 = import_ref Main//base, loc25, unloaded
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
// CHECK:STDOUT:     %c.patt: %pattern_type.7c7 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.7c7 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt [concrete]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %nsc.patt: %pattern_type.b52 = ref_binding_pattern nsc [concrete]
// CHECK:STDOUT:     %nsc.var_patt: %pattern_type.b52 = var_pattern %nsc.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc.var: ref %NSC = var %nsc.var_patt [concrete]
// CHECK:STDOUT:   %.loc29: type = splice_block %NSC.ref [concrete = constants.%NSC] {
// CHECK:STDOUT:     %NS.ref: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:     %NSC.ref: type = name_ref NSC, imports.%Main.NSC [concrete = constants.%NSC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc: ref %NSC = ref_binding nsc, %nsc.var [concrete = %nsc.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.56d
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.743
// CHECK:STDOUT:   .x = imports.%Main.import_ref.fec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC [from "base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.5ab
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.7da
// CHECK:STDOUT:   .y = imports.%Main.import_ref.045
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc23.1: %struct_type.x = struct_literal (%.loc22.1) [concrete = constants.%struct.948]
// CHECK:STDOUT:   %.loc23.2: ref %empty_tuple.type = class_element_access file.%c.var, element0 [concrete = constants.%.1d2]
// CHECK:STDOUT:   %.loc22.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc23.3: init %empty_tuple.type = converted %.loc22.1, %.loc22.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc23.4: init %C to file.%c.var = class_init (%.loc23.3) [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc12: init %C = converted %.loc23.1, %.loc23.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc12
// CHECK:STDOUT:   %.loc37.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc38.1: %struct_type.y = struct_literal (%.loc37.1) [concrete = constants.%struct.8ce]
// CHECK:STDOUT:   %.loc38.2: ref %empty_tuple.type = class_element_access file.%nsc.var, element0 [concrete = constants.%.b2f]
// CHECK:STDOUT:   %.loc37.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc38.3: init %empty_tuple.type = converted %.loc37.1, %.loc37.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc38.4: init %NSC to file.%nsc.var = class_init (%.loc38.3) [concrete = constants.%NSC.val]
// CHECK:STDOUT:   %.loc25: init %NSC = converted %.loc38.1, %.loc38.4 [concrete = constants.%NSC.val]
// CHECK:STDOUT:   assign file.%nsc.var, %.loc25
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
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct.948: %struct_type.x = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %.1d2: ref %empty_tuple.type = class_element_access file.%c.var, element0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [concrete]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type.9f4: <witness> = complete_type_witness %struct_type.y [concrete]
// CHECK:STDOUT:   %pattern_type.ccb: type = pattern_type %NSC [concrete]
// CHECK:STDOUT:   %struct.8ce: %struct_type.y = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %.160: ref %empty_tuple.type = class_element_access file.%nsc.var, element0 [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.ad3: <witness> = import_ref Main//export, inst{{[0-9A-F]+}} [indirect], loaded [concrete = constants.%complete_type.9be]
// CHECK:STDOUT:   %Main.import_ref.a60 = import_ref Main//export, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.e6b = import_ref Main//export, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.63c: <witness> = import_ref Main//export, inst{{[0-9A-F]+}} [indirect], loaded [concrete = constants.%complete_type.9f4]
// CHECK:STDOUT:   %Main.import_ref.8ad = import_ref Main//export, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.be6 = import_ref Main//export, inst{{[0-9A-F]+}} [indirect], unloaded
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
// CHECK:STDOUT:     %c.patt: %pattern_type.7c7 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.7c7 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt [concrete]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %nsc.patt: %pattern_type.ccb = ref_binding_pattern nsc [concrete]
// CHECK:STDOUT:     %nsc.var_patt: %pattern_type.ccb = var_pattern %nsc.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc.var: ref %NSC = var %nsc.var_patt [concrete]
// CHECK:STDOUT:   %.loc29: type = splice_block %NSC.ref [concrete = constants.%NSC] {
// CHECK:STDOUT:     %NS.ref: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:     %NSC.ref: type = name_ref NSC, imports.%Main.NSC [concrete = constants.%NSC]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %nsc: ref %NSC = ref_binding nsc, %nsc.var [concrete = %nsc.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.ad3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.a60
// CHECK:STDOUT:   .x = imports.%Main.import_ref.e6b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC [from "export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.63c
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.8ad
// CHECK:STDOUT:   .y = imports.%Main.import_ref.be6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc23.1: %struct_type.x = struct_literal (%.loc22.1) [concrete = constants.%struct.948]
// CHECK:STDOUT:   %.loc23.2: ref %empty_tuple.type = class_element_access file.%c.var, element0 [concrete = constants.%.1d2]
// CHECK:STDOUT:   %.loc22.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc23.3: init %empty_tuple.type = converted %.loc22.1, %.loc22.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc23.4: init %C to file.%c.var = class_init (%.loc23.3) [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc12: init %C = converted %.loc23.1, %.loc23.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc12
// CHECK:STDOUT:   %.loc37.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc38.1: %struct_type.y = struct_literal (%.loc37.1) [concrete = constants.%struct.8ce]
// CHECK:STDOUT:   %.loc38.2: ref %empty_tuple.type = class_element_access file.%nsc.var, element0 [concrete = constants.%.160]
// CHECK:STDOUT:   %.loc37.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc38.3: init %empty_tuple.type = converted %.loc37.1, %.loc37.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc38.4: init %NSC to file.%nsc.var = class_init (%.loc38.3) [concrete = constants.%NSC.val]
// CHECK:STDOUT:   %.loc25: init %NSC = converted %.loc38.1, %.loc38.4 [concrete = constants.%NSC.val]
// CHECK:STDOUT:   assign file.%nsc.var, %.loc25
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
// CHECK:STDOUT:   %Local: <error> = alias_binding Local, <error> [concrete = <error>]
// CHECK:STDOUT:   %NS.ref: <error> = name_ref NS, <error> [concrete = <error>]
// CHECK:STDOUT:   %NSC.ref: <error> = name_ref NSC, <error> [concrete = <error>]
// CHECK:STDOUT:   %NSLocal: <error> = alias_binding NSLocal, <error> [concrete = <error>]
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
// CHECK:STDOUT:   %Main.NSC = import_ref Main//base, NSC, unloaded
// CHECK:STDOUT:   %Main.import_ref.56d: <witness> = import_ref Main//base, loc13, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.743 = import_ref Main//base, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.fec = import_ref Main//base, loc9, unloaded
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
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.743
// CHECK:STDOUT:   .x = imports.%Main.import_ref.fec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_repeat_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.x = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %.1d2: ref %empty_tuple.type = class_element_access file.%c.var, element0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//repeat_export, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.ad3: <witness> = import_ref Main//repeat_export, inst{{[0-9A-F]+}} [indirect], loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.a60 = import_ref Main//repeat_export, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Main.import_ref.e6b = import_ref Main//repeat_export, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt [concrete]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "repeat_export.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.ad3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.a60
// CHECK:STDOUT:   .x = imports.%Main.import_ref.e6b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19.1: %struct_type.x = struct_literal (%.loc18.1) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc19.2: ref %empty_tuple.type = class_element_access file.%c.var, element0 [concrete = constants.%.1d2]
// CHECK:STDOUT:   %.loc18.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19.3: init %empty_tuple.type = converted %.loc18.1, %.loc18.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19.4: init %C to file.%c.var = class_init (%.loc19.3) [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8: init %C = converted %.loc19.1, %.loc19.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc8
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
// CHECK:STDOUT:   %Main.NSC = import_ref Main//base, NSC, unloaded
// CHECK:STDOUT:   %Main.import_ref.56d: <witness> = import_ref Main//base, loc13, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.743 = import_ref Main//base, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.fec = import_ref Main//base, loc9, unloaded
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
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.743
// CHECK:STDOUT:   .x = imports.%Main.import_ref.fec
// CHECK:STDOUT: }
// CHECK:STDOUT:
