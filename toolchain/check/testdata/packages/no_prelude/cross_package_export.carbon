// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/no_prelude/cross_package_export.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/no_prelude/cross_package_export.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- base.carbon

package Other library "[[@TEST_NAME]]";

class C {
  var x: ();
};

// --- conflict.carbon

package Other library "[[@TEST_NAME]]";

fn C() {}

// --- export_import.carbon

package Other library "[[@TEST_NAME]]";

export import library "base";

// --- export_import_copy.carbon

package Other library "[[@TEST_NAME]]";

export import library "base";

// --- export_import_indirect.carbon

package Other library "[[@TEST_NAME]]";

export import library "export_import";

// --- export_name.carbon

package Other library "[[@TEST_NAME]]";

import library "base";

export C;

// --- export_name_copy.carbon

package Other library "[[@TEST_NAME]]";

import library "base";

export C;

// --- export_name_indirect.carbon

package Other library "[[@TEST_NAME]]";

import library "export_name";

export C;

// ============================================================================
// Test files
// ============================================================================

// --- use_export_import.carbon

library "[[@TEST_NAME]]";

import Other library "export_import";

var c: Other.C = {.x = ()};

// --- use_export_import_with_copy.carbon

library "[[@TEST_NAME]]";

import Other library "export_import";
import Other library "export_import_copy";

var c: Other.C = {.x = ()};

// --- use_export_import_indirect.carbon

library "[[@TEST_NAME]]";

import Other library "export_import_indirect";

var c: Other.C = {.x = ()};

// --- use_export_name.carbon

library "[[@TEST_NAME]]";

import Other library "export_name";

var c: Other.C = {.x = ()};

// --- use_export_name_with_copy.carbon

library "[[@TEST_NAME]]";

import Other library "export_name";
import Other library "export_name_copy";

var c: Other.C = {.x = ()};

// --- use_export_name_indirect.carbon

library "[[@TEST_NAME]]";

import Other library "export_name_indirect";

var c: Other.C = {.x = ()};

// --- use_export_all.carbon

library "[[@TEST_NAME]]";

import Other library "export_import";
import Other library "export_name";
import Other library "export_import_copy";
import Other library "export_name_copy";
import Other library "export_import_indirect";
import Other library "export_name_indirect";

var c: Other.C = {.x = ()};

// --- unused_conflict_on_export_import.carbon

library "[[@TEST_NAME]]";

import Other library "export_import";
import Other library "conflict";

// --- unused_conflict_on_export_name.carbon

library "[[@TEST_NAME]]";

import Other library "export_name";
import Other library "conflict";

// --- fail_conflict_on_export_import.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_conflict_on_export_import.carbon:[[@LINE+8]]:1: in import [InImport]
// CHECK:STDERR: base.carbon:4:1: error: duplicate name `C` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: class C {
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_conflict_on_export_import.carbon:[[@LINE+4]]:1: in import [InImport]
// CHECK:STDERR: conflict.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn C() {}
// CHECK:STDERR: ^~~~~~~~
import Other library "export_import";
import Other library "conflict";

// CHECK:STDERR: fail_conflict_on_export_import.carbon:[[@LINE+4]]:11: note: in name lookup for `C` [InNameLookup]
// CHECK:STDERR: alias C = Other.C;
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR:
alias C = Other.C;

// --- fail_conflict_on_export_name.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_conflict_on_export_name.carbon:[[@LINE+9]]:1: in import [InImport]
// CHECK:STDERR: conflict.carbon:4:1: error: duplicate name `C` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn C() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_conflict_on_export_name.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: export_name.carbon:4:1: in import [InImport]
// CHECK:STDERR: base.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: class C {
// CHECK:STDERR: ^~~~~~~~~
import Other library "export_name";
import Other library "conflict";

// CHECK:STDERR: fail_conflict_on_export_name.carbon:[[@LINE+4]]:11: note: in name lookup for `C` [InNameLookup]
// CHECK:STDERR: alias C = Other.C;
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR:
alias C = Other.C;

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc5_8: %C.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc5_3: %C.elem = var_pattern %.loc5_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %C.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .x = %.loc5_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = fn_type @C [concrete]
// CHECK:STDOUT:   %C: %C.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other.C = import_ref Other//base, C, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Other.C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_import_copy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other.C = import_ref Other//base, C, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Other.C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_import_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other.C = import_ref Other//base, C, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Other.C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other.C: type = import_ref Other//base, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Other.import_ref.56d: <witness> = import_ref Other//base, loc6_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Other.import_ref.2c4 = import_ref Other//base, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Other.import_ref.276 = import_ref Other//base, loc5_8, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C: type = export C, imports.%Other.C [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Other.import_ref.56d
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Other.import_ref.2c4
// CHECK:STDOUT:   .x = imports.%Other.import_ref.276
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_name_copy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other.C: type = import_ref Other//base, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Other.import_ref.56d: <witness> = import_ref Other//base, loc6_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Other.import_ref.2c4 = import_ref Other//base, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Other.import_ref.276 = import_ref Other//base, loc5_8, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C: type = export C, imports.%Other.C [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Other.import_ref.56d
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Other.import_ref.2c4
// CHECK:STDOUT:   .x = imports.%Other.import_ref.276
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_name_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other.C: type = import_ref Other//export_name, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Other.import_ref.ad3: <witness> = import_ref Other//export_name, inst22 [indirect], loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Other.import_ref.6a9 = import_ref Other//export_name, inst23 [indirect], unloaded
// CHECK:STDOUT:   %Other.import_ref.f67 = import_ref Other//export_name, inst24 [indirect], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C: type = export C, imports.%Other.C [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "export_name.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Other.import_ref.ad3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Other.import_ref.6a9
// CHECK:STDOUT:   .x = imports.%Other.import_ref.f67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_import.carbon
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
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .C = %Other.C
// CHECK:STDOUT:     import Other//export_import
// CHECK:STDOUT:     import Other//base
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.C: type = import_ref Other//base, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Other.import_ref.56d: <witness> = import_ref Other//base, loc6_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Other.import_ref.2c4 = import_ref Other//base, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Other.import_ref.276 = import_ref Other//base, loc5_8, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %.loc6_1: %C = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %.loc6_13: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:     %Other.ref: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Other.C [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Other.import_ref.56d
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Other.import_ref.2c4
// CHECK:STDOUT:   .x = imports.%Other.import_ref.276
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_26.1: %struct_type.x = struct_literal (%.loc6_25.1)
// CHECK:STDOUT:   %.loc6_26.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_25.2: init %empty_tuple.type = tuple_init () to %.loc6_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_26.3: init %empty_tuple.type = converted %.loc6_25.1, %.loc6_25.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_26.4: init %C = class_init (%.loc6_26.3), file.%c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc6_1: init %C = converted %.loc6_26.1, %.loc6_26.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_import_with_copy.carbon
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
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .C = %Other.C
// CHECK:STDOUT:     import Other//export_import
// CHECK:STDOUT:     import Other//export_import_copy
// CHECK:STDOUT:     import Other//base
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.C: type = import_ref Other//base, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Other.import_ref.56d: <witness> = import_ref Other//base, loc6_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Other.import_ref.2c4 = import_ref Other//base, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Other.import_ref.276 = import_ref Other//base, loc5_8, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %.loc7_1: %C = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %.loc7_13: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:     %Other.ref: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Other.C [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Other.import_ref.56d
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Other.import_ref.2c4
// CHECK:STDOUT:   .x = imports.%Other.import_ref.276
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc7_26.1: %struct_type.x = struct_literal (%.loc7_25.1)
// CHECK:STDOUT:   %.loc7_26.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc7_25.2: init %empty_tuple.type = tuple_init () to %.loc7_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_26.3: init %empty_tuple.type = converted %.loc7_25.1, %.loc7_25.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_26.4: init %C = class_init (%.loc7_26.3), file.%c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc7_1: init %C = converted %.loc7_26.1, %.loc7_26.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc7_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_import_indirect.carbon
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
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .C = %Other.C
// CHECK:STDOUT:     import Other//export_import_indirect
// CHECK:STDOUT:     import Other//export_import
// CHECK:STDOUT:     import Other//base
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.C: type = import_ref Other//base, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Other.import_ref.56d: <witness> = import_ref Other//base, loc6_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Other.import_ref.2c4 = import_ref Other//base, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Other.import_ref.276 = import_ref Other//base, loc5_8, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %.loc6_1: %C = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %.loc6_13: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:     %Other.ref: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Other.C [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Other.import_ref.56d
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Other.import_ref.2c4
// CHECK:STDOUT:   .x = imports.%Other.import_ref.276
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_26.1: %struct_type.x = struct_literal (%.loc6_25.1)
// CHECK:STDOUT:   %.loc6_26.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_25.2: init %empty_tuple.type = tuple_init () to %.loc6_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_26.3: init %empty_tuple.type = converted %.loc6_25.1, %.loc6_25.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_26.4: init %C = class_init (%.loc6_26.3), file.%c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc6_1: init %C = converted %.loc6_26.1, %.loc6_26.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_name.carbon
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
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .C = %Other.C
// CHECK:STDOUT:     import Other//export_name
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.C: type = import_ref Other//export_name, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Other.import_ref.ad3: <witness> = import_ref Other//export_name, inst22 [indirect], loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Other.import_ref.6a9 = import_ref Other//export_name, inst23 [indirect], unloaded
// CHECK:STDOUT:   %Other.import_ref.f67 = import_ref Other//export_name, inst24 [indirect], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %.loc6_1: %C = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %.loc6_13: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:     %Other.ref: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Other.C [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "export_name.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Other.import_ref.ad3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Other.import_ref.6a9
// CHECK:STDOUT:   .x = imports.%Other.import_ref.f67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_26.1: %struct_type.x = struct_literal (%.loc6_25.1)
// CHECK:STDOUT:   %.loc6_26.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_25.2: init %empty_tuple.type = tuple_init () to %.loc6_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_26.3: init %empty_tuple.type = converted %.loc6_25.1, %.loc6_25.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_26.4: init %C = class_init (%.loc6_26.3), file.%c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc6_1: init %C = converted %.loc6_26.1, %.loc6_26.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_name_with_copy.carbon
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
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .C = %Other.C
// CHECK:STDOUT:     import Other//export_name
// CHECK:STDOUT:     import Other//export_name_copy
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.C: type = import_ref Other//export_name, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Other.import_ref.ad3: <witness> = import_ref Other//export_name, inst22 [indirect], loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Other.import_ref.6a9 = import_ref Other//export_name, inst23 [indirect], unloaded
// CHECK:STDOUT:   %Other.import_ref.f67 = import_ref Other//export_name, inst24 [indirect], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %.loc7_1: %C = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %.loc7_13: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:     %Other.ref: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Other.C [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "export_name.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Other.import_ref.ad3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Other.import_ref.6a9
// CHECK:STDOUT:   .x = imports.%Other.import_ref.f67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc7_26.1: %struct_type.x = struct_literal (%.loc7_25.1)
// CHECK:STDOUT:   %.loc7_26.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc7_25.2: init %empty_tuple.type = tuple_init () to %.loc7_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_26.3: init %empty_tuple.type = converted %.loc7_25.1, %.loc7_25.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_26.4: init %C = class_init (%.loc7_26.3), file.%c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc7_1: init %C = converted %.loc7_26.1, %.loc7_26.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc7_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_name_indirect.carbon
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
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .C = %Other.C
// CHECK:STDOUT:     import Other//export_name_indirect
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.C: type = import_ref Other//export_name_indirect, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Other.import_ref.328: <witness> = import_ref Other//export_name_indirect, inst22 [indirect], loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Other.import_ref.db8 = import_ref Other//export_name_indirect, inst23 [indirect], unloaded
// CHECK:STDOUT:   %Other.import_ref.3ef = import_ref Other//export_name_indirect, inst24 [indirect], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %.loc6_1: %C = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %.loc6_13: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:     %Other.ref: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Other.C [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "export_name_indirect.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Other.import_ref.328
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Other.import_ref.db8
// CHECK:STDOUT:   .x = imports.%Other.import_ref.3ef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_26.1: %struct_type.x = struct_literal (%.loc6_25.1)
// CHECK:STDOUT:   %.loc6_26.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_25.2: init %empty_tuple.type = tuple_init () to %.loc6_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_26.3: init %empty_tuple.type = converted %.loc6_25.1, %.loc6_25.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_26.4: init %C = class_init (%.loc6_26.3), file.%c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc6_1: init %C = converted %.loc6_26.1, %.loc6_26.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_all.carbon
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
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .C = %Other.C
// CHECK:STDOUT:     import Other//export_import
// CHECK:STDOUT:     import Other//export_name
// CHECK:STDOUT:     import Other//export_import_copy
// CHECK:STDOUT:     import Other//export_name_copy
// CHECK:STDOUT:     import Other//export_import_indirect
// CHECK:STDOUT:     import Other//export_name_indirect
// CHECK:STDOUT:     import Other//base
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.C: type = import_ref Other//export_name, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Other.import_ref.ad3: <witness> = import_ref Other//export_name, inst22 [indirect], loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Other.import_ref.6a9 = import_ref Other//export_name, inst23 [indirect], unloaded
// CHECK:STDOUT:   %Other.import_ref.f67 = import_ref Other//export_name, inst24 [indirect], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %.loc11_1: %C = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %.loc11_13: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:     %Other.ref: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Other.C [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "export_name.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Other.import_ref.ad3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Other.import_ref.6a9
// CHECK:STDOUT:   .x = imports.%Other.import_ref.f67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc11_26.1: %struct_type.x = struct_literal (%.loc11_25.1)
// CHECK:STDOUT:   %.loc11_26.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc11_25.2: init %empty_tuple.type = tuple_init () to %.loc11_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_26.3: init %empty_tuple.type = converted %.loc11_25.1, %.loc11_25.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_26.4: init %C = class_init (%.loc11_26.3), file.%c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc11_1: init %C = converted %.loc11_26.1, %.loc11_26.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc11_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unused_conflict_on_export_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     import Other//export_import
// CHECK:STDOUT:     import Other//conflict
// CHECK:STDOUT:     import Other//base
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unused_conflict_on_export_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     import Other//export_name
// CHECK:STDOUT:     import Other//conflict
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conflict_on_export_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = fn_type @C [concrete]
// CHECK:STDOUT:   %C: %C.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .C = %Other.C
// CHECK:STDOUT:     import Other//export_import
// CHECK:STDOUT:     import Other//conflict
// CHECK:STDOUT:     import Other//base
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.C: %C.type = import_ref Other//conflict, C, loaded [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%Other.C [concrete = constants.%C]
// CHECK:STDOUT:   %C: %C.type = bind_alias C, imports.%Other.C [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C() [from "conflict.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conflict_on_export_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [concrete] {
// CHECK:STDOUT:     .C = %Other.C
// CHECK:STDOUT:     import Other//export_name
// CHECK:STDOUT:     import Other//conflict
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.C: type = import_ref Other//export_name, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Other.import_ref.ad3: <witness> = import_ref Other//export_name, inst22 [indirect], loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Other.import_ref.6a9 = import_ref Other//export_name, inst23 [indirect], unloaded
// CHECK:STDOUT:   %Other.import_ref.f67 = import_ref Other//export_name, inst24 [indirect], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [concrete = imports.%Other]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Other.C [concrete = constants.%C]
// CHECK:STDOUT:   %C: type = bind_alias C, imports.%Other.C [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "export_name.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Other.import_ref.ad3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Other.import_ref.6a9
// CHECK:STDOUT:   .x = imports.%Other.import_ref.f67
// CHECK:STDOUT: }
// CHECK:STDOUT:
