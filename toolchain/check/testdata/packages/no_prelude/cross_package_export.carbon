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
// CHECK:STDERR: base.carbon:4:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
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
// CHECK:STDERR: conflict.carbon:4:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn C() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_conflict_on_export_name.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: export_name.carbon:4:1: in import [InImport]
// CHECK:STDERR: base.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: class C {
// CHECK:STDERR: ^~~~~~~~~
import Other library "export_name";
import Other library "conflict";

// CHECK:STDERR: fail_conflict_on_export_name.carbon:[[@LINE+3]]:11: note: in name lookup for `C` [InNameLookup]
// CHECK:STDERR: alias C = Other.C;
// CHECK:STDERR:           ^~~~~~~
alias C = Other.C;

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %empty_tuple.type [template]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc5_11.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc5_11.2: type = converted %.loc5_11.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc5_8: %C.elem = field_decl x, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .x = %.loc5_8
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [template = constants.%C] {} {}
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
// CHECK:STDOUT:   %import_ref = import_ref Other//base, inst+1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_import_copy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref = import_ref Other//base, inst+1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_import_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref = import_ref Other//base, inst+1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Other//base, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Other//base, inst+9, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Other//base, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Other//base, inst+7, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C: type = export C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .x = imports.%import_ref.4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_name_copy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Other//base, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Other//base, inst+9, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Other//base, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Other//base, inst+7, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C: type = export C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .x = imports.%import_ref.4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_name_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Other//export_name, inst+11, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Other//export_name, inst+8, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Other//export_name, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Other//export_name, inst+10, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C: type = export C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .x = imports.%import_ref.4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     import Other//export_import
// CHECK:STDOUT:     import Other//base
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Other//base, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Other//base, inst+9, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Other//base, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Other//base, inst+7, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .x = imports.%import_ref.4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_26.1: %struct_type.x = struct_literal (%.loc6_25.1)
// CHECK:STDOUT:   %.loc6_26.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_25.2: init %empty_tuple.type = tuple_init () to %.loc6_26.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_26.3: init %empty_tuple.type = converted %.loc6_25.1, %.loc6_25.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_26.4: init %C = class_init (%.loc6_26.3), file.%c.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc6_27: init %C = converted %.loc6_26.1, %.loc6_26.4 [template = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_import_with_copy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     import Other//export_import
// CHECK:STDOUT:     import Other//export_import_copy
// CHECK:STDOUT:     import Other//base
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Other//base, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Other//base, inst+9, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Other//base, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Other//base, inst+7, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .x = imports.%import_ref.4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc7_26.1: %struct_type.x = struct_literal (%.loc7_25.1)
// CHECK:STDOUT:   %.loc7_26.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc7_25.2: init %empty_tuple.type = tuple_init () to %.loc7_26.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_26.3: init %empty_tuple.type = converted %.loc7_25.1, %.loc7_25.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_26.4: init %C = class_init (%.loc7_26.3), file.%c.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc7_27: init %C = converted %.loc7_26.1, %.loc7_26.4 [template = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc7_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_import_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     import Other//export_import_indirect
// CHECK:STDOUT:     import Other//export_import
// CHECK:STDOUT:     import Other//base
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Other//base, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Other//base, inst+9, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Other//base, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Other//base, inst+7, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .x = imports.%import_ref.4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_26.1: %struct_type.x = struct_literal (%.loc6_25.1)
// CHECK:STDOUT:   %.loc6_26.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_25.2: init %empty_tuple.type = tuple_init () to %.loc6_26.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_26.3: init %empty_tuple.type = converted %.loc6_25.1, %.loc6_25.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_26.4: init %C = class_init (%.loc6_26.3), file.%c.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc6_27: init %C = converted %.loc6_26.1, %.loc6_26.4 [template = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     import Other//export_name
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Other//export_name, inst+11, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Other//export_name, inst+8, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Other//export_name, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Other//export_name, inst+10, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .x = imports.%import_ref.4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_26.1: %struct_type.x = struct_literal (%.loc6_25.1)
// CHECK:STDOUT:   %.loc6_26.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_25.2: init %empty_tuple.type = tuple_init () to %.loc6_26.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_26.3: init %empty_tuple.type = converted %.loc6_25.1, %.loc6_25.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_26.4: init %C = class_init (%.loc6_26.3), file.%c.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc6_27: init %C = converted %.loc6_26.1, %.loc6_26.4 [template = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_name_with_copy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     import Other//export_name
// CHECK:STDOUT:     import Other//export_name_copy
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Other//export_name, inst+11, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Other//export_name, inst+8, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Other//export_name, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Other//export_name, inst+10, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .x = imports.%import_ref.4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc7_26.1: %struct_type.x = struct_literal (%.loc7_25.1)
// CHECK:STDOUT:   %.loc7_26.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc7_25.2: init %empty_tuple.type = tuple_init () to %.loc7_26.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_26.3: init %empty_tuple.type = converted %.loc7_25.1, %.loc7_25.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_26.4: init %C = class_init (%.loc7_26.3), file.%c.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc7_27: init %C = converted %.loc7_26.1, %.loc7_26.4 [template = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc7_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_name_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     import Other//export_name_indirect
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Other//export_name_indirect, inst+11, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Other//export_name_indirect, inst+8, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Other//export_name_indirect, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Other//export_name_indirect, inst+10, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .x = imports.%import_ref.4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_26.1: %struct_type.x = struct_literal (%.loc6_25.1)
// CHECK:STDOUT:   %.loc6_26.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_25.2: init %empty_tuple.type = tuple_init () to %.loc6_26.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_26.3: init %empty_tuple.type = converted %.loc6_25.1, %.loc6_25.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_26.4: init %C = class_init (%.loc6_26.3), file.%c.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc6_27: init %C = converted %.loc6_26.1, %.loc6_26.4 [template = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_all.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %C.val: %C = struct_value (%empty_tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     import Other//export_import
// CHECK:STDOUT:     import Other//export_name
// CHECK:STDOUT:     import Other//export_import_copy
// CHECK:STDOUT:     import Other//export_name_copy
// CHECK:STDOUT:     import Other//export_import_indirect
// CHECK:STDOUT:     import Other//export_name_indirect
// CHECK:STDOUT:     import Other//base
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Other//export_name, inst+11, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Other//export_name, inst+8, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Other//export_name, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Other//export_name, inst+10, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .x = imports.%import_ref.4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc11_26.1: %struct_type.x = struct_literal (%.loc11_25.1)
// CHECK:STDOUT:   %.loc11_26.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc11_25.2: init %empty_tuple.type = tuple_init () to %.loc11_26.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_26.3: init %empty_tuple.type = converted %.loc11_25.1, %.loc11_25.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_26.4: init %C = class_init (%.loc11_26.3), file.%c.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc11_27: init %C = converted %.loc11_26.1, %.loc11_26.4 [template = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc11_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unused_conflict_on_export_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     import Other//export_import
// CHECK:STDOUT:     import Other//conflict
// CHECK:STDOUT:     import Other//base
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unused_conflict_on_export_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     import Other//export_name
// CHECK:STDOUT:     import Other//conflict
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conflict_on_export_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     import Other//export_import
// CHECK:STDOUT:     import Other//conflict
// CHECK:STDOUT:     import Other//base
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %C.type = import_ref Other//conflict, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %C: %C.type = bind_alias C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conflict_on_export_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Other: <namespace> = namespace file.%Other.import, [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     import Other//export_name
// CHECK:STDOUT:     import Other//conflict
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Other//export_name, inst+11, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Other//export_name, inst+8, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Other//export_name, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Other//export_name, inst+10, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = imports.%Other
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, imports.%Other [template = imports.%Other]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %C: type = bind_alias C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .x = imports.%import_ref.4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
