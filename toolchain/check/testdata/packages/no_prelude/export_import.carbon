// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/no_prelude/export_import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/no_prelude/export_import.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- base.carbon

library "[[@TEST_NAME]]";

class C {
  var x: ();
};

// --- export.carbon

library "[[@TEST_NAME]]";

export import library "base";

// --- export_copy.carbon

library "[[@TEST_NAME]]";

export import library "base";

// --- non_export.carbon

library "[[@TEST_NAME]]";

import library "export";

// --- export_export.carbon

library "[[@TEST_NAME]]";

export import library "export";

// --- import_then_export.carbon

library "[[@TEST_NAME]]";

import library "base";
export import library "export";

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

// --- fail_export_in_impl.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_export_in_impl.impl.carbon:[[@LINE+4]]:1: error: `export` is only allowed in API files [ExportFromImpl]
// CHECK:STDERR: export import library "base";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
export import library "base";

// Note the import still occurs.
var c: C = {.x = ()};

// --- export_export.impl.carbon

impl library "[[@TEST_NAME]]";

var c: C = {.x = ()};

// --- use_export_export.carbon

library "[[@TEST_NAME]]";

import library "export_export";

var c: C = {.x = ()};

// --- use_import_then_export.carbon

library "[[@TEST_NAME]]";

import library "import_then_export";

var c: C = {.x = ()};

// --- use_import_when_export.carbon

library "[[@TEST_NAME]]";

export import library "base";

var c: C = {.x = ()};

// --- use_base_and_export.carbon

library "[[@TEST_NAME]]";

import library "base";
import library "export";

var c: C = {.x = ()};

// --- use_export_and_base.carbon

library "[[@TEST_NAME]]";

import library "export";
import library "base";

var c: C = {.x = ()};

// --- use_export_copy.carbon

library "[[@TEST_NAME]]";

import library "export";
import library "export_copy";

var c: C = {.x = ()};

// --- fail_use_non_export.carbon

library "[[@TEST_NAME]]";

import library "non_export";

// CHECK:STDERR: fail_use_non_export.carbon:[[@LINE+3]]:15: error: name `C` not found [NameNotFound]
// CHECK:STDERR: alias Local = C;
// CHECK:STDERR:               ^
alias Local = C;

// --- use_non_export_then_base.carbon

library "[[@TEST_NAME]]";

import library "non_export";
export import library "base";

var c: C = {.x = ()};

// --- indirect_use_non_export_then_base.carbon

library "[[@TEST_NAME]]";

import library "use_non_export_then_base";

var indirect_c: C = {.x = ()};

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
// CHECK:STDOUT: --- export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref = import_ref Main//base, inst+1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_copy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref = import_ref Main//base, inst+1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref = import_ref Main//base, inst+1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref = import_ref Main//base, inst+1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_then_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref = import_ref Main//base, inst+1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export.carbon
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
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//base, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Main//base, inst+9, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//base, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//base, inst+7, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
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
// CHECK:STDOUT:   %.loc6_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_20.1: %struct_type.x = struct_literal (%.loc6_19.1)
// CHECK:STDOUT:   %.loc6_20.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_19.2: init %empty_tuple.type = tuple_init () to %.loc6_20.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_20.3: init %empty_tuple.type = converted %.loc6_19.1, %.loc6_19.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_20.4: init %C = class_init (%.loc6_20.3), file.%c.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc6_21: init %C = converted %.loc6_20.1, %.loc6_20.4 [template = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_export_in_impl.impl.carbon
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
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//base, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Main//base, inst+9, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//base, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//base, inst+7, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
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
// CHECK:STDOUT:   %.loc11_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc11_20.1: %struct_type.x = struct_literal (%.loc11_19.1)
// CHECK:STDOUT:   %.loc11_20.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc11_19.2: init %empty_tuple.type = tuple_init () to %.loc11_20.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_20.3: init %empty_tuple.type = converted %.loc11_19.1, %.loc11_19.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_20.4: init %C = class_init (%.loc11_20.3), file.%c.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc11_21: init %C = converted %.loc11_20.1, %.loc11_20.4 [template = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc11_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_export.impl.carbon
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
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//export_export, inst+2, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Main//base, inst+9, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//base, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//base, inst+7, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
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
// CHECK:STDOUT:   %.loc4_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_20.1: %struct_type.x = struct_literal (%.loc4_19.1)
// CHECK:STDOUT:   %.loc4_20.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc4_19.2: init %empty_tuple.type = tuple_init () to %.loc4_20.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_20.3: init %empty_tuple.type = converted %.loc4_19.1, %.loc4_19.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_20.4: init %C = class_init (%.loc4_20.3), file.%c.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc4_21: init %C = converted %.loc4_20.1, %.loc4_20.4 [template = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc4_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_export.carbon
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
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//base, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Main//base, inst+9, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//base, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//base, inst+7, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
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
// CHECK:STDOUT:   %.loc6_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_20.1: %struct_type.x = struct_literal (%.loc6_19.1)
// CHECK:STDOUT:   %.loc6_20.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_19.2: init %empty_tuple.type = tuple_init () to %.loc6_20.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_20.3: init %empty_tuple.type = converted %.loc6_19.1, %.loc6_19.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_20.4: init %C = class_init (%.loc6_20.3), file.%c.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc6_21: init %C = converted %.loc6_20.1, %.loc6_20.4 [template = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_import_then_export.carbon
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
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//base, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Main//base, inst+9, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//base, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//base, inst+7, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
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
// CHECK:STDOUT:   %.loc6_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_20.1: %struct_type.x = struct_literal (%.loc6_19.1)
// CHECK:STDOUT:   %.loc6_20.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_19.2: init %empty_tuple.type = tuple_init () to %.loc6_20.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_20.3: init %empty_tuple.type = converted %.loc6_19.1, %.loc6_19.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_20.4: init %C = class_init (%.loc6_20.3), file.%c.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc6_21: init %C = converted %.loc6_20.1, %.loc6_20.4 [template = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_import_when_export.carbon
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
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//base, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Main//base, inst+9, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//base, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//base, inst+7, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
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
// CHECK:STDOUT:   %.loc6_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_20.1: %struct_type.x = struct_literal (%.loc6_19.1)
// CHECK:STDOUT:   %.loc6_20.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_19.2: init %empty_tuple.type = tuple_init () to %.loc6_20.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_20.3: init %empty_tuple.type = converted %.loc6_19.1, %.loc6_19.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_20.4: init %C = class_init (%.loc6_20.3), file.%c.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc6_21: init %C = converted %.loc6_20.1, %.loc6_20.4 [template = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_base_and_export.carbon
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
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//base, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Main//base, inst+9, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//base, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//base, inst+7, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
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
// CHECK:STDOUT:   %.loc7_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc7_20.1: %struct_type.x = struct_literal (%.loc7_19.1)
// CHECK:STDOUT:   %.loc7_20.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc7_19.2: init %empty_tuple.type = tuple_init () to %.loc7_20.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_20.3: init %empty_tuple.type = converted %.loc7_19.1, %.loc7_19.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_20.4: init %C = class_init (%.loc7_20.3), file.%c.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc7_21: init %C = converted %.loc7_20.1, %.loc7_20.4 [template = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc7_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_and_base.carbon
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
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//base, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Main//base, inst+9, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//base, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//base, inst+7, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
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
// CHECK:STDOUT:   %.loc7_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc7_20.1: %struct_type.x = struct_literal (%.loc7_19.1)
// CHECK:STDOUT:   %.loc7_20.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc7_19.2: init %empty_tuple.type = tuple_init () to %.loc7_20.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_20.3: init %empty_tuple.type = converted %.loc7_19.1, %.loc7_19.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_20.4: init %C = class_init (%.loc7_20.3), file.%c.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc7_21: init %C = converted %.loc7_20.1, %.loc7_20.4 [template = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc7_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_copy.carbon
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
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//base, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Main//base, inst+9, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//base, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//base, inst+7, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
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
// CHECK:STDOUT:   %.loc7_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc7_20.1: %struct_type.x = struct_literal (%.loc7_19.1)
// CHECK:STDOUT:   %.loc7_20.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc7_19.2: init %empty_tuple.type = tuple_init () to %.loc7_20.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_20.3: init %empty_tuple.type = converted %.loc7_19.1, %.loc7_19.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_20.4: init %C = class_init (%.loc7_20.3), file.%c.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc7_21: init %C = converted %.loc7_20.1, %.loc7_20.4 [template = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc7_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_non_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Local = %Local
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C.ref: <error> = name_ref C, <error> [template = <error>]
// CHECK:STDOUT:   %Local: <error> = bind_alias Local, <error> [template = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_non_export_then_base.carbon
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
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//base, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Main//base, inst+9, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//base, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//base, inst+7, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
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
// CHECK:STDOUT:   %.loc7_19.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc7_20.1: %struct_type.x = struct_literal (%.loc7_19.1)
// CHECK:STDOUT:   %.loc7_20.2: ref %empty_tuple.type = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc7_19.2: init %empty_tuple.type = tuple_init () to %.loc7_20.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_20.3: init %empty_tuple.type = converted %.loc7_19.1, %.loc7_19.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_20.4: init %C = class_init (%.loc7_20.3), file.%c.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc7_21: init %C = converted %.loc7_20.1, %.loc7_20.4 [template = constants.%C.val]
// CHECK:STDOUT:   assign file.%c.var, %.loc7_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- indirect_use_non_export_then_base.carbon
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
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//use_non_export_then_base, inst+14, unloaded
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//base, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.3: <witness> = import_ref Main//base, inst+9, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//base, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//base, inst+7, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .c = imports.%import_ref.1
// CHECK:STDOUT:     .C = imports.%import_ref.2
// CHECK:STDOUT:     .indirect_c = %indirect_c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.2 [template = constants.%C]
// CHECK:STDOUT:   %indirect_c.var: ref %C = var indirect_c
// CHECK:STDOUT:   %indirect_c: ref %C = bind_name indirect_c, %indirect_c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT:   .x = imports.%import_ref.5
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_28.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_29.1: %struct_type.x = struct_literal (%.loc6_28.1)
// CHECK:STDOUT:   %.loc6_29.2: ref %empty_tuple.type = class_element_access file.%indirect_c.var, element0
// CHECK:STDOUT:   %.loc6_28.2: init %empty_tuple.type = tuple_init () to %.loc6_29.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_29.3: init %empty_tuple.type = converted %.loc6_28.1, %.loc6_28.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_29.4: init %C = class_init (%.loc6_29.3), file.%indirect_c.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc6_30: init %C = converted %.loc6_29.1, %.loc6_29.4 [template = constants.%C.val]
// CHECK:STDOUT:   assign file.%indirect_c.var, %.loc6_30
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
