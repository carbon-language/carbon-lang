// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// ============================================================================
// Setup files
// ============================================================================

// --- base.carbon

library "base" api;

class C {
  var x: ();
};

// --- export.carbon

library "export" api;

export import library "base";

// --- export_copy.carbon

library "export_copy" api;

export import library "base";

// --- non_export.carbon

library "non_export" api;

import library "export";

// --- export_export.carbon

library "export_export" api;

export import library "export";

// --- import_then_export.carbon

library "import_then_export" api;

import library "base";
export import library "export";

// --- export_in_impl.carbon

// This is just providing an API for the implicit import.
library "export_in_impl" api;

// ============================================================================
// Test files
// ============================================================================

// --- use_export.carbon

library "use_export" api;

import library "export";

var c: C = {.x = ()};

// --- fail_export_in_impl.impl.carbon

library "export_in_impl" impl;

// CHECK:STDERR: fail_export_in_impl.impl.carbon:[[@LINE+4]]:1: ERROR: `export` is only allowed in API files.
// CHECK:STDERR: export import library "base";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
export import library "base";

// Note the import still occurs.
var c: C = {.x = ()};

// --- export_export.impl.carbon

library "export_export" impl;

var c: C = {.x = ()};

// --- use_export_export.carbon

library "use_export_export" api;

import library "export_export";

var c: C = {.x = ()};

// --- use_import_then_export.carbon

library "use_import_then_export" api;

import library "import_then_export";

var c: C = {.x = ()};

// --- use_import_when_export.carbon

library "use_import_when_export" api;

export import library "base";

var c: C = {.x = ()};

// --- use_base_and_export.carbon

library "use_base_and_export" api;

import library "base";
import library "export";

var c: C = {.x = ()};

// --- use_export_and_base.carbon

library "import_both_reversed" api;

import library "export";
import library "base";

var c: C = {.x = ()};

// --- use_export_copy.carbon

library "use_export_copy" api;

import library "export";
import library "export_copy";

var c: C = {.x = ()};

// --- fail_use_non_export.carbon

library "fail_use_non_export" api;

import library "non_export";

// CHECK:STDERR: fail_use_non_export.carbon:[[@LINE+3]]:15: ERROR: Name `C` not found.
// CHECK:STDERR: alias Local = C;
// CHECK:STDERR:               ^
alias Local = C;

// --- use_non_export_then_base.carbon

library "use_non_export_then_base" api;

import library "non_export";
export import library "base";

var c: C = {.x = ()};

// --- indirect_use_non_export_then_base.carbon

library "indirect_use_non_export_then_base" api;

import library "use_non_export_then_base";

var indirect_c: C = {.x = ()};

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type C, () [template]
// CHECK:STDOUT:   %.3: type = struct_type {.x: ()} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc5_11.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc5_11.2: type = converted %.loc5_11.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc5_8: <unbound element of class C> = field_decl x, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .x = %.loc5_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref = import_ref ir1, inst+1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_copy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref = import_ref ir1, inst+1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref = import_ref ir2, inst+1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref = import_ref ir2, inst+1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_then_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref = import_ref ir1, inst+1, unloaded
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
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.x: ()} [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %struct: C = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir2, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir2, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .x = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_19.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_20.1: {.x: ()} = struct_literal (%.loc6_19.1)
// CHECK:STDOUT:   %.loc6_20.2: ref () = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_19.2: init () = tuple_init () to %.loc6_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_20.3: init () = converted %.loc6_19.1, %.loc6_19.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_20.4: init C = class_init (%.loc6_20.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc6_21: init C = converted %.loc6_20.1, %.loc6_20.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_export_in_impl.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.x: ()} [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %struct: C = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .x = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_19.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc11_20.1: {.x: ()} = struct_literal (%.loc11_19.1)
// CHECK:STDOUT:   %.loc11_20.2: ref () = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc11_19.2: init () = tuple_init () to %.loc11_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_20.3: init () = converted %.loc11_19.1, %.loc11_19.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_20.4: init C = class_init (%.loc11_20.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc11_21: init C = converted %.loc11_20.1, %.loc11_20.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc11_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_export.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.x: ()} [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %struct: C = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir2, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir2, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .x = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_19.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc4_20.1: {.x: ()} = struct_literal (%.loc4_19.1)
// CHECK:STDOUT:   %.loc4_20.2: ref () = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc4_19.2: init () = tuple_init () to %.loc4_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc4_20.3: init () = converted %.loc4_19.1, %.loc4_19.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc4_20.4: init C = class_init (%.loc4_20.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc4_21: init C = converted %.loc4_20.1, %.loc4_20.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc4_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.x: ()} [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %struct: C = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir3, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir3, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir3, inst+2, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .x = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_19.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_20.1: {.x: ()} = struct_literal (%.loc6_19.1)
// CHECK:STDOUT:   %.loc6_20.2: ref () = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_19.2: init () = tuple_init () to %.loc6_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_20.3: init () = converted %.loc6_19.1, %.loc6_19.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_20.4: init C = class_init (%.loc6_20.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc6_21: init C = converted %.loc6_20.1, %.loc6_20.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_import_then_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.x: ()} [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %struct: C = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir2, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir2, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .x = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_19.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_20.1: {.x: ()} = struct_literal (%.loc6_19.1)
// CHECK:STDOUT:   %.loc6_20.2: ref () = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_19.2: init () = tuple_init () to %.loc6_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_20.3: init () = converted %.loc6_19.1, %.loc6_19.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_20.4: init C = class_init (%.loc6_20.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc6_21: init C = converted %.loc6_20.1, %.loc6_20.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_import_when_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.x: ()} [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %struct: C = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .x = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_19.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_20.1: {.x: ()} = struct_literal (%.loc6_19.1)
// CHECK:STDOUT:   %.loc6_20.2: ref () = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_19.2: init () = tuple_init () to %.loc6_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_20.3: init () = converted %.loc6_19.1, %.loc6_19.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_20.4: init C = class_init (%.loc6_20.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc6_21: init C = converted %.loc6_20.1, %.loc6_20.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_base_and_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.x: ()} [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %struct: C = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .x = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_19.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc7_20.1: {.x: ()} = struct_literal (%.loc7_19.1)
// CHECK:STDOUT:   %.loc7_20.2: ref () = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc7_19.2: init () = tuple_init () to %.loc7_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_20.3: init () = converted %.loc7_19.1, %.loc7_19.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_20.4: init C = class_init (%.loc7_20.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc7_21: init C = converted %.loc7_20.1, %.loc7_20.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc7_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_and_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.x: ()} [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %struct: C = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir2, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir2, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .x = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_19.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc7_20.1: {.x: ()} = struct_literal (%.loc7_19.1)
// CHECK:STDOUT:   %.loc7_20.2: ref () = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc7_19.2: init () = tuple_init () to %.loc7_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_20.3: init () = converted %.loc7_19.1, %.loc7_19.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_20.4: init C = class_init (%.loc7_20.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc7_21: init C = converted %.loc7_20.1, %.loc7_20.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc7_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_copy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.x: ()} [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %struct: C = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir2, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir2, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .x = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_19.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc7_20.1: {.x: ()} = struct_literal (%.loc7_19.1)
// CHECK:STDOUT:   %.loc7_20.2: ref () = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc7_19.2: init () = tuple_init () to %.loc7_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_20.3: init () = converted %.loc7_19.1, %.loc7_19.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_20.4: init C = class_init (%.loc7_20.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc7_21: init C = converted %.loc7_20.1, %.loc7_20.4 [template = constants.%struct]
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
// CHECK:STDOUT:   %C.ref: <error> = name_ref C, <error> [template = <error>]
// CHECK:STDOUT:   %Local: <error> = bind_alias Local, <error> [template = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_non_export_then_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.x: ()} [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %struct: C = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir2, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir2, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .x = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_19.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc7_20.1: {.x: ()} = struct_literal (%.loc7_19.1)
// CHECK:STDOUT:   %.loc7_20.2: ref () = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc7_19.2: init () = tuple_init () to %.loc7_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_20.3: init () = converted %.loc7_19.1, %.loc7_19.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_20.4: init C = class_init (%.loc7_20.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc7_21: init C = converted %.loc7_20.1, %.loc7_20.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc7_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- indirect_use_non_export_then_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.x: ()} [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %struct: C = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .c = %import_ref.1
// CHECK:STDOUT:     .C = %import_ref.2
// CHECK:STDOUT:     .indirect_c = %indirect_c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.2: type = import_ref ir2, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.2 [template = constants.%C]
// CHECK:STDOUT:   %indirect_c.var: ref C = var indirect_c
// CHECK:STDOUT:   %indirect_c: ref C = bind_name indirect_c, %indirect_c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .x = file.%import_ref.3
// CHECK:STDOUT:   .Self = file.%import_ref.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_28.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_29.1: {.x: ()} = struct_literal (%.loc6_28.1)
// CHECK:STDOUT:   %.loc6_29.2: ref () = class_element_access file.%indirect_c.var, element0
// CHECK:STDOUT:   %.loc6_28.2: init () = tuple_init () to %.loc6_29.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.3: init () = converted %.loc6_28.1, %.loc6_28.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.4: init C = class_init (%.loc6_29.3), file.%indirect_c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc6_30: init C = converted %.loc6_29.1, %.loc6_29.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%indirect_c.var, %.loc6_30
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
