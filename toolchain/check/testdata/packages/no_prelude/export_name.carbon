// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// ============================================================================
// Setup files
// ============================================================================

// --- class.carbon

library "class" api;

class C {
  var x: ();
};

namespace NS;
class NS.NSC {
  var x: ();
};

// --- fail_todo_export_class.carbon

library "export_class" api;

import library "class";
// CHECK:STDERR: fail_todo_export_class.carbon:[[@LINE+4]]:1: ERROR: Semantics TODO: `ExportDirective`.
// CHECK:STDERR: export C;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
export C;

// --- non_export_class.carbon

library "non_export_class" api;

import library "export_class";

// --- fail_todo_export_export_class.carbon

library "export_export_class" api;

import library "export_class";
// CHECK:STDERR: fail_todo_export_export_class.carbon:[[@LINE+4]]:1: ERROR: Semantics TODO: `ExportDirective`.
// CHECK:STDERR: export C;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
export C;

// --- export_in_impl.carbon

// This is just providing an API for the implicit import.
library "export_in_impl" api;

// ============================================================================
// Test files
// ============================================================================

// --- fail_todo_use_export_class.carbon

library "use_export_class" api;

import library "export_class";

// CHECK:STDERR: fail_todo_use_export_class.carbon:[[@LINE+4]]:8: ERROR: Name `C` not found.
// CHECK:STDERR: var c: C = {.x = ()};
// CHECK:STDERR:        ^
// CHECK:STDERR:
var c: C = {.x = ()};

// --- fail_todo_export_in_impl.impl.carbon

library "export_in_impl" impl;

import library "class";
// CHECK:STDERR: fail_todo_export_in_impl.impl.carbon:[[@LINE+8]]:1: ERROR: `export` is only allowed in API files.
// CHECK:STDERR: export C;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_export_in_impl.impl.carbon:[[@LINE+4]]:1: ERROR: Semantics TODO: `ExportDirective`.
// CHECK:STDERR: export C;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
export C;

// --- fail_todo_export_export_class.impl.carbon

library "export_export_class" impl;

// CHECK:STDERR: fail_todo_export_export_class.impl.carbon:[[@LINE+4]]:8: ERROR: Name `C` not found.
// CHECK:STDERR: var c: C = {.x = ()};
// CHECK:STDERR:        ^
// CHECK:STDERR:
var c: C = {.x = ()};

// --- fail_todo_use_export_export_class.carbon

library "use_export_export_class" api;

import library "export_export_class";

// CHECK:STDERR: fail_todo_use_export_export_class.carbon:[[@LINE+4]]:8: ERROR: Name `C` not found.
// CHECK:STDERR: var c: C = {.x = ()};
// CHECK:STDERR:        ^
// CHECK:STDERR:
var c: C = {.x = ()};

// --- import_both.carbon

library "import_both" api;

import library "class";
import library "export_class";

var c: C = {.x = ()};

// --- import_both_reversed.carbon

library "import_both_reversed" api;

import library "export_class";
import library "class";

var c: C = {.x = ()};

// --- fail_use_non_export_class.carbon

library "fail_use_non_export_class" api;

import library "non_export_class";

// CHECK:STDERR: fail_use_non_export_class.carbon:[[@LINE+3]]:15: ERROR: Name `C` not found.
// CHECK:STDERR: alias Local = C;
// CHECK:STDERR:               ^
alias Local = C;

// CHECK:STDOUT: --- class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type C, () [template]
// CHECK:STDOUT:   %.3: type = struct_type {.x: ()} [template]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type NSC, () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NSC = %NSC.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NSC.decl: type = class_decl @NSC [template = constants.%NSC] {}
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
// CHECK:STDOUT: class @NSC {
// CHECK:STDOUT:   %.loc10_11.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc10_11.2: type = converted %.loc10_11.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_8: <unbound element of class NSC> = field_decl x, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NSC
// CHECK:STDOUT:   .x = %.loc10_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_export_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_export_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_export_export_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_in_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_export_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: <error> = name_ref C, <error> [template = <error>]
// CHECK:STDOUT:   %c.var: ref <error> = var c
// CHECK:STDOUT:   %c: ref <error> = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_19: () = tuple_literal ()
// CHECK:STDOUT:   %.loc10_20: {.x: ()} = struct_literal (%.loc10_19)
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_export_in_impl.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_export_export_class.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: <error> = name_ref C, <error> [template = <error>]
// CHECK:STDOUT:   %c.var: ref <error> = var c
// CHECK:STDOUT:   %c: ref <error> = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_19: () = tuple_literal ()
// CHECK:STDOUT:   %.loc8_20: {.x: ()} = struct_literal (%.loc8_19)
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_export_export_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: <error> = name_ref C, <error> [template = <error>]
// CHECK:STDOUT:   %c.var: ref <error> = var c
// CHECK:STDOUT:   %c: ref <error> = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_19: () = tuple_literal ()
// CHECK:STDOUT:   %.loc10_20: {.x: ()} = struct_literal (%.loc10_19)
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_both.carbon
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
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir1, inst+11, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.2, [template] {
// CHECK:STDOUT:     .NSC = %import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .x = file.%import_ref.4
// CHECK:STDOUT:   .Self = file.%import_ref.5
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
// CHECK:STDOUT: --- import_both_reversed.carbon
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
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir2, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir2, inst+11, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.2, [template] {
// CHECK:STDOUT:     .NSC = %import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir2, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .x = file.%import_ref.4
// CHECK:STDOUT:   .Self = file.%import_ref.5
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
// CHECK:STDOUT: --- fail_use_non_export_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Local = %Local
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: <error> = name_ref C, <error> [template = <error>]
// CHECK:STDOUT:   %Local: <error> = bind_alias Local, <error> [template = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
