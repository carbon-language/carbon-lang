// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// ============================================================================
// Setup files
// ============================================================================

// --- base.carbon

package Other library "base";

class C {
  var x: ();
};

// --- export_import.carbon

package Other library "export_import";

export import library "base";

// --- export_import_copy.carbon

package Other library "export_import_copy";

export import library "base";

// --- export_name.carbon

package Other library "export_name";

import library "base";

export C;

// --- export_name_copy.carbon

package Other library "export_name_copy";

import library "base";

export C;

// ============================================================================
// Test files
// ============================================================================

// --- fail_todo_use_export_import.carbon

library "use_export_import";

import Other library "export_import";

// CHECK:STDERR: fail_todo_use_export_import.carbon:[[@LINE+4]]:8: ERROR: Name `C` not found.
// CHECK:STDERR: var c: Other.C = {.x = ()};
// CHECK:STDERR:        ^~~~~~~
// CHECK:STDERR:
var c: Other.C = {.x = ()};

// --- fail_todo_use_export_import_both.carbon

library "use_export_import_both";

import Other library "export_import";
import Other library "export_import_copy";

// CHECK:STDERR: fail_todo_use_export_import_both.carbon:[[@LINE+4]]:8: ERROR: Name `C` not found.
// CHECK:STDERR: var c: Other.C = {.x = ()};
// CHECK:STDERR:        ^~~~~~~
// CHECK:STDERR:
var c: Other.C = {.x = ()};

// --- use_export_name.carbon

library "use_export_name";

import Other library "export_name";

var c: Other.C = {.x = ()};

// --- fail_todo_use_export_name_both.carbon

library "use_export_name_both";

import Other library "export_name";
// CHECK:STDERR: fail_todo_use_export_name_both.carbon:[[@LINE+18]]:1: In import.
// CHECK:STDERR: import Other library "export_name_copy";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: export_name_copy.carbon:4:1: In import.
// CHECK:STDERR: import library "base";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: base.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: class C {
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_todo_use_export_name_both.carbon:[[@LINE-10]]:1: In import.
// CHECK:STDERR: import Other library "export_name";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: export_name.carbon:4:1: In import.
// CHECK:STDERR: import library "base";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: base.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: class C {
// CHECK:STDERR: ^~~~~~~~~
import Other library "export_name_copy";

// CHECK:STDERR: fail_todo_use_export_name_both.carbon:[[@LINE+4]]:8: In name lookup for `C`.
// CHECK:STDERR: var c: Other.C = {.x = ()};
// CHECK:STDERR:        ^~~~~~~
// CHECK:STDERR:
var c: Other.C = {.x = ()};

// --- fail_todo_use_export_all.carbon

library "use_export_all";

import Other library "export_import";
import Other library "export_name";
import Other library "export_import_copy";
// CHECK:STDERR: fail_todo_use_export_all.carbon:[[@LINE+18]]:1: In import.
// CHECK:STDERR: import Other library "export_name_copy";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: export_name_copy.carbon:4:1: In import.
// CHECK:STDERR: import library "base";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: base.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: class C {
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_todo_use_export_all.carbon:[[@LINE-11]]:1: In import.
// CHECK:STDERR: import Other library "export_name";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: export_name.carbon:4:1: In import.
// CHECK:STDERR: import library "base";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: base.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: class C {
// CHECK:STDERR: ^~~~~~~~~
import Other library "export_name_copy";

// CHECK:STDERR: fail_todo_use_export_all.carbon:[[@LINE+3]]:8: In name lookup for `C`.
// CHECK:STDERR: var c: Other.C = {.x = ()};
// CHECK:STDERR:        ^~~~~~~
var c: Other.C = {.x = ()};

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
// CHECK:STDOUT: --- export_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref = import_ref ir1, inst+1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_import_copy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref = import_ref ir1, inst+1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+7, unloaded
// CHECK:STDOUT:   %C: type = export C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT:   .x = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_name_copy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+7, unloaded
// CHECK:STDOUT:   %C: type = export C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT:   .x = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_export_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, %Other [template = %Other]
// CHECK:STDOUT:   %C.ref: <error> = name_ref C, <error> [template = <error>]
// CHECK:STDOUT:   %c.var: ref <error> = var c
// CHECK:STDOUT:   %c: ref <error> = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_25: () = tuple_literal ()
// CHECK:STDOUT:   %.loc10_26: {.x: ()} = struct_literal (%.loc10_25)
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_export_import_both.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, %Other [template = %Other]
// CHECK:STDOUT:   %C.ref: <error> = name_ref C, <error> [template = <error>]
// CHECK:STDOUT:   %c.var: ref <error> = var c
// CHECK:STDOUT:   %c: ref <error> = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_25: () = tuple_literal ()
// CHECK:STDOUT:   %.loc11_26: {.x: ()} = struct_literal (%.loc11_25)
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_name.carbon
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
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, %Other [template = %Other]
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+10, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+8, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+9, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT:   .x = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_25.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_26.1: {.x: ()} = struct_literal (%.loc6_25.1)
// CHECK:STDOUT:   %.loc6_26.2: ref () = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_25.2: init () = tuple_init () to %.loc6_26.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_26.3: init () = converted %.loc6_25.1, %.loc6_25.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_26.4: init C = class_init (%.loc6_26.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc6_27: init C = converted %.loc6_26.1, %.loc6_26.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_export_name_both.carbon
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
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, %Other [template = %Other]
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+10, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+8, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir2, inst+10, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT:   .x = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc29_25.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc29_26.1: {.x: ()} = struct_literal (%.loc29_25.1)
// CHECK:STDOUT:   %.loc29_26.2: ref () = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc29_25.2: init () = tuple_init () to %.loc29_26.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc29_26.3: init () = converted %.loc29_25.1, %.loc29_25.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc29_26.4: init C = class_init (%.loc29_26.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc29_27: init C = converted %.loc29_26.1, %.loc29_26.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc29_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_export_all.carbon
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
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, %Other [template = %Other]
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir2, inst+10, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir2, inst+8, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir4, inst+10, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT:   .x = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc30_25.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc30_26.1: {.x: ()} = struct_literal (%.loc30_25.1)
// CHECK:STDOUT:   %.loc30_26.2: ref () = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc30_25.2: init () = tuple_init () to %.loc30_26.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc30_26.3: init () = converted %.loc30_25.1, %.loc30_25.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc30_26.4: init C = class_init (%.loc30_26.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc30_27: init C = converted %.loc30_26.1, %.loc30_26.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc30_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
