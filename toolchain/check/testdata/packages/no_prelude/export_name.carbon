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

namespace NS;
class NS.NSC {
  var y: ();
};

// --- export.carbon

library "export" api;

import library "base";
export C;
export NS.NSC;

// --- not_reexporting.carbon

library "not_reexporting" api;

import library "export";

// --- export_export.carbon

library "export_export" api;

import library "export";
export C;
export NS.NSC;

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
var nsc: NS.NSC = {.y = ()};

// --- export_export.impl.carbon

library "export_export" impl;

var c: C = {.x = ()};
var nsc: NS.NSC = {.y = ()};

// --- use_export_export.carbon

library "use_export_export" api;

import library "export_export";

var c: C = {.x = ()};
var nsc: NS.NSC = {.y = ()};

// --- fail_export_ns.carbon

library "fail_export_ns" api;

import library "base";

// CHECK:STDERR: fail_export_ns.carbon:[[@LINE+10]]:1: ERROR: Only imported entities are valid for `export`.
// CHECK:STDERR: export NS;
// CHECK:STDERR: ^~~~~~~~~~
// CHECK:STDERR: fail_export_ns.carbon:[[@LINE-5]]:1: In import.
// CHECK:STDERR: import library "base";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: base.carbon:8:1: Name is declared here.
// CHECK:STDERR: namespace NS;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
export NS;

// --- fail_export_decl.carbon

library "fail_export_decl" api;

class Local {}

// CHECK:STDERR: fail_export_decl.carbon:[[@LINE+7]]:1: ERROR: Only imported entities are valid for `export`.
// CHECK:STDERR: export Local;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_export_decl.carbon:[[@LINE-5]]:1: Name is declared here.
// CHECK:STDERR: class Local {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
export Local;

// --- fail_export_member.carbon

library "fail_export_member" api;

import library "base";

// CHECK:STDERR: fail_export_member.carbon:[[@LINE+7]]:10: ERROR: Name qualifiers are only allowed for entities that provide a scope.
// CHECK:STDERR: export C.x;
// CHECK:STDERR:          ^
// CHECK:STDERR: fail_export_member.carbon:[[@LINE+4]]:8: Non-scope entity referenced here.
// CHECK:STDERR: export C.x;
// CHECK:STDERR:        ^
// CHECK:STDERR:
export C.x;

// --- fail_todo_import_both.carbon

library "import_both" api;

import library "base";
// CHECK:STDERR: fail_todo_import_both.carbon:[[@LINE+32]]:1: In import.
// CHECK:STDERR: import library "export";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: export.carbon:4:1: In import.
// CHECK:STDERR: import library "base";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: base.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: class C {
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_todo_import_both.carbon:[[@LINE-10]]:1: In import.
// CHECK:STDERR: import library "base";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: base.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: class C {
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_import_both.carbon:[[@LINE+16]]:1: In import.
// CHECK:STDERR: import library "export";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: export.carbon:4:1: In import.
// CHECK:STDERR: import library "base";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: base.carbon:9:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: class NS.NSC {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_import_both.carbon:[[@LINE-26]]:1: In import.
// CHECK:STDERR: import library "base";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: base.carbon:9:1: Name is previously declared here.
// CHECK:STDERR: class NS.NSC {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
import library "export";

var c: C = {.x = ()};
var nsc: NS.NSC = {.y = ()};

// --- fail_todo_import_both_reversed.carbon

library "import_both_reversed" api;

import library "export";
// CHECK:STDERR: fail_todo_import_both_reversed.carbon:[[@LINE+32]]:1: In import.
// CHECK:STDERR: import library "base";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: base.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: class C {
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_todo_import_both_reversed.carbon:[[@LINE-7]]:1: In import.
// CHECK:STDERR: import library "export";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: export.carbon:4:1: In import.
// CHECK:STDERR: import library "base";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: base.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: class C {
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_import_both_reversed.carbon:[[@LINE+16]]:1: In import.
// CHECK:STDERR: import library "base";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: base.carbon:9:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: class NS.NSC {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_import_both_reversed.carbon:[[@LINE-23]]:1: In import.
// CHECK:STDERR: import library "export";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: export.carbon:4:1: In import.
// CHECK:STDERR: import library "base";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: base.carbon:9:1: Name is previously declared here.
// CHECK:STDERR: class NS.NSC {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
import library "base";

var c: C = {.x = ()};
var nsc: NS.NSC = {.y = ()};

// --- fail_use_not_reexporting.carbon

library "fail_use_not_reexporting" api;

import library "not_reexporting";

// CHECK:STDERR: fail_use_not_reexporting.carbon:[[@LINE+4]]:15: ERROR: Name `C` not found.
// CHECK:STDERR: alias Local = C;
// CHECK:STDERR:               ^
// CHECK:STDERR:
alias Local = C;
// CHECK:STDERR: fail_use_not_reexporting.carbon:[[@LINE+4]]:17: ERROR: Name `NS` not found.
// CHECK:STDERR: alias NSLocal = NS.NSC;
// CHECK:STDERR:                 ^~
// CHECK:STDERR:
alias NSLocal = NS.NSC;

// --- repeat_export.carbon

library "repeat_export" api;

import library "base";

export C;
export C;

// --- fail_todo_use_repeat_export.carbon

library "use_repeat_export" api;

// CHECK:STDERR: fail_todo_use_repeat_export.carbon:[[@LINE+19]]:1: In import.
// CHECK:STDERR: import library "repeat_export";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: repeat_export.carbon:4:1: In import.
// CHECK:STDERR: import library "base";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: base.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: class C {
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_todo_use_repeat_export.carbon:[[@LINE+10]]:1: In import.
// CHECK:STDERR: import library "repeat_export";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: repeat_export.carbon:4:1: In import.
// CHECK:STDERR: import library "base";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: base.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: class C {
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
import library "repeat_export";

var c: C = {.x = ()};

// --- fail_modifiers.carbon

library "fail_modifiers" api;

import library "base";

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: `private` not allowed on `export` declaration.
// CHECK:STDERR: private export C;
// CHECK:STDERR: ^~~~~~~
private export C;

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type C, () [template]
// CHECK:STDOUT:   %.3: type = struct_type {.x: ()} [template]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type NSC, () [template]
// CHECK:STDOUT:   %.5: type = struct_type {.y: ()} [template]
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
// CHECK:STDOUT:   %.loc10_8: <unbound element of class NSC> = field_decl y, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NSC
// CHECK:STDOUT:   .y = %.loc10_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [template]
// CHECK:STDOUT:   %.3: type = struct_type {.y: ()} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir1, inst+11, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.2, [template] {
// CHECK:STDOUT:     .NSC = %import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: type = import_ref ir1, inst+12, loaded [template = constants.%NSC]
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %C: type = bind_export C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref ir1, inst+17, unloaded
// CHECK:STDOUT:   %NSC: type = bind_export NSC, %import_ref.3 [template = constants.%NSC]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .x = file.%import_ref.4
// CHECK:STDOUT:   .Self = file.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.6
// CHECK:STDOUT:   .y = file.%import_ref.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- not_reexporting.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir1, inst+3, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.2, [template] {
// CHECK:STDOUT:     .NSC = %import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+21, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [template]
// CHECK:STDOUT:   %.3: type = struct_type {.y: ()} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+13, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir1, inst+3, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.2, [template] {
// CHECK:STDOUT:     .NSC = %import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: type = import_ref ir1, inst+21, loaded [template = constants.%NSC]
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref ir1, inst+11, unloaded
// CHECK:STDOUT:   %C: type = bind_export C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+20, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref ir1, inst+19, unloaded
// CHECK:STDOUT:   %NSC: type = bind_export NSC, %import_ref.3 [template = constants.%NSC]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.4
// CHECK:STDOUT:   .x = file.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .y = file.%import_ref.6
// CHECK:STDOUT:   .Self = file.%import_ref.7
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
// CHECK:STDOUT:   %struct.1: C = struct_value (%tuple) [template]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [template]
// CHECK:STDOUT:   %.4: type = struct_type {.y: ()} [template]
// CHECK:STDOUT:   %.5: type = ptr_type {.y: ()} [template]
// CHECK:STDOUT:   %struct.2: NSC = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .nsc = %nsc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+13, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir1, inst+3, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.2, [template] {
// CHECK:STDOUT:     .NSC = %import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: type = import_ref ir1, inst+21, loaded [template = constants.%NSC]
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref ir1, inst+11, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+20, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref ir1, inst+19, unloaded
// CHECK:STDOUT:   %NSC.ref: type = name_ref NSC, %import_ref.3 [template = constants.%NSC]
// CHECK:STDOUT:   %nsc.var: ref NSC = var nsc
// CHECK:STDOUT:   %nsc: ref NSC = bind_name nsc, %nsc.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.4
// CHECK:STDOUT:   .x = file.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .y = file.%import_ref.6
// CHECK:STDOUT:   .Self = file.%import_ref.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_19.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_20.1: {.x: ()} = struct_literal (%.loc6_19.1)
// CHECK:STDOUT:   %.loc6_20.2: ref () = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_19.2: init () = tuple_init () to %.loc6_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_20.3: init () = converted %.loc6_19.1, %.loc6_19.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_20.4: init C = class_init (%.loc6_20.3), file.%c.var [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc6_21: init C = converted %.loc6_20.1, %.loc6_20.4 [template = constants.%struct.1]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_21
// CHECK:STDOUT:   %.loc7_26.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc7_27.1: {.y: ()} = struct_literal (%.loc7_26.1)
// CHECK:STDOUT:   %.loc7_27.2: ref () = class_element_access file.%nsc.var, element0
// CHECK:STDOUT:   %.loc7_26.2: init () = tuple_init () to %.loc7_27.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_27.3: init () = converted %.loc7_26.1, %.loc7_26.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_27.4: init NSC = class_init (%.loc7_27.3), file.%nsc.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc7_28: init NSC = converted %.loc7_27.1, %.loc7_27.4 [template = constants.%struct.2]
// CHECK:STDOUT:   assign file.%nsc.var, %.loc7_28
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
// CHECK:STDOUT:   %struct.1: C = struct_value (%tuple) [template]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [template]
// CHECK:STDOUT:   %.4: type = struct_type {.y: ()} [template]
// CHECK:STDOUT:   %.5: type = ptr_type {.y: ()} [template]
// CHECK:STDOUT:   %struct.2: NSC = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .nsc = %nsc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir0, inst+13, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir0, inst+3, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.2, [template] {
// CHECK:STDOUT:     .NSC = %import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: type = import_ref ir0, inst+21, loaded [template = constants.%NSC]
// CHECK:STDOUT:   %import_ref.4 = import_ref ir0, inst+11, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref ir0, inst+12, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:   %import_ref.6 = import_ref ir0, inst+19, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref ir0, inst+20, unloaded
// CHECK:STDOUT:   %NSC.ref: type = name_ref NSC, %import_ref.3 [template = constants.%NSC]
// CHECK:STDOUT:   %nsc.var: ref NSC = var nsc
// CHECK:STDOUT:   %nsc: ref NSC = bind_name nsc, %nsc.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.4
// CHECK:STDOUT:   .x = file.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .y = file.%import_ref.6
// CHECK:STDOUT:   .Self = file.%import_ref.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_19.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc4_20.1: {.x: ()} = struct_literal (%.loc4_19.1)
// CHECK:STDOUT:   %.loc4_20.2: ref () = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc4_19.2: init () = tuple_init () to %.loc4_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc4_20.3: init () = converted %.loc4_19.1, %.loc4_19.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc4_20.4: init C = class_init (%.loc4_20.3), file.%c.var [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc4_21: init C = converted %.loc4_20.1, %.loc4_20.4 [template = constants.%struct.1]
// CHECK:STDOUT:   assign file.%c.var, %.loc4_21
// CHECK:STDOUT:   %.loc5_26.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc5_27.1: {.y: ()} = struct_literal (%.loc5_26.1)
// CHECK:STDOUT:   %.loc5_27.2: ref () = class_element_access file.%nsc.var, element0
// CHECK:STDOUT:   %.loc5_26.2: init () = tuple_init () to %.loc5_27.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc5_27.3: init () = converted %.loc5_26.1, %.loc5_26.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc5_27.4: init NSC = class_init (%.loc5_27.3), file.%nsc.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc5_28: init NSC = converted %.loc5_27.1, %.loc5_27.4 [template = constants.%struct.2]
// CHECK:STDOUT:   assign file.%nsc.var, %.loc5_28
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
// CHECK:STDOUT:   %struct.1: C = struct_value (%tuple) [template]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [template]
// CHECK:STDOUT:   %.4: type = struct_type {.y: ()} [template]
// CHECK:STDOUT:   %.5: type = ptr_type {.y: ()} [template]
// CHECK:STDOUT:   %struct.2: NSC = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .nsc = %nsc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+13, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir1, inst+3, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.2, [template] {
// CHECK:STDOUT:     .NSC = %import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: type = import_ref ir1, inst+21, loaded [template = constants.%NSC]
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+11, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref ir1, inst+12, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+19, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref ir1, inst+20, unloaded
// CHECK:STDOUT:   %NSC.ref: type = name_ref NSC, %import_ref.3 [template = constants.%NSC]
// CHECK:STDOUT:   %nsc.var: ref NSC = var nsc
// CHECK:STDOUT:   %nsc: ref NSC = bind_name nsc, %nsc.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.4
// CHECK:STDOUT:   .x = file.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .y = file.%import_ref.6
// CHECK:STDOUT:   .Self = file.%import_ref.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_19.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_20.1: {.x: ()} = struct_literal (%.loc6_19.1)
// CHECK:STDOUT:   %.loc6_20.2: ref () = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_19.2: init () = tuple_init () to %.loc6_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_20.3: init () = converted %.loc6_19.1, %.loc6_19.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_20.4: init C = class_init (%.loc6_20.3), file.%c.var [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc6_21: init C = converted %.loc6_20.1, %.loc6_20.4 [template = constants.%struct.1]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_21
// CHECK:STDOUT:   %.loc7_26.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc7_27.1: {.y: ()} = struct_literal (%.loc7_26.1)
// CHECK:STDOUT:   %.loc7_27.2: ref () = class_element_access file.%nsc.var, element0
// CHECK:STDOUT:   %.loc7_26.2: init () = tuple_init () to %.loc7_27.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_27.3: init () = converted %.loc7_26.1, %.loc7_26.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_27.4: init NSC = class_init (%.loc7_27.3), file.%nsc.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc7_28: init NSC = converted %.loc7_27.1, %.loc7_27.4 [template = constants.%struct.2]
// CHECK:STDOUT:   assign file.%nsc.var, %.loc7_28
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_export_ns.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+1, unloaded
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir1, inst+11, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.2, [template] {
// CHECK:STDOUT:     .NSC = %import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+12, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_export_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Local: type = class_type @Local [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Local = %Local.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Local.decl: type = class_decl @Local [template = constants.%Local] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Local {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Local
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_export_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir1, inst+11, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.2, [template] {
// CHECK:STDOUT:     .NSC = %import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .x = file.%import_ref.4
// CHECK:STDOUT:   .Self = file.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_both.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.x: ()} [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %struct.1: C = struct_value (%tuple) [template]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [template]
// CHECK:STDOUT:   %.4: type = struct_type {.y: ()} [template]
// CHECK:STDOUT:   %.5: type = ptr_type {.y: ()} [template]
// CHECK:STDOUT:   %struct.2: NSC = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .nsc = %nsc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir1, inst+11, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.2, [template] {
// CHECK:STDOUT:     .NSC = %import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: type = import_ref ir1, inst+12, loaded [template = constants.%NSC]
// CHECK:STDOUT:   %import_ref.4 = import_ref ir2, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref ir2, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:   %import_ref.8 = import_ref ir1, inst+13, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref ir1, inst+17, unloaded
// CHECK:STDOUT:   %NSC.ref: type = name_ref NSC, %import_ref.3 [template = constants.%NSC]
// CHECK:STDOUT:   %nsc.var: ref NSC = var nsc
// CHECK:STDOUT:   %nsc: ref NSC = bind_name nsc, %nsc.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .x = file.%import_ref.6
// CHECK:STDOUT:   .Self = file.%import_ref.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.8
// CHECK:STDOUT:   .y = file.%import_ref.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc39_19.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc39_20.1: {.x: ()} = struct_literal (%.loc39_19.1)
// CHECK:STDOUT:   %.loc39_20.2: ref () = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc39_19.2: init () = tuple_init () to %.loc39_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc39_20.3: init () = converted %.loc39_19.1, %.loc39_19.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc39_20.4: init C = class_init (%.loc39_20.3), file.%c.var [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc39_21: init C = converted %.loc39_20.1, %.loc39_20.4 [template = constants.%struct.1]
// CHECK:STDOUT:   assign file.%c.var, %.loc39_21
// CHECK:STDOUT:   %.loc40_26.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc40_27.1: {.y: ()} = struct_literal (%.loc40_26.1)
// CHECK:STDOUT:   %.loc40_27.2: ref () = class_element_access file.%nsc.var, element0
// CHECK:STDOUT:   %.loc40_26.2: init () = tuple_init () to %.loc40_27.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc40_27.3: init () = converted %.loc40_26.1, %.loc40_26.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc40_27.4: init NSC = class_init (%.loc40_27.3), file.%nsc.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc40_28: init NSC = converted %.loc40_27.1, %.loc40_27.4 [template = constants.%struct.2]
// CHECK:STDOUT:   assign file.%nsc.var, %.loc40_28
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_both_reversed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.x: ()} [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %struct.1: C = struct_value (%tuple) [template]
// CHECK:STDOUT:   %NSC: type = class_type @NSC [template]
// CHECK:STDOUT:   %.4: type = struct_type {.y: ()} [template]
// CHECK:STDOUT:   %.5: type = ptr_type {.y: ()} [template]
// CHECK:STDOUT:   %struct.2: NSC = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .nsc = %nsc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+13, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir1, inst+3, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.2, [template] {
// CHECK:STDOUT:     .NSC = %import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: type = import_ref ir1, inst+21, loaded [template = constants.%NSC]
// CHECK:STDOUT:   %import_ref.4 = import_ref ir2, inst+1, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref ir2, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref ir1, inst+11, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:   %import_ref.8 = import_ref ir1, inst+20, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref ir1, inst+19, unloaded
// CHECK:STDOUT:   %NSC.ref: type = name_ref NSC, %import_ref.3 [template = constants.%NSC]
// CHECK:STDOUT:   %nsc.var: ref NSC = var nsc
// CHECK:STDOUT:   %nsc: ref NSC = bind_name nsc, %nsc.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.6
// CHECK:STDOUT:   .x = file.%import_ref.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NSC {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .y = file.%import_ref.8
// CHECK:STDOUT:   .Self = file.%import_ref.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc39_19.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc39_20.1: {.x: ()} = struct_literal (%.loc39_19.1)
// CHECK:STDOUT:   %.loc39_20.2: ref () = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc39_19.2: init () = tuple_init () to %.loc39_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc39_20.3: init () = converted %.loc39_19.1, %.loc39_19.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc39_20.4: init C = class_init (%.loc39_20.3), file.%c.var [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc39_21: init C = converted %.loc39_20.1, %.loc39_20.4 [template = constants.%struct.1]
// CHECK:STDOUT:   assign file.%c.var, %.loc39_21
// CHECK:STDOUT:   %.loc40_26.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc40_27.1: {.y: ()} = struct_literal (%.loc40_26.1)
// CHECK:STDOUT:   %.loc40_27.2: ref () = class_element_access file.%nsc.var, element0
// CHECK:STDOUT:   %.loc40_26.2: init () = tuple_init () to %.loc40_27.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc40_27.3: init () = converted %.loc40_26.1, %.loc40_26.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc40_27.4: init NSC = class_init (%.loc40_27.3), file.%nsc.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc40_28: init NSC = converted %.loc40_27.1, %.loc40_27.4 [template = constants.%struct.2]
// CHECK:STDOUT:   assign file.%nsc.var, %.loc40_28
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_not_reexporting.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Local = %Local
// CHECK:STDOUT:     .NSLocal = %NSLocal
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: <error> = name_ref C, <error> [template = <error>]
// CHECK:STDOUT:   %Local: <error> = bind_alias Local, <error> [template = <error>]
// CHECK:STDOUT:   %NS.ref: <error> = name_ref NS, <error> [template = <error>]
// CHECK:STDOUT:   %NSLocal: <error> = bind_alias NSLocal, <error> [template = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- repeat_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir1, inst+11, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.2, [template] {
// CHECK:STDOUT:     .NSC = %import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %C.loc6: type = bind_export C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %C.loc7: type = bind_export C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .x = file.%import_ref.4
// CHECK:STDOUT:   .Self = file.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_repeat_export.carbon
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
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+13, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+14, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+11, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref C = var c
// CHECK:STDOUT:   %c: ref C = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT:   .x = file.%import_ref.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc25_19.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc25_20.1: {.x: ()} = struct_literal (%.loc25_19.1)
// CHECK:STDOUT:   %.loc25_20.2: ref () = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc25_19.2: init () = tuple_init () to %.loc25_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc25_20.3: init () = converted %.loc25_19.1, %.loc25_19.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc25_20.4: init C = class_init (%.loc25_20.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc25_21: init C = converted %.loc25_20.1, %.loc25_20.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc25_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: ()} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir1, inst+11, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.2, [template] {
// CHECK:STDOUT:     .NSC = %import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+7, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %C: type = bind_export C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .x = file.%import_ref.4
// CHECK:STDOUT:   .Self = file.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
