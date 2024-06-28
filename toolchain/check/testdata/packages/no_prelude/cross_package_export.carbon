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

package Other library "base";

class C {
  var x: ();
};

// --- conflict.carbon

package Other library "conflict";

fn C() {}

// --- export_import.carbon

package Other library "export_import";

export import library "base";

// --- export_import_copy.carbon

package Other library "export_import_copy";

export import library "base";

// --- export_import_indirect.carbon

package Other library "export_import_indirect";

export import library "export_import";

// --- export_name.carbon

package Other library "export_name";

import library "base";

export C;

// --- export_name_copy.carbon

package Other library "export_name_copy";

import library "base";

export C;

// --- export_name_indirect.carbon

package Other library "export_name_indirect";

import library "export_name";

export C;

// ============================================================================
// Test files
// ============================================================================

// --- use_export_import.carbon

library "use_export_import";

import Other library "export_import";

var c: Other.C = {.x = ()};

// --- use_export_import_with_copy.carbon

library "use_export_import_with_copy";

import Other library "export_import";
import Other library "export_import_copy";

var c: Other.C = {.x = ()};

// --- use_export_import_indirect.carbon

library "use_export_import_indirect";

import Other library "export_import_indirect";

var c: Other.C = {.x = ()};

// --- use_export_name.carbon

library "use_export_name";

import Other library "export_name";

var c: Other.C = {.x = ()};

// --- use_export_name_with_copy.carbon

library "use_export_name_with_copy";

import Other library "export_name";
import Other library "export_name_copy";

var c: Other.C = {.x = ()};

// --- use_export_name_indirect.carbon

library "use_export_name_indirect";

import Other library "export_name_indirect";

var c: Other.C = {.x = ()};

// --- use_export_all.carbon

library "use_export_all";

import Other library "export_import";
import Other library "export_name";
import Other library "export_import_copy";
import Other library "export_name_copy";
import Other library "export_import_indirect";
import Other library "export_name_indirect";

var c: Other.C = {.x = ()};

// --- unused_conflict_on_export_import.carbon

library "unused_conflict_on_export_import";

import Other library "export_import";
import Other library "conflict";

// --- unused_conflict_on_export_name.carbon

library "unused_conflict_on_export_name";

import Other library "export_name";
import Other library "conflict";

// --- fail_conflict_on_export_import.carbon

library "fail_conflict_on_export_import";

// CHECK:STDERR: fail_conflict_on_export_import.carbon:[[@LINE+12]]:1: In import.
// CHECK:STDERR: import Other library "export_import";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: base.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: class C {
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_conflict_on_export_import.carbon:[[@LINE+6]]:1: In import.
// CHECK:STDERR: import Other library "export_import";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: conflict.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: fn C() {}
// CHECK:STDERR: ^~~~~~~~
import Other library "export_import";
import Other library "conflict";

// CHECK:STDERR: fail_conflict_on_export_import.carbon:[[@LINE+4]]:11: In name lookup for `C`.
// CHECK:STDERR: alias C = Other.C;
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR:
alias C = Other.C;

// --- fail_conflict_on_export_name.carbon

library "fail_conflict_on_export_name";

// CHECK:STDERR: fail_conflict_on_export_name.carbon:[[@LINE+15]]:1: In import.
// CHECK:STDERR: import Other library "export_name";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: conflict.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn C() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_conflict_on_export_name.carbon:[[@LINE+9]]:1: In import.
// CHECK:STDERR: import Other library "export_name";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: export_name.carbon:4:1: In import.
// CHECK:STDERR: import library "base";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: base.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: class C {
// CHECK:STDERR: ^~~~~~~~~
import Other library "export_name";
import Other library "conflict";

// CHECK:STDERR: fail_conflict_on_export_name.carbon:[[@LINE+3]]:11: In name lookup for `C`.
// CHECK:STDERR: alias C = Other.C;
// CHECK:STDERR:           ^~~~~~~
alias C = Other.C;

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %C, %.1 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.x: %.1} [template]
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
// CHECK:STDOUT:   %.loc5_11.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc5_11.2: type = converted %.loc5_11.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc5_8: %.2 = field_decl x, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .x = %.loc5_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %<default>.import = import <invalid>
// CHECK:STDOUT:   %import_ref = import_ref ir1, inst+1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_import_copy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %<default>.import = import <invalid>
// CHECK:STDOUT:   %import_ref = import_ref ir1, inst+1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_import_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %<default>.import = import <invalid>
// CHECK:STDOUT:   %import_ref = import_ref ir2, inst+1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: %.1} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %<default>.import = import <invalid>
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
// CHECK:STDOUT:   %.2: type = struct_type {.x: %.1} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %<default>.import = import <invalid>
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
// CHECK:STDOUT: --- export_name_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: %.1} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %<default>.import = import <invalid>
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+11, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+10, unloaded
// CHECK:STDOUT:   %C: type = export C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT:   .x = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: %.1} [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %struct: %C = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other: <namespace> = namespace %Other.import, [template] {}
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, %Other [template = %Other]
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir2, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+7, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
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
// CHECK:STDOUT:   %.loc6_25.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc6_26.1: %.2 = struct_literal (%.loc6_25.1)
// CHECK:STDOUT:   %.loc6_26.2: ref %.1 = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_25.2: init %.1 = tuple_init () to %.loc6_26.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_26.3: init %.1 = converted %.loc6_25.1, %.loc6_25.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_26.4: init %C = class_init (%.loc6_26.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc6_27: init %C = converted %.loc6_26.1, %.loc6_26.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_import_with_copy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: %.1} [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %struct: %C = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other: <namespace> = namespace %Other.import, [template] {}
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, %Other [template = %Other]
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir3, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir3, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir3, inst+7, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
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
// CHECK:STDOUT:   %.loc7_25.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc7_26.1: %.2 = struct_literal (%.loc7_25.1)
// CHECK:STDOUT:   %.loc7_26.2: ref %.1 = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc7_25.2: init %.1 = tuple_init () to %.loc7_26.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_26.3: init %.1 = converted %.loc7_25.1, %.loc7_25.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_26.4: init %C = class_init (%.loc7_26.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc7_27: init %C = converted %.loc7_26.1, %.loc7_26.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc7_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_import_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: %.1} [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %struct: %C = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other: <namespace> = namespace %Other.import, [template] {}
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, %Other [template = %Other]
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir3, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir3, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir3, inst+7, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
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
// CHECK:STDOUT:   %.loc6_25.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc6_26.1: %.2 = struct_literal (%.loc6_25.1)
// CHECK:STDOUT:   %.loc6_26.2: ref %.1 = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_25.2: init %.1 = tuple_init () to %.loc6_26.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_26.3: init %.1 = converted %.loc6_25.1, %.loc6_25.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_26.4: init %C = class_init (%.loc6_26.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc6_27: init %C = converted %.loc6_26.1, %.loc6_26.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: %.1} [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %struct: %C = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other: <namespace> = namespace %Other.import, [template] {}
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, %Other [template = %Other]
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+11, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+10, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
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
// CHECK:STDOUT:   %.loc6_25.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc6_26.1: %.2 = struct_literal (%.loc6_25.1)
// CHECK:STDOUT:   %.loc6_26.2: ref %.1 = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_25.2: init %.1 = tuple_init () to %.loc6_26.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_26.3: init %.1 = converted %.loc6_25.1, %.loc6_25.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_26.4: init %C = class_init (%.loc6_26.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc6_27: init %C = converted %.loc6_26.1, %.loc6_26.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_name_with_copy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: %.1} [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %struct: %C = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other: <namespace> = namespace %Other.import, [template] {}
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, %Other [template = %Other]
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+11, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+10, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
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
// CHECK:STDOUT:   %.loc7_25.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc7_26.1: %.2 = struct_literal (%.loc7_25.1)
// CHECK:STDOUT:   %.loc7_26.2: ref %.1 = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc7_25.2: init %.1 = tuple_init () to %.loc7_26.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_26.3: init %.1 = converted %.loc7_25.1, %.loc7_25.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_26.4: init %C = class_init (%.loc7_26.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc7_27: init %C = converted %.loc7_26.1, %.loc7_26.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc7_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_name_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: %.1} [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %struct: %C = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other: <namespace> = namespace %Other.import, [template] {}
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, %Other [template = %Other]
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+11, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+10, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
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
// CHECK:STDOUT:   %.loc6_25.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc6_26.1: %.2 = struct_literal (%.loc6_25.1)
// CHECK:STDOUT:   %.loc6_26.2: ref %.1 = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc6_25.2: init %.1 = tuple_init () to %.loc6_26.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_26.3: init %.1 = converted %.loc6_25.1, %.loc6_25.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_26.4: init %C = class_init (%.loc6_26.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc6_27: init %C = converted %.loc6_26.1, %.loc6_26.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_export_all.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: %.1} [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %struct: %C = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other: <namespace> = namespace %Other.import, [template] {}
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, %Other [template = %Other]
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir2, inst+11, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir2, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+10, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
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
// CHECK:STDOUT:   %.loc11_25.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc11_26.1: %.2 = struct_literal (%.loc11_25.1)
// CHECK:STDOUT:   %.loc11_26.2: ref %.1 = class_element_access file.%c.var, element0
// CHECK:STDOUT:   %.loc11_25.2: init %.1 = tuple_init () to %.loc11_26.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_26.3: init %.1 = converted %.loc11_25.1, %.loc11_25.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_26.4: init %C = class_init (%.loc11_26.3), file.%c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc11_27: init %C = converted %.loc11_26.1, %.loc11_26.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%c.var, %.loc11_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unused_conflict_on_export_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other: <namespace> = namespace %Other.import, [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unused_conflict_on_export_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other: <namespace> = namespace %Other.import, [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conflict_on_export_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other: <namespace> = namespace %Other.import, [template] {}
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, %Other [template = %Other]
// CHECK:STDOUT:   %import_ref.1: %C.type = import_ref ir2, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir3, inst+1, unloaded
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %C: %C.type = bind_alias C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conflict_on_export_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: %.1} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Other = %Other
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Other.import = import Other
// CHECK:STDOUT:   %Other: <namespace> = namespace %Other.import, [template] {}
// CHECK:STDOUT:   %Other.ref: <namespace> = name_ref Other, %Other [template = %Other]
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+11, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+10, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir2, inst+1, unloaded
// CHECK:STDOUT:   %C.ref: type = name_ref C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT:   %C: type = bind_alias C, %import_ref.1 [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT:   .x = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
