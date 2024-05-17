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

var v: ();

// --- export.carbon

library "export" api;

import library "base";

export v;

// ============================================================================
// Test files
// ============================================================================

// --- fail_todo_use_export.carbon

library "use_export" api;

// CHECK:STDERR: fail_todo_use_export.carbon:[[@LINE+9]]:1: In import.
// CHECK:STDERR: import library "export";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: export.carbon:4:1: In import.
// CHECK:STDERR: import library "base";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: base.carbon:4:5: ERROR: Semantics TODO: `Non-constant ImportRefLoaded (comes up with var)`.
// CHECK:STDERR: var v: ();
// CHECK:STDERR:     ^
import library "export";

var w: () = v;

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc4_9.2: type = converted %.loc4_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %v.var: ref () = var v
// CHECK:STDOUT:   %v: ref () = bind_name v, %v.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .v = %import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: ref () = import_ref ir1, inst+5, loaded
// CHECK:STDOUT:   %v: ref () = bind_export v, %import_ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .v = %import_ref
// CHECK:STDOUT:     .w = %w
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: ref () = import_ref ir1, inst+3, loaded [template = <error>]
// CHECK:STDOUT:   %.loc15_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc15_9.2: type = converted %.loc15_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %w.var: ref () = var w
// CHECK:STDOUT:   %w: ref () = bind_name w, %w.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: ref () = name_ref v, file.%import_ref [template = <error>]
// CHECK:STDOUT:   %.loc15_13: init () = tuple_init () to file.%w.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc15_14: init () = converted %v.ref, %.loc15_13 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%w.var, %.loc15_14
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
