// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/export_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/no_prelude/export_name.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- base.carbon

library "[[@TEST_NAME]]";

var v: ();

// --- export.carbon

library "[[@TEST_NAME]]";

import library "base";

export v;

// ============================================================================
// Test files
// ============================================================================

// --- fail_todo_use_export.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_use_export.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: export.carbon:4:1: in import [InImport]
// CHECK:STDERR: base.carbon:4:5: error: semantics TODO: `Non-constant ImportRefLoaded (comes up with var)` [SemanticsTodo]
// CHECK:STDERR: var v: ();
// CHECK:STDERR:     ^
import library "export";

var w: () = v;

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_9.2: type = converted %.loc4_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %v.var: ref %empty_tuple.type = var v
// CHECK:STDOUT:   %v: ref %empty_tuple.type = bind_name v, %v.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref: ref %empty_tuple.type = import_ref Main//base, inst+5, loaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %v: ref %empty_tuple.type = export v, imports.%import_ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref: ref %empty_tuple.type = import_ref Main//export, inst+4, loaded [template = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .v = imports.%import_ref
// CHECK:STDOUT:     .w = %w
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %.loc11_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc11_9.2: type = converted %.loc11_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %w.var: ref %empty_tuple.type = var w
// CHECK:STDOUT:   %w: ref %empty_tuple.type = bind_name w, %w.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: ref %empty_tuple.type = name_ref v, imports.%import_ref [template = <error>]
// CHECK:STDOUT:   %.loc11_13: init %empty_tuple.type = tuple_init () to file.%w.var [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_14: init %empty_tuple.type = converted %v.ref, %.loc11_13 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%w.var, %.loc11_14
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
