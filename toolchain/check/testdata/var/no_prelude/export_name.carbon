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

// CHECK:STDERR: fail_todo_use_export.carbon:[[@LINE+6]]:1: in import [InImport]
// CHECK:STDERR: export.carbon:4:1: in import [InImport]
// CHECK:STDERR: base.carbon:4:5: error: semantics TODO: `Non-constant ImportRefLoaded (comes up with var)` [SemanticsTodo]
// CHECK:STDERR: var v: ();
// CHECK:STDERR:     ^
// CHECK:STDERR:
import library "export";

var w: () = v;

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %empty_tuple.type = binding_pattern v
// CHECK:STDOUT:     %.loc4_1: %empty_tuple.type = var_pattern %v.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %empty_tuple.type = var v
// CHECK:STDOUT:   %.loc4_9.1: type = splice_block %.loc4_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc4_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_9.3: type = converted %.loc4_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %empty_tuple.type = bind_name v, %v.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.v: ref %empty_tuple.type = import_ref Main//base, v, loaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %v: ref %empty_tuple.type = export v, imports.%Main.v
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_export.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.v: ref %empty_tuple.type = import_ref Main//export, v, loaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .v = imports.%Main.v
// CHECK:STDOUT:     .w = %w
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %empty_tuple.type = binding_pattern w
// CHECK:STDOUT:     %.loc12_1: %empty_tuple.type = var_pattern %w.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.var: ref %empty_tuple.type = var w
// CHECK:STDOUT:   %.loc12_9.1: type = splice_block %.loc12_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc12_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_9.3: type = converted %.loc12_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w: ref %empty_tuple.type = bind_name w, %w.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: ref %empty_tuple.type = name_ref v, imports.%Main.v
// CHECK:STDOUT:   %.loc12_13: init %empty_tuple.type = tuple_init () to file.%w.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_1: init %empty_tuple.type = converted %v.ref, %.loc12_13 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%w.var, %.loc12_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
