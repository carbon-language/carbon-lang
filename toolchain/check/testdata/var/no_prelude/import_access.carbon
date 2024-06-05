// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/import_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/no_prelude/import_access.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- def.carbon

package Test library "def";

private var v: () = ();

// ============================================================================
// Test files
// ============================================================================

// --- fail_todo_def.impl.carbon

impl package Test library "def";

// CHECK:STDERR: fail_todo_def.impl.carbon:[[@LINE+4]]:14: ERROR: Name `v` not found.
// CHECK:STDERR: var v2: () = v;
// CHECK:STDERR:              ^
// CHECK:STDERR:
var v2: () = v;

// --- fail_local_def.carbon

package Test library "fail_local_def";

import library "def";

// CHECK:STDERR: fail_local_def.carbon:[[@LINE+4]]:14: ERROR: Name `v` not found.
// CHECK:STDERR: var v2: () = v;
// CHECK:STDERR:              ^
// CHECK:STDERR:
var v2: () = v;

// --- fail_other_def.carbon

package Other library "fail_other_def";

import Test library "def";

// CHECK:STDERR: fail_other_def.carbon:[[@LINE+3]]:14: ERROR: Name `v` not found.
// CHECK:STDERR: var v2: () = Test.v;
// CHECK:STDERR:              ^~~~~~
var v2: () = Test.v;

// CHECK:STDOUT: --- def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .v [private] = %v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_17.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc4_17.2: type = converted %.loc4_17.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %v.var: ref () = var v
// CHECK:STDOUT:   %v: ref () = bind_name v, %v.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_22.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc4_22.2: init () = tuple_init () to file.%v.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc4_23: init () = converted %.loc4_22.1, %.loc4_22.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%v.var, %.loc4_23
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_def.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .v2 = %v2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_10.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc8_10.2: type = converted %.loc8_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %v2.var: ref () = var v2
// CHECK:STDOUT:   %v2: ref () = bind_name v2, %v2.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: <error> = name_ref v, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%v2.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .v2 = %v2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_10.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc10_10.2: type = converted %.loc10_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %v2.var: ref () = var v2
// CHECK:STDOUT:   %v2: ref () = bind_name v2, %v2.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: <error> = name_ref v, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%v2.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Test = %Test
// CHECK:STDOUT:     .v2 = %v2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc9_10.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_10.2: type = converted %.loc9_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %v2.var: ref () = var v2
// CHECK:STDOUT:   %v2: ref () = bind_name v2, %v2.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref: <namespace> = name_ref Test, file.%Test [template = file.%Test]
// CHECK:STDOUT:   %v.ref: <error> = name_ref v, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%v2.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
