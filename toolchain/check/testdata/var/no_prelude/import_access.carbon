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

package Test library "[[@TEST_NAME]]";

private var v: () = ();

// ============================================================================
// Test files
// ============================================================================

// --- def.impl.carbon

impl package Test library "[[@TEST_NAME]]";

var v2: () = v;

// --- fail_local_def.carbon

package Test library "[[@TEST_NAME]]";

import library "def";

// CHECK:STDERR: fail_local_def.carbon:[[@LINE+4]]:14: error: name `v` not found [NameNotFound]
// CHECK:STDERR: var v2: () = v;
// CHECK:STDERR:              ^
// CHECK:STDERR:
var v2: () = v;

// --- fail_other_def.carbon

package Other library "[[@TEST_NAME]]";

import Test library "def";

// CHECK:STDERR: fail_other_def.carbon:[[@LINE+3]]:14: error: name `v` not found [NameNotFound]
// CHECK:STDERR: var v2: () = Test.v;
// CHECK:STDERR:              ^~~~~~
var v2: () = Test.v;

// CHECK:STDOUT: --- def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .v [private] = %v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_17.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_17.2: type = converted %.loc4_17.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %v.var: ref %empty_tuple.type = var v
// CHECK:STDOUT:   %v: ref %empty_tuple.type = bind_name v, %v.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_22.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_22.2: init %empty_tuple.type = tuple_init () to file.%v.var [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_23: init %empty_tuple.type = converted %.loc4_22.1, %.loc4_22.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%v.var, %.loc4_23
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- def.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref: ref %empty_tuple.type = import_ref Test//def, inst+5, loaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .v [private] = imports.%import_ref
// CHECK:STDOUT:     .v2 = %v2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %.loc4_10.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_10.2: type = converted %.loc4_10.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %v2.var: ref %empty_tuple.type = var v2
// CHECK:STDOUT:   %v2: ref %empty_tuple.type = bind_name v2, %v2.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: ref %empty_tuple.type = name_ref v, imports.%import_ref
// CHECK:STDOUT:   %.loc4_14: init %empty_tuple.type = tuple_init () to file.%v2.var [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_15: init %empty_tuple.type = converted %v.ref, %.loc4_14 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%v2.var, %.loc4_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .v2 = %v2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %.loc10_10.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc10_10.2: type = converted %.loc10_10.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %v2.var: ref %empty_tuple.type = var v2
// CHECK:STDOUT:   %v2: ref %empty_tuple.type = bind_name v2, %v2.var
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Test: <namespace> = namespace file.%Test.import, [template] {
// CHECK:STDOUT:     import Test//def
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Test = imports.%Test
// CHECK:STDOUT:     .v2 = %v2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %.loc9_10.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc9_10.2: type = converted %.loc9_10.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %v2.var: ref %empty_tuple.type = var v2
// CHECK:STDOUT:   %v2: ref %empty_tuple.type = bind_name v2, %v2.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref: <namespace> = name_ref Test, imports.%Test [template = imports.%Test]
// CHECK:STDOUT:   %v.ref: <error> = name_ref v, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%v2.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
