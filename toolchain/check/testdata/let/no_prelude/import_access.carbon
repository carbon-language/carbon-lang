// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/no_prelude/import_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/no_prelude/import_access.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- def.carbon

package Test library "[[@TEST_NAME]]";

private let v: () = ();

// ============================================================================
// Test files
// ============================================================================

// --- def.impl.carbon

impl package Test library "[[@TEST_NAME]]";

let v2: () = v;

// --- fail_local_def.carbon

package Test library "[[@TEST_NAME]]";

import library "def";

// CHECK:STDERR: fail_local_def.carbon:[[@LINE+4]]:14: error: name `v` not found [NameNotFound]
// CHECK:STDERR: let v2: () = v;
// CHECK:STDERR:              ^
// CHECK:STDERR:
let v2: () = v;

// --- fail_other_def.carbon

package Other library "[[@TEST_NAME]]";

import Test library "def";

// CHECK:STDERR: fail_other_def.carbon:[[@LINE+4]]:14: error: member name `v` not found in `Test` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: let v2: () = Test.v;
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
let v2: () = Test.v;

// CHECK:STDOUT: --- def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .v [private] = %v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %empty_tuple.type = binding_pattern v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_17.1: type = splice_block %.loc4_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc4_17.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_17.3: type = converted %.loc4_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_22: %empty_tuple.type = converted @__global_init.%.loc4, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %v: %empty_tuple.type = bind_name v, %.loc4_22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- def.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Test.v: %empty_tuple.type = import_ref Test//def, v, loaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .v [private] = imports.%Test.v
// CHECK:STDOUT:     .v2 = %v2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v2.patt: %empty_tuple.type = binding_pattern v2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_10.1: type = splice_block %.loc4_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc4_10.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_10.3: type = converted %.loc4_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v2: %empty_tuple.type = bind_name v2, @__global_init.%v.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: %empty_tuple.type = name_ref v, imports.%Test.v
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .v2 = %v2
// CHECK:STDOUT:     .v = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v2.patt: %empty_tuple.type = binding_pattern v2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_10.1: type = splice_block %.loc10_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc10_10.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_10.3: type = converted %.loc10_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v2: %empty_tuple.type = bind_name v2, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.ref: <error> = name_ref v, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Test: <namespace> = namespace file.%Test.import, [concrete] {
// CHECK:STDOUT:     .v = <poisoned>
// CHECK:STDOUT:     import Test//def
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Test = imports.%Test
// CHECK:STDOUT:     .v2 = %v2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v2.patt: %empty_tuple.type = binding_pattern v2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_10.1: type = splice_block %.loc10_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc10_10.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_10.3: type = converted %.loc10_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v2: %empty_tuple.type = bind_name v2, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref: <namespace> = name_ref Test, imports.%Test [concrete = imports.%Test]
// CHECK:STDOUT:   %v.ref: <error> = name_ref v, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
