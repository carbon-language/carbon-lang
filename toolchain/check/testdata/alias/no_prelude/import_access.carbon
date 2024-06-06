// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/no_prelude/import_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/no_prelude/import_access.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- def.carbon

package Test library "def";

class C {}
private alias A = C;

// ============================================================================
// Test files
// ============================================================================

// --- fail_todo_def.impl.carbon

impl package Test library "def";

// CHECK:STDERR: fail_todo_def.impl.carbon:[[@LINE+4]]:11: ERROR: Name `A` not found.
// CHECK:STDERR: var inst: A = {};
// CHECK:STDERR:           ^
// CHECK:STDERR:
var inst: A = {};

// --- fail_local_def.carbon

package Test library "fail_local_def";

import library "def";

// CHECK:STDERR: fail_local_def.carbon:[[@LINE+4]]:11: ERROR: Name `A` not found.
// CHECK:STDERR: var inst: A = {};
// CHECK:STDERR:           ^
// CHECK:STDERR:
var inst: A = {};

// --- fail_other_def.carbon

package Other library "fail_other_def";

import Test library "def";

// CHECK:STDERR: fail_other_def.carbon:[[@LINE+3]]:11: ERROR: Name `A` not found.
// CHECK:STDERR: var inst: Test.A = {};
// CHECK:STDERR:           ^~~~~~
var inst: Test.A = {};

// CHECK:STDOUT: --- def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .A [private] = %A
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %A: type = bind_alias A, %C.decl [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_def.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref
// CHECK:STDOUT:     .inst = %inst
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref = import_ref ir0, inst+1, unloaded
// CHECK:STDOUT:   %A.ref: <error> = name_ref A, <error> [template = <error>]
// CHECK:STDOUT:   %inst.var: ref <error> = var inst
// CHECK:STDOUT:   %inst: ref <error> = bind_name inst, %inst.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: {} = struct_literal ()
// CHECK:STDOUT:   assign file.%inst.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref
// CHECK:STDOUT:     .inst = %inst
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref = import_ref ir1, inst+1, unloaded
// CHECK:STDOUT:   %A.ref: <error> = name_ref A, <error> [template = <error>]
// CHECK:STDOUT:   %inst.var: ref <error> = var inst
// CHECK:STDOUT:   %inst: ref <error> = bind_name inst, %inst.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10: {} = struct_literal ()
// CHECK:STDOUT:   assign file.%inst.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Test = %Test
// CHECK:STDOUT:     .inst = %inst
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Test.ref: <namespace> = name_ref Test, %Test [template = %Test]
// CHECK:STDOUT:   %A.ref: <error> = name_ref A, <error> [template = <error>]
// CHECK:STDOUT:   %inst.var: ref <error> = var inst
// CHECK:STDOUT:   %inst: ref <error> = bind_name inst, %inst.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9: {} = struct_literal ()
// CHECK:STDOUT:   assign file.%inst.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
