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

package Test library "[[@TEST_NAME]]";

class C {}
private alias A = C;

// ============================================================================
// Test files
// ============================================================================

// --- def.impl.carbon

impl package Test library "[[@TEST_NAME]]";

var inst: A = {};

// --- fail_local_def.carbon

package Test library "[[@TEST_NAME]]";

import library "def";

// CHECK:STDERR: fail_local_def.carbon:[[@LINE+4]]:11: error: name `A` not found [NameNotFound]
// CHECK:STDERR: var inst: A = {};
// CHECK:STDERR:           ^
// CHECK:STDERR:
var inst: A = {};

// --- fail_other_def.carbon

package Other library "[[@TEST_NAME]]";

import Test library "def";

// CHECK:STDERR: fail_other_def.carbon:[[@LINE+4]]:11: error: member name `A` not found in `Test` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: var inst: Test.A = {};
// CHECK:STDERR:           ^~~~~~
// CHECK:STDERR:
var inst: Test.A = {};

// CHECK:STDOUT: --- def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .A [private] = %A
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %A: type = bind_alias A, %C.decl [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- def.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Test.C = import_ref Test//def, C, unloaded
// CHECK:STDOUT:   %Test.A: type = import_ref Test//def, A, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Test.import_ref.8f2: <witness> = import_ref Test//def, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Test.import_ref.2c4 = import_ref Test//def, inst15 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Test.C
// CHECK:STDOUT:     .A [private] = imports.%Test.A
// CHECK:STDOUT:     .inst = %inst
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %inst.patt: %C = binding_pattern inst
// CHECK:STDOUT:     %.loc4: %C = var_pattern %inst.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.var: ref %C = var inst
// CHECK:STDOUT:   %A.ref: type = name_ref A, imports.%Test.A [concrete = constants.%C]
// CHECK:STDOUT:   %inst: ref %C = bind_name inst, %inst.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "def.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Test.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Test.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_16.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc4_16.2: init %C = class_init (), file.%inst.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc4_1: init %C = converted %.loc4_16.1, %.loc4_16.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%inst.var, %.loc4_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_local_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Test.C = import_ref Test//def, C, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Test.C
// CHECK:STDOUT:     .A = <poisoned>
// CHECK:STDOUT:     .inst = %inst
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %inst.patt: <error> = binding_pattern inst
// CHECK:STDOUT:     %.loc10: <error> = var_pattern %inst.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.var: ref <error> = var inst
// CHECK:STDOUT:   %A.ref: <error> = name_ref A, <error> [concrete = <error>]
// CHECK:STDOUT:   %inst: <error> = bind_name inst, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   assign file.%inst.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_other_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Test: <namespace> = namespace file.%Test.import, [concrete] {
// CHECK:STDOUT:     .A = <poisoned>
// CHECK:STDOUT:     import Test//def
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Test = imports.%Test
// CHECK:STDOUT:     .inst = %inst
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.import = import Test
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %inst.patt: <error> = binding_pattern inst
// CHECK:STDOUT:     %.loc10: <error> = var_pattern %inst.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.var: ref <error> = var inst
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %Test.ref: <namespace> = name_ref Test, imports.%Test [concrete = imports.%Test]
// CHECK:STDOUT:     %A.ref: <error> = name_ref A, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst: <error> = bind_name inst, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   assign file.%inst.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
