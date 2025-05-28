// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/import_indirect.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/import_indirect.carbon

// Triangle:
//
//     a    <-- Define
//     |\
//     | b  <-- Re-export
//     |/
//     X    <-- Use both
//
// Diamond:
//
//     a    <-- Define
//    / \
//   b   c  <-- Re-export
//    \ /
//     X    <-- Use both

// ============================================================================
// Setup files
// ============================================================================

// --- a.carbon

library "[[@TEST_NAME]]";

class C {}

// --- b.carbon

library "[[@TEST_NAME]]";

import library "a";

alias D = C;

var b_val: C = {};
var b_ptr: D* = &b_val;

// --- c.carbon

library "[[@TEST_NAME]]";

import library "a";

alias E = C;

var c_val: C = {};
var c_ptr: E* = &c_val;

// ============================================================================
// Test files
// ============================================================================

// --- triangle.carbon

library "[[@TEST_NAME]]";

import library "a";
import library "b";

//@dump-sem-ir-begin
var val: C = {};
var ptr: D* = &val;
//@dump-sem-ir-end

// --- triangle_reverse.carbon

library "[[@TEST_NAME]]";

import library "b";
import library "a";

//@dump-sem-ir-begin
var val: C = {};
var ptr: D* = &val;
//@dump-sem-ir-end

// --- diamond.carbon

library "[[@TEST_NAME]]";

import library "b";
import library "c";

//@dump-sem-ir-begin
var val: D = {};
var ptr: E* = &val;
//@dump-sem-ir-end

// --- diamond_reverse.carbon

library "[[@TEST_NAME]]";

import library "c";
import library "b";

//@dump-sem-ir-begin
var val: D = {};
var ptr: E* = &val;
//@dump-sem-ir-end

// CHECK:STDOUT: --- triangle.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//a, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.D: type = import_ref Main//b, D, loaded [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %val.patt: %pattern_type.c48 = binding_pattern val [concrete]
// CHECK:STDOUT:     %val.var_patt: %pattern_type.c48 = var_pattern %val.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %val.var: ref %C = var %val.var_patt [concrete]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %val: ref %C = bind_name val, %val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ptr.patt: %pattern_type.44a = binding_pattern ptr [concrete]
// CHECK:STDOUT:     %ptr.var_patt: %pattern_type.44a = var_pattern %ptr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.var: ref %ptr.019 = var %ptr.var_patt [concrete]
// CHECK:STDOUT:   %.loc9: type = splice_block %ptr.loc9_11 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc9_11: type = ptr_type %D.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.loc9_5: ref %ptr.019 = bind_name ptr, %ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_15.2: init %C = class_init (), file.%val.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_1: init %C = converted %.loc8_15.1, %.loc8_15.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%val.var, %.loc8_1
// CHECK:STDOUT:   %val.ref: ref %C = name_ref val, file.%val
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %val.ref
// CHECK:STDOUT:   assign file.%ptr.var, %addr
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- triangle_reverse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.D: type = import_ref Main//b, D, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//a, C, loaded [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %val.patt: %pattern_type.c48 = binding_pattern val [concrete]
// CHECK:STDOUT:     %val.var_patt: %pattern_type.c48 = var_pattern %val.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %val.var: ref %C = var %val.var_patt [concrete]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %val: ref %C = bind_name val, %val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ptr.patt: %pattern_type.44a = binding_pattern ptr [concrete]
// CHECK:STDOUT:     %ptr.var_patt: %pattern_type.44a = var_pattern %ptr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.var: ref %ptr.019 = var %ptr.var_patt [concrete]
// CHECK:STDOUT:   %.loc9: type = splice_block %ptr.loc9_11 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc9_11: type = ptr_type %D.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.loc9_5: ref %ptr.019 = bind_name ptr, %ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_15.2: init %C = class_init (), file.%val.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_1: init %C = converted %.loc8_15.1, %.loc8_15.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%val.var, %.loc8_1
// CHECK:STDOUT:   %val.ref: ref %C = name_ref val, file.%val
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %val.ref
// CHECK:STDOUT:   assign file.%ptr.var, %addr
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- diamond.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.D: type = import_ref Main//b, D, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.E: type = import_ref Main//c, E, loaded [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %val.patt: %pattern_type.c48 = binding_pattern val [concrete]
// CHECK:STDOUT:     %val.var_patt: %pattern_type.c48 = var_pattern %val.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %val.var: ref %C = var %val.var_patt [concrete]
// CHECK:STDOUT:   %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:   %val: ref %C = bind_name val, %val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ptr.patt: %pattern_type.44a = binding_pattern ptr [concrete]
// CHECK:STDOUT:     %ptr.var_patt: %pattern_type.44a = var_pattern %ptr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.var: ref %ptr.019 = var %ptr.var_patt [concrete]
// CHECK:STDOUT:   %.loc9: type = splice_block %ptr.loc9_11 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:     %E.ref: type = name_ref E, imports.%Main.E [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc9_11: type = ptr_type %E.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.loc9_5: ref %ptr.019 = bind_name ptr, %ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_15.2: init %C = class_init (), file.%val.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_1: init %C = converted %.loc8_15.1, %.loc8_15.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%val.var, %.loc8_1
// CHECK:STDOUT:   %val.ref: ref %C = name_ref val, file.%val
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %val.ref
// CHECK:STDOUT:   assign file.%ptr.var, %addr
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- diamond_reverse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.E: type = import_ref Main//c, E, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.D: type = import_ref Main//b, D, loaded [concrete = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %val.patt: %pattern_type.c48 = binding_pattern val [concrete]
// CHECK:STDOUT:     %val.var_patt: %pattern_type.c48 = var_pattern %val.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %val.var: ref %C = var %val.var_patt [concrete]
// CHECK:STDOUT:   %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:   %val: ref %C = bind_name val, %val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ptr.patt: %pattern_type.44a = binding_pattern ptr [concrete]
// CHECK:STDOUT:     %ptr.var_patt: %pattern_type.44a = var_pattern %ptr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.var: ref %ptr.019 = var %ptr.var_patt [concrete]
// CHECK:STDOUT:   %.loc9: type = splice_block %ptr.loc9_11 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:     %E.ref: type = name_ref E, imports.%Main.E [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc9_11: type = ptr_type %E.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.loc9_5: ref %ptr.019 = bind_name ptr, %ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_15.2: init %C = class_init (), file.%val.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_1: init %C = converted %.loc8_15.1, %.loc8_15.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%val.var, %.loc8_1
// CHECK:STDOUT:   %val.ref: ref %C = name_ref val, file.%val
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %val.ref
// CHECK:STDOUT:   assign file.%ptr.var, %addr
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
