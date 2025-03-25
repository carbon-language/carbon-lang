// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/import_indirect.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/import_indirect.carbon

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

var val: C = {};
var ptr: D* = &val;

// --- triangle_reverse.carbon

library "[[@TEST_NAME]]";

import library "b";
import library "a";

var val: C = {};
var ptr: D* = &val;

// --- diamond.carbon

library "[[@TEST_NAME]]";

import library "b";
import library "c";

var val: D = {};
var ptr: E* = &val;

// --- diamond_reverse.carbon

library "[[@TEST_NAME]]";

import library "c";
import library "b";

var val: D = {};
var ptr: E* = &val;

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
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
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//a, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//a, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//a, inst17 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .b_val = %b_val
// CHECK:STDOUT:     .b_ptr = %b_ptr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C.ref.loc6: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %D: type = bind_alias D, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_val.patt: %C = binding_pattern b_val
// CHECK:STDOUT:     %.loc8: %C = var_pattern %b_val.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_val.var: ref %C = var b_val
// CHECK:STDOUT:   %C.ref.loc8: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %b_val: ref %C = bind_name b_val, %b_val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_ptr.patt: %ptr.019 = binding_pattern b_ptr
// CHECK:STDOUT:     %.loc9_1: %ptr.019 = var_pattern %b_ptr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_ptr.var: ref %ptr.019 = var b_ptr
// CHECK:STDOUT:   %.loc9_13: type = splice_block %ptr [concrete = constants.%ptr.019] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, %D [concrete = constants.%C]
// CHECK:STDOUT:     %ptr: type = ptr_type %C [concrete = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_ptr: ref %ptr.019 = bind_name b_ptr, %b_ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_17.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_17.2: init %C = class_init (), file.%b_val.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_1: init %C = converted %.loc8_17.1, %.loc8_17.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%b_val.var, %.loc8_1
// CHECK:STDOUT:   %b_val.ref: ref %C = name_ref b_val, file.%b_val
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %b_val.ref
// CHECK:STDOUT:   assign file.%b_ptr.var, %addr
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- c.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//a, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//a, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//a, inst17 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .E = %E
// CHECK:STDOUT:     .c_val = %c_val
// CHECK:STDOUT:     .c_ptr = %c_ptr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C.ref.loc6: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %E: type = bind_alias E, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c_val.patt: %C = binding_pattern c_val
// CHECK:STDOUT:     %.loc8: %C = var_pattern %c_val.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_val.var: ref %C = var c_val
// CHECK:STDOUT:   %C.ref.loc8: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c_val: ref %C = bind_name c_val, %c_val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c_ptr.patt: %ptr.019 = binding_pattern c_ptr
// CHECK:STDOUT:     %.loc9_1: %ptr.019 = var_pattern %c_ptr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_ptr.var: ref %ptr.019 = var c_ptr
// CHECK:STDOUT:   %.loc9_13: type = splice_block %ptr [concrete = constants.%ptr.019] {
// CHECK:STDOUT:     %E.ref: type = name_ref E, %E [concrete = constants.%C]
// CHECK:STDOUT:     %ptr: type = ptr_type %C [concrete = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_ptr: ref %ptr.019 = bind_name c_ptr, %c_ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_17.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_17.2: init %C = class_init (), file.%c_val.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc8_1: init %C = converted %.loc8_17.1, %.loc8_17.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%c_val.var, %.loc8_1
// CHECK:STDOUT:   %c_val.ref: ref %C = name_ref c_val, file.%c_val
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %c_val.ref
// CHECK:STDOUT:   assign file.%c_ptr.var, %addr
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- triangle.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//a, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.D: type = import_ref Main//b, D, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.b_val = import_ref Main//b, b_val, unloaded
// CHECK:STDOUT:   %Main.b_ptr = import_ref Main//b, b_ptr, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//a, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//a, inst17 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .b_val = imports.%Main.b_val
// CHECK:STDOUT:     .b_ptr = imports.%Main.b_ptr
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .val = %val
// CHECK:STDOUT:     .ptr = %ptr.loc8_5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %val.patt: %C = binding_pattern val
// CHECK:STDOUT:     %.loc7: %C = var_pattern %val.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %val.var: ref %C = var val
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %val: ref %C = bind_name val, %val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ptr.patt: %ptr.019 = binding_pattern ptr
// CHECK:STDOUT:     %.loc8_1: %ptr.019 = var_pattern %ptr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.var: ref %ptr.019 = var ptr
// CHECK:STDOUT:   %.loc8_11: type = splice_block %ptr.loc8_11 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc8_11: type = ptr_type %C [concrete = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.loc8_5: ref %ptr.019 = bind_name ptr, %ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_15.2: init %C = class_init (), file.%val.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc7_1: init %C = converted %.loc7_15.1, %.loc7_15.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%val.var, %.loc7_1
// CHECK:STDOUT:   %val.ref: ref %C = name_ref val, file.%val
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %val.ref
// CHECK:STDOUT:   assign file.%ptr.var, %addr
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- triangle_reverse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.D: type = import_ref Main//b, D, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.b_val = import_ref Main//b, b_val, unloaded
// CHECK:STDOUT:   %Main.b_ptr = import_ref Main//b, b_ptr, unloaded
// CHECK:STDOUT:   %Main.C: type = import_ref Main//a, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//a, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//a, inst17 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .b_val = imports.%Main.b_val
// CHECK:STDOUT:     .b_ptr = imports.%Main.b_ptr
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .val = %val
// CHECK:STDOUT:     .ptr = %ptr.loc8_5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %val.patt: %C = binding_pattern val
// CHECK:STDOUT:     %.loc7: %C = var_pattern %val.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %val.var: ref %C = var val
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %val: ref %C = bind_name val, %val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ptr.patt: %ptr.019 = binding_pattern ptr
// CHECK:STDOUT:     %.loc8_1: %ptr.019 = var_pattern %ptr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.var: ref %ptr.019 = var ptr
// CHECK:STDOUT:   %.loc8_11: type = splice_block %ptr.loc8_11 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc8_11: type = ptr_type %C [concrete = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.loc8_5: ref %ptr.019 = bind_name ptr, %ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_15.2: init %C = class_init (), file.%val.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc7_1: init %C = converted %.loc7_15.1, %.loc7_15.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%val.var, %.loc7_1
// CHECK:STDOUT:   %val.ref: ref %C = name_ref val, file.%val
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %val.ref
// CHECK:STDOUT:   assign file.%ptr.var, %addr
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- diamond.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.D: type = import_ref Main//b, D, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.b_val = import_ref Main//b, b_val, unloaded
// CHECK:STDOUT:   %Main.b_ptr = import_ref Main//b, b_ptr, unloaded
// CHECK:STDOUT:   %Main.E: type = import_ref Main//c, E, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.c_val = import_ref Main//c, c_val, unloaded
// CHECK:STDOUT:   %Main.c_ptr = import_ref Main//c, c_ptr, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8db: <witness> = import_ref Main//b, inst22 [indirect], loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.6a9 = import_ref Main//b, inst23 [indirect], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .b_val = imports.%Main.b_val
// CHECK:STDOUT:     .b_ptr = imports.%Main.b_ptr
// CHECK:STDOUT:     .E = imports.%Main.E
// CHECK:STDOUT:     .c_val = imports.%Main.c_val
// CHECK:STDOUT:     .c_ptr = imports.%Main.c_ptr
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .val = %val
// CHECK:STDOUT:     .ptr = %ptr.loc8_5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %val.patt: %C = binding_pattern val
// CHECK:STDOUT:     %.loc7: %C = var_pattern %val.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %val.var: ref %C = var val
// CHECK:STDOUT:   %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:   %val: ref %C = bind_name val, %val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ptr.patt: %ptr.019 = binding_pattern ptr
// CHECK:STDOUT:     %.loc8_1: %ptr.019 = var_pattern %ptr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.var: ref %ptr.019 = var ptr
// CHECK:STDOUT:   %.loc8_11: type = splice_block %ptr.loc8_11 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:     %E.ref: type = name_ref E, imports.%Main.E [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc8_11: type = ptr_type %C [concrete = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.loc8_5: ref %ptr.019 = bind_name ptr, %ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "b.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8db
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.6a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_15.2: init %C = class_init (), file.%val.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc7_1: init %C = converted %.loc7_15.1, %.loc7_15.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%val.var, %.loc7_1
// CHECK:STDOUT:   %val.ref: ref %C = name_ref val, file.%val
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %val.ref
// CHECK:STDOUT:   assign file.%ptr.var, %addr
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- diamond_reverse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.E: type = import_ref Main//c, E, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.c_val = import_ref Main//c, c_val, unloaded
// CHECK:STDOUT:   %Main.c_ptr = import_ref Main//c, c_ptr, unloaded
// CHECK:STDOUT:   %Main.D: type = import_ref Main//b, D, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.b_val = import_ref Main//b, b_val, unloaded
// CHECK:STDOUT:   %Main.b_ptr = import_ref Main//b, b_ptr, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8db: <witness> = import_ref Main//b, inst22 [indirect], loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.6a9 = import_ref Main//b, inst23 [indirect], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .E = imports.%Main.E
// CHECK:STDOUT:     .c_val = imports.%Main.c_val
// CHECK:STDOUT:     .c_ptr = imports.%Main.c_ptr
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .b_val = imports.%Main.b_val
// CHECK:STDOUT:     .b_ptr = imports.%Main.b_ptr
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .val = %val
// CHECK:STDOUT:     .ptr = %ptr.loc8_5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %val.patt: %C = binding_pattern val
// CHECK:STDOUT:     %.loc7: %C = var_pattern %val.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %val.var: ref %C = var val
// CHECK:STDOUT:   %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:   %val: ref %C = bind_name val, %val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ptr.patt: %ptr.019 = binding_pattern ptr
// CHECK:STDOUT:     %.loc8_1: %ptr.019 = var_pattern %ptr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.var: ref %ptr.019 = var ptr
// CHECK:STDOUT:   %.loc8_11: type = splice_block %ptr.loc8_11 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:     %E.ref: type = name_ref E, imports.%Main.E [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc8_11: type = ptr_type %C [concrete = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.loc8_5: ref %ptr.019 = bind_name ptr, %ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "b.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8db
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.6a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_15.2: init %C = class_init (), file.%val.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc7_1: init %C = converted %.loc7_15.1, %.loc7_15.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign file.%val.var, %.loc7_1
// CHECK:STDOUT:   %val.ref: ref %C = name_ref val, file.%val
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %val.ref
// CHECK:STDOUT:   assign file.%ptr.var, %addr
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
