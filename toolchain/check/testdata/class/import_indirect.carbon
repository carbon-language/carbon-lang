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
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.3b0: type = import_ref Main//a, C, loaded [template = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8f2: <witness> = import_ref Main//a, loc4_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.3b0
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .b_val = %b_val
// CHECK:STDOUT:     .b_ptr = %b_ptr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C.ref.loc6: type = name_ref C, imports.%import_ref.3b0 [template = constants.%C]
// CHECK:STDOUT:   %D: type = bind_alias D, imports.%import_ref.3b0 [template = constants.%C]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_val.patt: %C = binding_pattern b_val
// CHECK:STDOUT:     %.loc8: %C = var_pattern %b_val.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_val.var: ref %C = var b_val
// CHECK:STDOUT:   %C.ref.loc8: type = name_ref C, imports.%import_ref.3b0 [template = constants.%C]
// CHECK:STDOUT:   %b_val: ref %C = bind_name b_val, %b_val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_ptr.patt: %ptr.019 = binding_pattern b_ptr
// CHECK:STDOUT:     %.loc9_1: %ptr.019 = var_pattern %b_ptr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_ptr.var: ref %ptr.019 = var b_ptr
// CHECK:STDOUT:   %.loc9_13: type = splice_block %ptr [template = constants.%ptr.019] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, %D [template = constants.%C]
// CHECK:STDOUT:     %ptr: type = ptr_type %C [template = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_ptr: ref %ptr.019 = bind_name b_ptr, %b_ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_17.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_17.2: init %C = class_init (), file.%b_val.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc8_1: init %C = converted %.loc8_17.1, %.loc8_17.2 [template = constants.%C.val]
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
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.3b0: type = import_ref Main//a, C, loaded [template = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8f2: <witness> = import_ref Main//a, loc4_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.3b0
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .E = %E
// CHECK:STDOUT:     .c_val = %c_val
// CHECK:STDOUT:     .c_ptr = %c_ptr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C.ref.loc6: type = name_ref C, imports.%import_ref.3b0 [template = constants.%C]
// CHECK:STDOUT:   %E: type = bind_alias E, imports.%import_ref.3b0 [template = constants.%C]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c_val.patt: %C = binding_pattern c_val
// CHECK:STDOUT:     %.loc8: %C = var_pattern %c_val.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_val.var: ref %C = var c_val
// CHECK:STDOUT:   %C.ref.loc8: type = name_ref C, imports.%import_ref.3b0 [template = constants.%C]
// CHECK:STDOUT:   %c_val: ref %C = bind_name c_val, %c_val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c_ptr.patt: %ptr.019 = binding_pattern c_ptr
// CHECK:STDOUT:     %.loc9_1: %ptr.019 = var_pattern %c_ptr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_ptr.var: ref %ptr.019 = var c_ptr
// CHECK:STDOUT:   %.loc9_13: type = splice_block %ptr [template = constants.%ptr.019] {
// CHECK:STDOUT:     %E.ref: type = name_ref E, %E [template = constants.%C]
// CHECK:STDOUT:     %ptr: type = ptr_type %C [template = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_ptr: ref %ptr.019 = bind_name c_ptr, %c_ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_17.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_17.2: init %C = class_init (), file.%c_val.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc8_1: init %C = converted %.loc8_17.1, %.loc8_17.2 [template = constants.%C.val]
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
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.3b0: type = import_ref Main//a, C, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.ff6: type = import_ref Main//b, D, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.0da = import_ref Main//b, b_val, unloaded
// CHECK:STDOUT:   %import_ref.ff5 = import_ref Main//b, b_ptr, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8f2: <witness> = import_ref Main//a, loc4_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.3b0
// CHECK:STDOUT:     .D = imports.%import_ref.ff6
// CHECK:STDOUT:     .b_val = imports.%import_ref.0da
// CHECK:STDOUT:     .b_ptr = imports.%import_ref.ff5
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .val = %val
// CHECK:STDOUT:     .ptr = %ptr.loc8_5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %val.patt: %C = binding_pattern val
// CHECK:STDOUT:     %.loc7: %C = var_pattern %val.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %val.var: ref %C = var val
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.3b0 [template = constants.%C]
// CHECK:STDOUT:   %val: ref %C = bind_name val, %val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ptr.patt: %ptr.019 = binding_pattern ptr
// CHECK:STDOUT:     %.loc8_1: %ptr.019 = var_pattern %ptr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.var: ref %ptr.019 = var ptr
// CHECK:STDOUT:   %.loc8_11: type = splice_block %ptr.loc8_11 [template = constants.%ptr.019] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%import_ref.ff6 [template = constants.%C]
// CHECK:STDOUT:     %ptr.loc8_11: type = ptr_type %C [template = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.loc8_5: ref %ptr.019 = bind_name ptr, %ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_15.2: init %C = class_init (), file.%val.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc7_1: init %C = converted %.loc7_15.1, %.loc7_15.2 [template = constants.%C.val]
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
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.ff6: type = import_ref Main//b, D, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.0da = import_ref Main//b, b_val, unloaded
// CHECK:STDOUT:   %import_ref.ff5 = import_ref Main//b, b_ptr, unloaded
// CHECK:STDOUT:   %import_ref.3b0: type = import_ref Main//a, C, loaded [template = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8f2: <witness> = import_ref Main//a, loc4_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .D = imports.%import_ref.ff6
// CHECK:STDOUT:     .b_val = imports.%import_ref.0da
// CHECK:STDOUT:     .b_ptr = imports.%import_ref.ff5
// CHECK:STDOUT:     .C = imports.%import_ref.3b0
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .val = %val
// CHECK:STDOUT:     .ptr = %ptr.loc8_5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %val.patt: %C = binding_pattern val
// CHECK:STDOUT:     %.loc7: %C = var_pattern %val.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %val.var: ref %C = var val
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%import_ref.3b0 [template = constants.%C]
// CHECK:STDOUT:   %val: ref %C = bind_name val, %val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ptr.patt: %ptr.019 = binding_pattern ptr
// CHECK:STDOUT:     %.loc8_1: %ptr.019 = var_pattern %ptr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.var: ref %ptr.019 = var ptr
// CHECK:STDOUT:   %.loc8_11: type = splice_block %ptr.loc8_11 [template = constants.%ptr.019] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%import_ref.ff6 [template = constants.%C]
// CHECK:STDOUT:     %ptr.loc8_11: type = ptr_type %C [template = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.loc8_5: ref %ptr.019 = bind_name ptr, %ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "a.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_15.2: init %C = class_init (), file.%val.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc7_1: init %C = converted %.loc7_15.1, %.loc7_15.2 [template = constants.%C.val]
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
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.ff6: type = import_ref Main//b, D, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.0da = import_ref Main//b, b_val, unloaded
// CHECK:STDOUT:   %import_ref.ff5 = import_ref Main//b, b_ptr, unloaded
// CHECK:STDOUT:   %import_ref.2c9: type = import_ref Main//c, E, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.963 = import_ref Main//c, c_val, unloaded
// CHECK:STDOUT:   %import_ref.be6 = import_ref Main//c, c_ptr, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8db: <witness> = import_ref Main//b, inst21 [indirect], loaded [template = constants.%complete_type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .D = imports.%import_ref.ff6
// CHECK:STDOUT:     .b_val = imports.%import_ref.0da
// CHECK:STDOUT:     .b_ptr = imports.%import_ref.ff5
// CHECK:STDOUT:     .E = imports.%import_ref.2c9
// CHECK:STDOUT:     .c_val = imports.%import_ref.963
// CHECK:STDOUT:     .c_ptr = imports.%import_ref.be6
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .val = %val
// CHECK:STDOUT:     .ptr = %ptr.loc8_5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %val.patt: %C = binding_pattern val
// CHECK:STDOUT:     %.loc7: %C = var_pattern %val.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %val.var: ref %C = var val
// CHECK:STDOUT:   %D.ref: type = name_ref D, imports.%import_ref.ff6 [template = constants.%C]
// CHECK:STDOUT:   %val: ref %C = bind_name val, %val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ptr.patt: %ptr.019 = binding_pattern ptr
// CHECK:STDOUT:     %.loc8_1: %ptr.019 = var_pattern %ptr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.var: ref %ptr.019 = var ptr
// CHECK:STDOUT:   %.loc8_11: type = splice_block %ptr.loc8_11 [template = constants.%ptr.019] {
// CHECK:STDOUT:     %E.ref: type = name_ref E, imports.%import_ref.2c9 [template = constants.%C]
// CHECK:STDOUT:     %ptr.loc8_11: type = ptr_type %C [template = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.loc8_5: ref %ptr.019 = bind_name ptr, %ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "b.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8db
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_15.2: init %C = class_init (), file.%val.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc7_1: init %C = converted %.loc7_15.1, %.loc7_15.2 [template = constants.%C.val]
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
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %C.val: %C = struct_value () [template]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.2c9: type = import_ref Main//c, E, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.963 = import_ref Main//c, c_val, unloaded
// CHECK:STDOUT:   %import_ref.be6 = import_ref Main//c, c_ptr, unloaded
// CHECK:STDOUT:   %import_ref.ff6: type = import_ref Main//b, D, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.0da = import_ref Main//b, b_val, unloaded
// CHECK:STDOUT:   %import_ref.ff5 = import_ref Main//b, b_ptr, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8db: <witness> = import_ref Main//b, inst21 [indirect], loaded [template = constants.%complete_type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .E = imports.%import_ref.2c9
// CHECK:STDOUT:     .c_val = imports.%import_ref.963
// CHECK:STDOUT:     .c_ptr = imports.%import_ref.be6
// CHECK:STDOUT:     .D = imports.%import_ref.ff6
// CHECK:STDOUT:     .b_val = imports.%import_ref.0da
// CHECK:STDOUT:     .b_ptr = imports.%import_ref.ff5
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .val = %val
// CHECK:STDOUT:     .ptr = %ptr.loc8_5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %val.patt: %C = binding_pattern val
// CHECK:STDOUT:     %.loc7: %C = var_pattern %val.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %val.var: ref %C = var val
// CHECK:STDOUT:   %D.ref: type = name_ref D, imports.%import_ref.ff6 [template = constants.%C]
// CHECK:STDOUT:   %val: ref %C = bind_name val, %val.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ptr.patt: %ptr.019 = binding_pattern ptr
// CHECK:STDOUT:     %.loc8_1: %ptr.019 = var_pattern %ptr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.var: ref %ptr.019 = var ptr
// CHECK:STDOUT:   %.loc8_11: type = splice_block %ptr.loc8_11 [template = constants.%ptr.019] {
// CHECK:STDOUT:     %E.ref: type = name_ref E, imports.%import_ref.2c9 [template = constants.%C]
// CHECK:STDOUT:     %ptr.loc8_11: type = ptr_type %C [template = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.loc8_5: ref %ptr.019 = bind_name ptr, %ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "b.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8db
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_15.2: init %C = class_init (), file.%val.var [template = constants.%C.val]
// CHECK:STDOUT:   %.loc7_1: init %C = converted %.loc7_15.1, %.loc7_15.2 [template = constants.%C.val]
// CHECK:STDOUT:   assign file.%val.var, %.loc7_1
// CHECK:STDOUT:   %val.ref: ref %C = name_ref val, file.%val
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %val.ref
// CHECK:STDOUT:   assign file.%ptr.var, %addr
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
