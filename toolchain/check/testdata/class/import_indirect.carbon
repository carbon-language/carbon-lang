// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT: impl @C.as.Destroy.impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of file.%b_val.var [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.31f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.8a8: %ptr.as.Copy.impl.Op.type.31f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.999: <witness> = impl_witness imports.%Copy.impl_witness_table.53c, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.c3f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.fb8: %ptr.as.Copy.impl.Op.type.c3f = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.019, (%Copy.impl_witness.999) [concrete]
// CHECK:STDOUT:   %.7e9: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.fb8 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.fb8, @ptr.as.Copy.impl.Op(%C) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//a, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//a, loc4_10, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//a, inst18 [no loc], unloaded
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.0e4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.31f) = import_ref Core//prelude/parts/copy, loc36_31, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.8a8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.53c = impl_witness_table (%Core.import_ref.0e4), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %b_val.patt: %pattern_type.c48 = binding_pattern b_val [concrete]
// CHECK:STDOUT:     %b_val.var_patt: %pattern_type.c48 = var_pattern %b_val.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_val.var: ref %C = var %b_val.var_patt [concrete]
// CHECK:STDOUT:   %C.ref.loc8: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %b_val: ref %C = bind_name b_val, %b_val.var [concrete = %b_val.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_ptr.patt: %pattern_type.44a = binding_pattern b_ptr [concrete]
// CHECK:STDOUT:     %b_ptr.var_patt: %pattern_type.44a = var_pattern %b_ptr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_ptr.var: ref %ptr.019 = var %b_ptr.var_patt [concrete]
// CHECK:STDOUT:   %.loc9: type = splice_block %ptr [concrete = constants.%ptr.019] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, %D [concrete = constants.%C]
// CHECK:STDOUT:     %ptr: type = ptr_type %D.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_ptr: ref %ptr.019 = bind_name b_ptr, %b_ptr.var [concrete = %b_ptr.var]
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
// CHECK:STDOUT:   %b_val.ref: ref %C = name_ref b_val, file.%b_val [concrete = file.%b_val.var]
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %b_val.ref [concrete = constants.%addr]
// CHECK:STDOUT:   %impl.elem0: %.7e9 = impl_witness_access constants.%Copy.impl_witness.999, element0 [concrete = constants.%ptr.as.Copy.impl.Op.fb8]
// CHECK:STDOUT:   %bound_method.loc9_17.1: <bound method> = bound_method %addr, %impl.elem0 [concrete = constants.%ptr.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%C) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_17.2: <bound method> = bound_method %addr, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.019 = call %bound_method.loc9_17.2(%addr) [concrete = constants.%addr]
// CHECK:STDOUT:   assign file.%b_ptr.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- c.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of file.%c_val.var [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.31f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.8a8: %ptr.as.Copy.impl.Op.type.31f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.999: <witness> = impl_witness imports.%Copy.impl_witness_table.53c, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.c3f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.fb8: %ptr.as.Copy.impl.Op.type.c3f = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.019, (%Copy.impl_witness.999) [concrete]
// CHECK:STDOUT:   %.7e9: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.fb8 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.fb8, @ptr.as.Copy.impl.Op(%C) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//a, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//a, loc4_10, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//a, inst18 [no loc], unloaded
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.0e4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.31f) = import_ref Core//prelude/parts/copy, loc36_31, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.8a8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.53c = impl_witness_table (%Core.import_ref.0e4), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %c_val.patt: %pattern_type.c48 = binding_pattern c_val [concrete]
// CHECK:STDOUT:     %c_val.var_patt: %pattern_type.c48 = var_pattern %c_val.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_val.var: ref %C = var %c_val.var_patt [concrete]
// CHECK:STDOUT:   %C.ref.loc8: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %c_val: ref %C = bind_name c_val, %c_val.var [concrete = %c_val.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c_ptr.patt: %pattern_type.44a = binding_pattern c_ptr [concrete]
// CHECK:STDOUT:     %c_ptr.var_patt: %pattern_type.44a = var_pattern %c_ptr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_ptr.var: ref %ptr.019 = var %c_ptr.var_patt [concrete]
// CHECK:STDOUT:   %.loc9: type = splice_block %ptr [concrete = constants.%ptr.019] {
// CHECK:STDOUT:     %E.ref: type = name_ref E, %E [concrete = constants.%C]
// CHECK:STDOUT:     %ptr: type = ptr_type %E.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_ptr: ref %ptr.019 = bind_name c_ptr, %c_ptr.var [concrete = %c_ptr.var]
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
// CHECK:STDOUT:   %c_val.ref: ref %C = name_ref c_val, file.%c_val [concrete = file.%c_val.var]
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %c_val.ref [concrete = constants.%addr]
// CHECK:STDOUT:   %impl.elem0: %.7e9 = impl_witness_access constants.%Copy.impl_witness.999, element0 [concrete = constants.%ptr.as.Copy.impl.Op.fb8]
// CHECK:STDOUT:   %bound_method.loc9_17.1: <bound method> = bound_method %addr, %impl.elem0 [concrete = constants.%ptr.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%C) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_17.2: <bound method> = bound_method %addr, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.019 = call %bound_method.loc9_17.2(%addr) [concrete = constants.%addr]
// CHECK:STDOUT:   assign file.%c_ptr.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- triangle.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of file.%val.var [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.31f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.8a8: %ptr.as.Copy.impl.Op.type.31f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.999: <witness> = impl_witness imports.%Copy.impl_witness_table.53c, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.c3f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.fb8: %ptr.as.Copy.impl.Op.type.c3f = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.019, (%Copy.impl_witness.999) [concrete]
// CHECK:STDOUT:   %.7e9: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.fb8 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.fb8, @ptr.as.Copy.impl.Op(%C) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//a, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.D: type = import_ref Main//b, D, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.b_val = import_ref Main//b, b_val, unloaded
// CHECK:STDOUT:   %Main.b_ptr = import_ref Main//b, b_ptr, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//a, loc4_10, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//a, inst18 [no loc], unloaded
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.0e4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.31f) = import_ref Core//prelude/parts/copy, loc36_31, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.8a8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.53c = impl_witness_table (%Core.import_ref.0e4), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %val.patt: %pattern_type.c48 = binding_pattern val [concrete]
// CHECK:STDOUT:     %val.var_patt: %pattern_type.c48 = var_pattern %val.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %val.var: ref %C = var %val.var_patt [concrete]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %val: ref %C = bind_name val, %val.var [concrete = %val.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ptr.patt: %pattern_type.44a = binding_pattern ptr [concrete]
// CHECK:STDOUT:     %ptr.var_patt: %pattern_type.44a = var_pattern %ptr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.var: ref %ptr.019 = var %ptr.var_patt [concrete]
// CHECK:STDOUT:   %.loc8: type = splice_block %ptr.loc8_11 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc8_11: type = ptr_type %D.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.loc8_5: ref %ptr.019 = bind_name ptr, %ptr.var [concrete = %ptr.var]
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
// CHECK:STDOUT:   %val.ref: ref %C = name_ref val, file.%val [concrete = file.%val.var]
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %val.ref [concrete = constants.%addr]
// CHECK:STDOUT:   %impl.elem0: %.7e9 = impl_witness_access constants.%Copy.impl_witness.999, element0 [concrete = constants.%ptr.as.Copy.impl.Op.fb8]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %addr, %impl.elem0 [concrete = constants.%ptr.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%C) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %addr, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.019 = call %bound_method.loc8_15.2(%addr) [concrete = constants.%addr]
// CHECK:STDOUT:   assign file.%ptr.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- triangle_reverse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of file.%val.var [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.31f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.8a8: %ptr.as.Copy.impl.Op.type.31f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.999: <witness> = impl_witness imports.%Copy.impl_witness_table.53c, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.c3f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.fb8: %ptr.as.Copy.impl.Op.type.c3f = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.019, (%Copy.impl_witness.999) [concrete]
// CHECK:STDOUT:   %.7e9: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.fb8 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.fb8, @ptr.as.Copy.impl.Op(%C) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.D: type = import_ref Main//b, D, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.b_val = import_ref Main//b, b_val, unloaded
// CHECK:STDOUT:   %Main.b_ptr = import_ref Main//b, b_ptr, unloaded
// CHECK:STDOUT:   %Main.C: type = import_ref Main//a, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//a, loc4_10, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//a, inst18 [no loc], unloaded
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.0e4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.31f) = import_ref Core//prelude/parts/copy, loc36_31, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.8a8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.53c = impl_witness_table (%Core.import_ref.0e4), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %val.patt: %pattern_type.c48 = binding_pattern val [concrete]
// CHECK:STDOUT:     %val.var_patt: %pattern_type.c48 = var_pattern %val.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %val.var: ref %C = var %val.var_patt [concrete]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %val: ref %C = bind_name val, %val.var [concrete = %val.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ptr.patt: %pattern_type.44a = binding_pattern ptr [concrete]
// CHECK:STDOUT:     %ptr.var_patt: %pattern_type.44a = var_pattern %ptr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.var: ref %ptr.019 = var %ptr.var_patt [concrete]
// CHECK:STDOUT:   %.loc8: type = splice_block %ptr.loc8_11 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc8_11: type = ptr_type %D.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.loc8_5: ref %ptr.019 = bind_name ptr, %ptr.var [concrete = %ptr.var]
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
// CHECK:STDOUT:   %val.ref: ref %C = name_ref val, file.%val [concrete = file.%val.var]
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %val.ref [concrete = constants.%addr]
// CHECK:STDOUT:   %impl.elem0: %.7e9 = impl_witness_access constants.%Copy.impl_witness.999, element0 [concrete = constants.%ptr.as.Copy.impl.Op.fb8]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %addr, %impl.elem0 [concrete = constants.%ptr.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%C) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %addr, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.019 = call %bound_method.loc8_15.2(%addr) [concrete = constants.%addr]
// CHECK:STDOUT:   assign file.%ptr.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- diamond.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of file.%val.var [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.31f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.8a8: %ptr.as.Copy.impl.Op.type.31f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.999: <witness> = impl_witness imports.%Copy.impl_witness_table.53c, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.c3f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.fb8: %ptr.as.Copy.impl.Op.type.c3f = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.019, (%Copy.impl_witness.999) [concrete]
// CHECK:STDOUT:   %.7e9: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.fb8 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.fb8, @ptr.as.Copy.impl.Op(%C) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.specific_fn [concrete]
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
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8db: <witness> = import_ref Main//b, inst23 [indirect], loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.6a9 = import_ref Main//b, inst24 [indirect], unloaded
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.0e4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.31f) = import_ref Core//prelude/parts/copy, loc36_31, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.8a8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.53c = impl_witness_table (%Core.import_ref.0e4), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %val.patt: %pattern_type.c48 = binding_pattern val [concrete]
// CHECK:STDOUT:     %val.var_patt: %pattern_type.c48 = var_pattern %val.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %val.var: ref %C = var %val.var_patt [concrete]
// CHECK:STDOUT:   %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:   %val: ref %C = bind_name val, %val.var [concrete = %val.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ptr.patt: %pattern_type.44a = binding_pattern ptr [concrete]
// CHECK:STDOUT:     %ptr.var_patt: %pattern_type.44a = var_pattern %ptr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.var: ref %ptr.019 = var %ptr.var_patt [concrete]
// CHECK:STDOUT:   %.loc8: type = splice_block %ptr.loc8_11 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:     %E.ref: type = name_ref E, imports.%Main.E [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc8_11: type = ptr_type %E.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.loc8_5: ref %ptr.019 = bind_name ptr, %ptr.var [concrete = %ptr.var]
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
// CHECK:STDOUT:   %val.ref: ref %C = name_ref val, file.%val [concrete = file.%val.var]
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %val.ref [concrete = constants.%addr]
// CHECK:STDOUT:   %impl.elem0: %.7e9 = impl_witness_access constants.%Copy.impl_witness.999, element0 [concrete = constants.%ptr.as.Copy.impl.Op.fb8]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %addr, %impl.elem0 [concrete = constants.%ptr.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%C) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %addr, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.019 = call %bound_method.loc8_15.2(%addr) [concrete = constants.%addr]
// CHECK:STDOUT:   assign file.%ptr.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- diamond_reverse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of file.%val.var [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.31f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.8a8: %ptr.as.Copy.impl.Op.type.31f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.999: <witness> = impl_witness imports.%Copy.impl_witness_table.53c, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.c3f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.fb8: %ptr.as.Copy.impl.Op.type.c3f = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.019, (%Copy.impl_witness.999) [concrete]
// CHECK:STDOUT:   %.7e9: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.fb8 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.fb8, @ptr.as.Copy.impl.Op(%C) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.specific_fn [concrete]
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
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8db: <witness> = import_ref Main//b, inst23 [indirect], loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.6a9 = import_ref Main//b, inst24 [indirect], unloaded
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.0e4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.31f) = import_ref Core//prelude/parts/copy, loc36_31, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.8a8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.53c = impl_witness_table (%Core.import_ref.0e4), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %val.patt: %pattern_type.c48 = binding_pattern val [concrete]
// CHECK:STDOUT:     %val.var_patt: %pattern_type.c48 = var_pattern %val.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %val.var: ref %C = var %val.var_patt [concrete]
// CHECK:STDOUT:   %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%C]
// CHECK:STDOUT:   %val: ref %C = bind_name val, %val.var [concrete = %val.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ptr.patt: %pattern_type.44a = binding_pattern ptr [concrete]
// CHECK:STDOUT:     %ptr.var_patt: %pattern_type.44a = var_pattern %ptr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.var: ref %ptr.019 = var %ptr.var_patt [concrete]
// CHECK:STDOUT:   %.loc8: type = splice_block %ptr.loc8_11 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:     %E.ref: type = name_ref E, imports.%Main.E [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc8_11: type = ptr_type %E.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.loc8_5: ref %ptr.019 = bind_name ptr, %ptr.var [concrete = %ptr.var]
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
// CHECK:STDOUT:   %val.ref: ref %C = name_ref val, file.%val [concrete = file.%val.var]
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %val.ref [concrete = constants.%addr]
// CHECK:STDOUT:   %impl.elem0: %.7e9 = impl_witness_access constants.%Copy.impl_witness.999, element0 [concrete = constants.%ptr.as.Copy.impl.Op.fb8]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %addr, %impl.elem0 [concrete = constants.%ptr.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%C) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %addr, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.019 = call %bound_method.loc8_15.2(%addr) [concrete = constants.%addr]
// CHECK:STDOUT:   assign file.%ptr.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
