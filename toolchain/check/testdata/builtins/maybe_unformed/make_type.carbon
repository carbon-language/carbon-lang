// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/maybe_unformed/make_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/maybe_unformed/make_type.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

fn Make(t: type) -> type = "maybe_unformed.make_type";

// --- use_types.carbon

library "[[@TEST_NAME]]";

import library "types";

//@dump-sem-ir-begin
fn F(unused b: Make(Make({}))) {}
//@dump-sem-ir-end

// --- runtime_call.carbon

library "[[@TEST_NAME]]";

import library "types";

//@dump-sem-ir-begin
let t: type = {};
let u: type = Make(t);
//@dump-sem-ir-end

// --- fail_bad_decl.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "maybe_unformed.make_type" [InvalidBuiltinSignature]
// CHECK:STDERR: fn NoParam() -> type = "maybe_unformed.make_type";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NoParam() -> type = "maybe_unformed.make_type";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "maybe_unformed.make_type" [InvalidBuiltinSignature]
// CHECK:STDERR: fn ColonBangParam(T:! type) -> type = "maybe_unformed.make_type";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn ColonBangParam(T:! type) -> type = "maybe_unformed.make_type";

// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.a8d: type = maybe_unformed_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %.2ba: type = maybe_unformed_type %.a8d [concrete]
// CHECK:STDOUT:   %pattern_type.ce6: type = pattern_type %.2ba [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Make: %Make.type = import_ref Main//types, Make, loaded [concrete = constants.%Make]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %b.param_patt: %pattern_type.ce6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.ce6 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %.2ba = value_param call_param0
// CHECK:STDOUT:     %.loc7_29.1: type = splice_block %.loc7_29.3 [concrete = constants.%.2ba] {
// CHECK:STDOUT:       %Make.ref.loc7_16: %Make.type = name_ref Make, imports.%Main.Make [concrete = constants.%Make]
// CHECK:STDOUT:       %Make.ref.loc7_21: %Make.type = name_ref Make, imports.%Main.Make [concrete = constants.%Make]
// CHECK:STDOUT:       %.loc7_27.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc7_27.2: type = converted %.loc7_27.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %Make.call.loc7_28: init type = call %Make.ref.loc7_21(%.loc7_27.2) [concrete = constants.%.a8d]
// CHECK:STDOUT:       %.loc7_28.1: type = value_of_initializer %Make.call.loc7_28 [concrete = constants.%.a8d]
// CHECK:STDOUT:       %.loc7_28.2: type = converted %Make.call.loc7_28, %.loc7_28.1 [concrete = constants.%.a8d]
// CHECK:STDOUT:       %Make.call.loc7_29: init type = call %Make.ref.loc7_16(%.loc7_28.2) [concrete = constants.%.2ba]
// CHECK:STDOUT:       %.loc7_29.2: type = value_of_initializer %Make.call.loc7_29 [concrete = constants.%.2ba]
// CHECK:STDOUT:       %.loc7_29.3: type = converted %Make.call.loc7_29, %.loc7_29.2 [concrete = constants.%.2ba]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %.2ba = value_binding b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b.param: %.2ba) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- runtime_call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Make: %Make.type = import_ref Main//types, Make, loaded [concrete = constants.%Make]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type = value_binding_pattern t [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_8: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %.loc7_16: type = converted @__global_init.%.loc7, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %t: type = value_binding t, %.loc7_16
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %u.patt: %pattern_type = value_binding_pattern u [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_8: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %.loc8_21.1: type = value_of_initializer @__global_init.%Make.call
// CHECK:STDOUT:   %.loc8_21.2: type = converted @__global_init.%Make.call, %.loc8_21.1
// CHECK:STDOUT:   %u: type = value_binding u, %.loc8_21.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, imports.%Main.Make [concrete = constants.%Make]
// CHECK:STDOUT:   %t.ref: type = name_ref t, file.%t
// CHECK:STDOUT:   %Make.call: init type = call %Make.ref(%t.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
