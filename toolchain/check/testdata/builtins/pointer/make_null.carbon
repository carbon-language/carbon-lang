// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/pointer/make_null.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/pointer/make_null.carbon

// --- call_exact.carbon

library "[[@TEST_NAME]]";

class C {}

fn MakeUnformed(t: type) -> type = "maybe_unformed.make_type";
fn MakeNullEmptyStruct() -> MakeUnformed({}*) = "pointer.make_null";
fn MakeNullC() -> MakeUnformed(C*) = "pointer.make_null";

//@dump-sem-ir-begin
let s: MakeUnformed({}*) = MakeNullEmptyStruct();
let c: MakeUnformed(C*) = MakeNullC();
//@dump-sem-ir-end

// --- call_generic.carbon

library "[[@TEST_NAME]]";

fn MakeUnformed(t: type) -> type = "maybe_unformed.make_type";
fn MakeNull(T:! type) -> MakeUnformed(T*) = "pointer.make_null";

class C {}

//@dump-sem-ir-begin
let s: MakeUnformed({}*) = MakeNull({});
let c: MakeUnformed(C*) = MakeNull(C);
//@dump-sem-ir-end

// --- fail_bad_decl.carbon

library "[[@TEST_NAME]]";

fn MakeUnformed(t: type) -> type = "maybe_unformed.make_type";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "pointer.make_null" [InvalidBuiltinSignature]
// CHECK:STDERR: fn NoRetType() = "pointer.make_null";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NoRetType() = "pointer.make_null";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "pointer.make_null" [InvalidBuiltinSignature]
// CHECK:STDERR: fn NoUnformed() -> {}* = "pointer.make_null";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NoUnformed() -> {}* = "pointer.make_null";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "pointer.make_null" [InvalidBuiltinSignature]
// CHECK:STDERR: fn NotPointer() -> MakeUnformed({}) = "pointer.make_null";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NotPointer() -> MakeUnformed({}) = "pointer.make_null";

// CHECK:STDOUT: --- call_exact.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %MakeUnformed.type: type = fn_type @MakeUnformed [concrete]
// CHECK:STDOUT:   %MakeUnformed: %MakeUnformed.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %.b2d: type = maybe_unformed_type %ptr.c28 [concrete]
// CHECK:STDOUT:   %pattern_type.b42: type = pattern_type %.b2d [concrete]
// CHECK:STDOUT:   %MakeNullEmptyStruct.type: type = fn_type @MakeNullEmptyStruct [concrete]
// CHECK:STDOUT:   %MakeNullEmptyStruct: %MakeNullEmptyStruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.273: type = maybe_unformed_type %ptr.019 [concrete]
// CHECK:STDOUT:   %pattern_type.ad6: type = pattern_type %.273 [concrete]
// CHECK:STDOUT:   %MakeNullC.type: type = fn_type @MakeNullC [concrete]
// CHECK:STDOUT:   %MakeNullC: %MakeNullC.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.b42 = value_binding_pattern s [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_24.1: type = splice_block %.loc11_24.3 [concrete = constants.%.b2d] {
// CHECK:STDOUT:     %MakeUnformed.ref.loc11: %MakeUnformed.type = name_ref MakeUnformed, %MakeUnformed.decl [concrete = constants.%MakeUnformed]
// CHECK:STDOUT:     %.loc11_22: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc11_23: type = converted %.loc11_22, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %ptr.loc11: type = ptr_type %.loc11_23 [concrete = constants.%ptr.c28]
// CHECK:STDOUT:     %MakeUnformed.call.loc11: init type = call %MakeUnformed.ref.loc11(%ptr.loc11) [concrete = constants.%.b2d]
// CHECK:STDOUT:     %.loc11_24.2: type = value_of_initializer %MakeUnformed.call.loc11 [concrete = constants.%.b2d]
// CHECK:STDOUT:     %.loc11_24.3: type = converted %MakeUnformed.call.loc11, %.loc11_24.2 [concrete = constants.%.b2d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_48.1: %.b2d = value_of_initializer @__global_init.%MakeNullEmptyStruct.call
// CHECK:STDOUT:   %.loc11_48.2: %.b2d = converted @__global_init.%MakeNullEmptyStruct.call, %.loc11_48.1
// CHECK:STDOUT:   %s: %.b2d = value_binding s, %.loc11_48.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.ad6 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_23.1: type = splice_block %.loc12_23.3 [concrete = constants.%.273] {
// CHECK:STDOUT:     %MakeUnformed.ref.loc12: %MakeUnformed.type = name_ref MakeUnformed, %MakeUnformed.decl [concrete = constants.%MakeUnformed]
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc12: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     %MakeUnformed.call.loc12: init type = call %MakeUnformed.ref.loc12(%ptr.loc12) [concrete = constants.%.273]
// CHECK:STDOUT:     %.loc12_23.2: type = value_of_initializer %MakeUnformed.call.loc12 [concrete = constants.%.273]
// CHECK:STDOUT:     %.loc12_23.3: type = converted %MakeUnformed.call.loc12, %.loc12_23.2 [concrete = constants.%.273]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_37.1: %.273 = value_of_initializer @__global_init.%MakeNullC.call
// CHECK:STDOUT:   %.loc12_37.2: %.273 = converted @__global_init.%MakeNullC.call, %.loc12_37.1
// CHECK:STDOUT:   %c: %.273 = value_binding c, %.loc12_37.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MakeNullEmptyStruct.ref: %MakeNullEmptyStruct.type = name_ref MakeNullEmptyStruct, file.%MakeNullEmptyStruct.decl [concrete = constants.%MakeNullEmptyStruct]
// CHECK:STDOUT:   %MakeNullEmptyStruct.call: init %.b2d = call %MakeNullEmptyStruct.ref()
// CHECK:STDOUT:   %MakeNullC.ref: %MakeNullC.type = name_ref MakeNullC, file.%MakeNullC.decl [concrete = constants.%MakeNullC]
// CHECK:STDOUT:   %MakeNullC.call: init %.273 = call %MakeNullC.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MakeUnformed.type: type = fn_type @MakeUnformed [concrete]
// CHECK:STDOUT:   %MakeUnformed: %MakeUnformed.type = struct_value () [concrete]
// CHECK:STDOUT:   %MakeNull.type: type = fn_type @MakeNull [concrete]
// CHECK:STDOUT:   %MakeNull: %MakeNull.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %.b2d: type = maybe_unformed_type %ptr.c28 [concrete]
// CHECK:STDOUT:   %pattern_type.b42: type = pattern_type %.b2d [concrete]
// CHECK:STDOUT:   %MakeNull.specific_fn.7cb: <specific function> = specific_function %MakeNull, @MakeNull(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.273: type = maybe_unformed_type %ptr.019 [concrete]
// CHECK:STDOUT:   %pattern_type.ad6: type = pattern_type %.273 [concrete]
// CHECK:STDOUT:   %MakeNull.specific_fn.22e: <specific function> = specific_function %MakeNull, @MakeNull(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.b42 = value_binding_pattern s [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_24.1: type = splice_block %.loc10_24.3 [concrete = constants.%.b2d] {
// CHECK:STDOUT:     %MakeUnformed.ref.loc10: %MakeUnformed.type = name_ref MakeUnformed, %MakeUnformed.decl [concrete = constants.%MakeUnformed]
// CHECK:STDOUT:     %.loc10_22: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc10_23: type = converted %.loc10_22, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %.loc10_23 [concrete = constants.%ptr.c28]
// CHECK:STDOUT:     %MakeUnformed.call.loc10: init type = call %MakeUnformed.ref.loc10(%ptr.loc10) [concrete = constants.%.b2d]
// CHECK:STDOUT:     %.loc10_24.2: type = value_of_initializer %MakeUnformed.call.loc10 [concrete = constants.%.b2d]
// CHECK:STDOUT:     %.loc10_24.3: type = converted %MakeUnformed.call.loc10, %.loc10_24.2 [concrete = constants.%.b2d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_39.1: %.b2d = value_of_initializer @__global_init.%MakeNull.call.loc10
// CHECK:STDOUT:   %.loc10_39.2: %.b2d = converted @__global_init.%MakeNull.call.loc10, %.loc10_39.1
// CHECK:STDOUT:   %s: %.b2d = value_binding s, %.loc10_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.ad6 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_23.1: type = splice_block %.loc11_23.3 [concrete = constants.%.273] {
// CHECK:STDOUT:     %MakeUnformed.ref.loc11: %MakeUnformed.type = name_ref MakeUnformed, %MakeUnformed.decl [concrete = constants.%MakeUnformed]
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc11: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     %MakeUnformed.call.loc11: init type = call %MakeUnformed.ref.loc11(%ptr.loc11) [concrete = constants.%.273]
// CHECK:STDOUT:     %.loc11_23.2: type = value_of_initializer %MakeUnformed.call.loc11 [concrete = constants.%.273]
// CHECK:STDOUT:     %.loc11_23.3: type = converted %MakeUnformed.call.loc11, %.loc11_23.2 [concrete = constants.%.273]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_37.1: %.273 = value_of_initializer @__global_init.%MakeNull.call.loc11
// CHECK:STDOUT:   %.loc11_37.2: %.273 = converted @__global_init.%MakeNull.call.loc11, %.loc11_37.1
// CHECK:STDOUT:   %c: %.273 = value_binding c, %.loc11_37.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MakeNull.ref.loc10: %MakeNull.type = name_ref MakeNull, file.%MakeNull.decl [concrete = constants.%MakeNull]
// CHECK:STDOUT:   %.loc10_38: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_39: type = converted %.loc10_38, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %MakeNull.specific_fn.loc10: <specific function> = specific_function %MakeNull.ref.loc10, @MakeNull(constants.%empty_struct_type) [concrete = constants.%MakeNull.specific_fn.7cb]
// CHECK:STDOUT:   %MakeNull.call.loc10: init %.b2d = call %MakeNull.specific_fn.loc10()
// CHECK:STDOUT:   %MakeNull.ref.loc11: %MakeNull.type = name_ref MakeNull, file.%MakeNull.decl [concrete = constants.%MakeNull]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %MakeNull.specific_fn.loc11: <specific function> = specific_function %MakeNull.ref.loc11, @MakeNull(constants.%C) [concrete = constants.%MakeNull.specific_fn.22e]
// CHECK:STDOUT:   %MakeNull.call.loc11: init %.273 = call %MakeNull.specific_fn.loc11()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
