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
// CHECK:STDOUT:   %ptr.31e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.edf: type = maybe_unformed_type %ptr.31e [concrete]
// CHECK:STDOUT:   %pattern_type.b78: type = pattern_type %.edf [concrete]
// CHECK:STDOUT:   %MakeNullC.type: type = fn_type @MakeNullC [concrete]
// CHECK:STDOUT:   %MakeNullC: %MakeNullC.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.b42 = value_binding_pattern s [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc55.1: type = splice_block %.loc55.3 [concrete = constants.%.b2d] {
// CHECK:STDOUT:     %MakeUnformed.ref.loc50: %MakeUnformed.type = name_ref MakeUnformed, %MakeUnformed.decl [concrete = constants.%MakeUnformed]
// CHECK:STDOUT:     %.loc53: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc54: type = converted %.loc53, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %ptr.loc54: type = ptr_type %.loc54 [concrete = constants.%ptr.c28]
// CHECK:STDOUT:     %MakeUnformed.call.loc55: init type = call %MakeUnformed.ref.loc50(%ptr.loc54) [concrete = constants.%.b2d]
// CHECK:STDOUT:     %.loc55.2: type = value_of_initializer %MakeUnformed.call.loc55 [concrete = constants.%.b2d]
// CHECK:STDOUT:     %.loc55.3: type = converted %MakeUnformed.call.loc55, %.loc55.2 [concrete = constants.%.b2d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc59.1: %.b2d = value_of_initializer @__global_init.%MakeNullEmptyStruct.call
// CHECK:STDOUT:   %.loc59.2: %.b2d = converted @__global_init.%MakeNullEmptyStruct.call, %.loc59.1
// CHECK:STDOUT:   %s: %.b2d = value_binding s, %.loc59.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.b78 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc68.1: type = splice_block %.loc68.3 [concrete = constants.%.edf] {
// CHECK:STDOUT:     %MakeUnformed.ref.loc64: %MakeUnformed.type = name_ref MakeUnformed, %MakeUnformed.decl [concrete = constants.%MakeUnformed]
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc67: type = ptr_type %C.ref [concrete = constants.%ptr.31e]
// CHECK:STDOUT:     %MakeUnformed.call.loc68: init type = call %MakeUnformed.ref.loc64(%ptr.loc67) [concrete = constants.%.edf]
// CHECK:STDOUT:     %.loc68.2: type = value_of_initializer %MakeUnformed.call.loc68 [concrete = constants.%.edf]
// CHECK:STDOUT:     %.loc68.3: type = converted %MakeUnformed.call.loc68, %.loc68.2 [concrete = constants.%.edf]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc72.1: %.edf = value_of_initializer @__global_init.%MakeNullC.call
// CHECK:STDOUT:   %.loc72.2: %.edf = converted @__global_init.%MakeNullC.call, %.loc72.1
// CHECK:STDOUT:   %c: %.edf = value_binding c, %.loc72.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MakeNullEmptyStruct.ref: %MakeNullEmptyStruct.type = name_ref MakeNullEmptyStruct, file.%MakeNullEmptyStruct.decl [concrete = constants.%MakeNullEmptyStruct]
// CHECK:STDOUT:   %MakeNullEmptyStruct.call: init %.b2d = call %MakeNullEmptyStruct.ref()
// CHECK:STDOUT:   %MakeNullC.ref: %MakeNullC.type = name_ref MakeNullC, file.%MakeNullC.decl [concrete = constants.%MakeNullC]
// CHECK:STDOUT:   %MakeNullC.call: init %.edf = call %MakeNullC.ref()
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
// CHECK:STDOUT:   %MakeNull.specific_fn.4e3: <specific function> = specific_function %MakeNull, @MakeNull(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.31e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.edf: type = maybe_unformed_type %ptr.31e [concrete]
// CHECK:STDOUT:   %pattern_type.b78: type = pattern_type %.edf [concrete]
// CHECK:STDOUT:   %MakeNull.specific_fn.152: <specific function> = specific_function %MakeNull, @MakeNull(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.b42 = value_binding_pattern s [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc44.1: type = splice_block %.loc44.3 [concrete = constants.%.b2d] {
// CHECK:STDOUT:     %MakeUnformed.ref.loc39: %MakeUnformed.type = name_ref MakeUnformed, %MakeUnformed.decl [concrete = constants.%MakeUnformed]
// CHECK:STDOUT:     %.loc42: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc43: type = converted %.loc42, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %ptr.loc43: type = ptr_type %.loc43 [concrete = constants.%ptr.c28]
// CHECK:STDOUT:     %MakeUnformed.call.loc44: init type = call %MakeUnformed.ref.loc39(%ptr.loc43) [concrete = constants.%.b2d]
// CHECK:STDOUT:     %.loc44.2: type = value_of_initializer %MakeUnformed.call.loc44 [concrete = constants.%.b2d]
// CHECK:STDOUT:     %.loc44.3: type = converted %MakeUnformed.call.loc44, %.loc44.2 [concrete = constants.%.b2d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc50.1: %.b2d = value_of_initializer @__global_init.%MakeNull.call.loc50
// CHECK:STDOUT:   %.loc50.2: %.b2d = converted @__global_init.%MakeNull.call.loc50, %.loc50.1
// CHECK:STDOUT:   %s: %.b2d = value_binding s, %.loc50.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.b78 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc59.1: type = splice_block %.loc59.3 [concrete = constants.%.edf] {
// CHECK:STDOUT:     %MakeUnformed.ref.loc55: %MakeUnformed.type = name_ref MakeUnformed, %MakeUnformed.decl [concrete = constants.%MakeUnformed]
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc58: type = ptr_type %C.ref [concrete = constants.%ptr.31e]
// CHECK:STDOUT:     %MakeUnformed.call.loc59: init type = call %MakeUnformed.ref.loc55(%ptr.loc58) [concrete = constants.%.edf]
// CHECK:STDOUT:     %.loc59.2: type = value_of_initializer %MakeUnformed.call.loc59 [concrete = constants.%.edf]
// CHECK:STDOUT:     %.loc59.3: type = converted %MakeUnformed.call.loc59, %.loc59.2 [concrete = constants.%.edf]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc64.1: %.edf = value_of_initializer @__global_init.%MakeNull.call.loc64
// CHECK:STDOUT:   %.loc64.2: %.edf = converted @__global_init.%MakeNull.call.loc64, %.loc64.1
// CHECK:STDOUT:   %c: %.edf = value_binding c, %.loc64.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MakeNull.ref.loc46: %MakeNull.type = name_ref MakeNull, file.%MakeNull.decl [concrete = constants.%MakeNull]
// CHECK:STDOUT:   %.loc49: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc50: type = converted %.loc49, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %MakeNull.specific_fn.loc46: <specific function> = specific_function %MakeNull.ref.loc46, @MakeNull(constants.%empty_struct_type) [concrete = constants.%MakeNull.specific_fn.4e3]
// CHECK:STDOUT:   %MakeNull.call.loc50: init %.b2d = call %MakeNull.specific_fn.loc46()
// CHECK:STDOUT:   %MakeNull.ref.loc61: %MakeNull.type = name_ref MakeNull, file.%MakeNull.decl [concrete = constants.%MakeNull]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %MakeNull.specific_fn.loc61: <specific function> = specific_function %MakeNull.ref.loc61, @MakeNull(constants.%C) [concrete = constants.%MakeNull.specific_fn.152]
// CHECK:STDOUT:   %MakeNull.call.loc64: init %.edf = call %MakeNull.specific_fn.loc61()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
