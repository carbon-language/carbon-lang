// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/pointer/is_null.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/pointer/is_null.carbon

// --- call_exact.carbon

library "[[@TEST_NAME]]";

class C {}

fn MakeUnformed(t: type) -> type = "maybe_unformed.make_type";
fn IsNullEmptyStruct(p: MakeUnformed({}*)) -> bool = "pointer.is_null";
fn IsNullC(p: MakeUnformed(C*)) -> bool = "pointer.is_null";

//@dump-sem-ir-begin
fn TestEmptyStruct(s: MakeUnformed({}*)) -> bool {
  return IsNullEmptyStruct(s);
}

fn TestC(c: MakeUnformed(C*)) -> bool {
  return IsNullC(c);
}
//@dump-sem-ir-end

// --- call_generic.carbon

library "[[@TEST_NAME]]";

fn MakeUnformed(t: type) -> type = "maybe_unformed.make_type";
fn IsNull[T:! type](p: MakeUnformed(T*)) -> bool = "pointer.is_null";

class C {}

//@dump-sem-ir-begin
fn TestEmptyStruct(s: MakeUnformed({}*)) -> bool {
  return IsNull(s);
}

fn TestC(c: MakeUnformed(C*)) -> bool {
  return IsNull(c);
}
//@dump-sem-ir-end

// --- fail_bad_decl.carbon

library "[[@TEST_NAME]]";

fn MakeUnformed(t: type) -> type = "maybe_unformed.make_type";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "pointer.is_null" [InvalidBuiltinSignature]
// CHECK:STDERR: fn NoParam() -> bool = "pointer.is_null";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NoParam() -> bool = "pointer.is_null";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "pointer.is_null" [InvalidBuiltinSignature]
// CHECK:STDERR: fn NoRetType(p: MakeUnformed({}*)) = "pointer.is_null";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NoRetType(p: MakeUnformed({}*)) = "pointer.is_null";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "pointer.is_null" [InvalidBuiltinSignature]
// CHECK:STDERR: fn WrongRetType(p: MakeUnformed({}*)) -> {} = "pointer.is_null";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn WrongRetType(p: MakeUnformed({}*)) -> {} = "pointer.is_null";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "pointer.is_null" [InvalidBuiltinSignature]
// CHECK:STDERR: fn NoUnformed(p: {}*) -> bool = "pointer.is_null";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NoUnformed(p: {}*) -> bool = "pointer.is_null";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "pointer.is_null" [InvalidBuiltinSignature]
// CHECK:STDERR: fn NotPointer(p: MakeUnformed({})) -> bool = "pointer.is_null";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NotPointer(p: MakeUnformed({})) -> bool = "pointer.is_null";

// CHECK:STDOUT: --- call_exact.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %MakeUnformed.type: type = fn_type @MakeUnformed [concrete]
// CHECK:STDOUT:   %MakeUnformed: %MakeUnformed.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %.b2d: type = maybe_unformed_type %ptr.c28 [concrete]
// CHECK:STDOUT:   %pattern_type.b42: type = pattern_type %.b2d [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %IsNullEmptyStruct.type: type = fn_type @IsNullEmptyStruct [concrete]
// CHECK:STDOUT:   %IsNullEmptyStruct: %IsNullEmptyStruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.273: type = maybe_unformed_type %ptr.019 [concrete]
// CHECK:STDOUT:   %pattern_type.ad6: type = pattern_type %.273 [concrete]
// CHECK:STDOUT:   %IsNullC.type: type = fn_type @IsNullC [concrete]
// CHECK:STDOUT:   %IsNullC: %IsNullC.type = struct_value () [concrete]
// CHECK:STDOUT:   %TestEmptyStruct.type: type = fn_type @TestEmptyStruct [concrete]
// CHECK:STDOUT:   %TestEmptyStruct: %TestEmptyStruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %TestC.type: type = fn_type @TestC [concrete]
// CHECK:STDOUT:   %TestC: %TestC.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %TestEmptyStruct.decl: %TestEmptyStruct.type = fn_decl @TestEmptyStruct [concrete = constants.%TestEmptyStruct] {
// CHECK:STDOUT:     %s.patt: %pattern_type.b42 = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.param_patt: %pattern_type.b42 = value_param_pattern %s.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc11_45.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc11_45.2: type = converted %Bool.call, %.loc11_45.1 [concrete = bool]
// CHECK:STDOUT:     %s.param: %.b2d = value_param call_param0
// CHECK:STDOUT:     %.loc11_39.1: type = splice_block %.loc11_39.3 [concrete = constants.%.b2d] {
// CHECK:STDOUT:       %MakeUnformed.ref: %MakeUnformed.type = name_ref MakeUnformed, file.%MakeUnformed.decl [concrete = constants.%MakeUnformed]
// CHECK:STDOUT:       %.loc11_37: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc11_38: type = converted %.loc11_37, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %ptr: type = ptr_type %.loc11_38 [concrete = constants.%ptr.c28]
// CHECK:STDOUT:       %MakeUnformed.call: init type = call %MakeUnformed.ref(%ptr) [concrete = constants.%.b2d]
// CHECK:STDOUT:       %.loc11_39.2: type = value_of_initializer %MakeUnformed.call [concrete = constants.%.b2d]
// CHECK:STDOUT:       %.loc11_39.3: type = converted %MakeUnformed.call, %.loc11_39.2 [concrete = constants.%.b2d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s: %.b2d = bind_name s, %s.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestC.decl: %TestC.type = fn_decl @TestC [concrete = constants.%TestC] {
// CHECK:STDOUT:     %c.patt: %pattern_type.ad6 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.ad6 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc15_34.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc15_34.2: type = converted %Bool.call, %.loc15_34.1 [concrete = bool]
// CHECK:STDOUT:     %c.param: %.273 = value_param call_param0
// CHECK:STDOUT:     %.loc15_28.1: type = splice_block %.loc15_28.3 [concrete = constants.%.273] {
// CHECK:STDOUT:       %MakeUnformed.ref: %MakeUnformed.type = name_ref MakeUnformed, file.%MakeUnformed.decl [concrete = constants.%MakeUnformed]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:       %MakeUnformed.call: init type = call %MakeUnformed.ref(%ptr) [concrete = constants.%.273]
// CHECK:STDOUT:       %.loc15_28.2: type = value_of_initializer %MakeUnformed.call [concrete = constants.%.273]
// CHECK:STDOUT:       %.loc15_28.3: type = converted %MakeUnformed.call, %.loc15_28.2 [concrete = constants.%.273]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %.273 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEmptyStruct(%s.param: %.b2d) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IsNullEmptyStruct.ref: %IsNullEmptyStruct.type = name_ref IsNullEmptyStruct, file.%IsNullEmptyStruct.decl [concrete = constants.%IsNullEmptyStruct]
// CHECK:STDOUT:   %s.ref: %.b2d = name_ref s, %s
// CHECK:STDOUT:   %IsNullEmptyStruct.call: init bool = call %IsNullEmptyStruct.ref(%s.ref)
// CHECK:STDOUT:   return %IsNullEmptyStruct.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestC(%c.param: %.273) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IsNullC.ref: %IsNullC.type = name_ref IsNullC, file.%IsNullC.decl [concrete = constants.%IsNullC]
// CHECK:STDOUT:   %c.ref: %.273 = name_ref c, %c
// CHECK:STDOUT:   %IsNullC.call: init bool = call %IsNullC.ref(%c.ref)
// CHECK:STDOUT:   return %IsNullC.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %MakeUnformed.type: type = fn_type @MakeUnformed [concrete]
// CHECK:STDOUT:   %MakeUnformed: %MakeUnformed.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %IsNull.type: type = fn_type @IsNull [concrete]
// CHECK:STDOUT:   %IsNull: %IsNull.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %.b2d: type = maybe_unformed_type %ptr.c28 [concrete]
// CHECK:STDOUT:   %pattern_type.b42: type = pattern_type %.b2d [concrete]
// CHECK:STDOUT:   %TestEmptyStruct.type: type = fn_type @TestEmptyStruct [concrete]
// CHECK:STDOUT:   %TestEmptyStruct: %TestEmptyStruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %IsNull.specific_fn.c98: <specific function> = specific_function %IsNull, @IsNull(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.273: type = maybe_unformed_type %ptr.019 [concrete]
// CHECK:STDOUT:   %pattern_type.ad6: type = pattern_type %.273 [concrete]
// CHECK:STDOUT:   %TestC.type: type = fn_type @TestC [concrete]
// CHECK:STDOUT:   %TestC: %TestC.type = struct_value () [concrete]
// CHECK:STDOUT:   %IsNull.specific_fn.87c: <specific function> = specific_function %IsNull, @IsNull(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %TestEmptyStruct.decl: %TestEmptyStruct.type = fn_decl @TestEmptyStruct [concrete = constants.%TestEmptyStruct] {
// CHECK:STDOUT:     %s.patt: %pattern_type.b42 = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.param_patt: %pattern_type.b42 = value_param_pattern %s.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc10_45.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc10_45.2: type = converted %Bool.call, %.loc10_45.1 [concrete = bool]
// CHECK:STDOUT:     %s.param: %.b2d = value_param call_param0
// CHECK:STDOUT:     %.loc10_39.1: type = splice_block %.loc10_39.3 [concrete = constants.%.b2d] {
// CHECK:STDOUT:       %MakeUnformed.ref: %MakeUnformed.type = name_ref MakeUnformed, file.%MakeUnformed.decl [concrete = constants.%MakeUnformed]
// CHECK:STDOUT:       %.loc10_37: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc10_38: type = converted %.loc10_37, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %ptr: type = ptr_type %.loc10_38 [concrete = constants.%ptr.c28]
// CHECK:STDOUT:       %MakeUnformed.call: init type = call %MakeUnformed.ref(%ptr) [concrete = constants.%.b2d]
// CHECK:STDOUT:       %.loc10_39.2: type = value_of_initializer %MakeUnformed.call [concrete = constants.%.b2d]
// CHECK:STDOUT:       %.loc10_39.3: type = converted %MakeUnformed.call, %.loc10_39.2 [concrete = constants.%.b2d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s: %.b2d = bind_name s, %s.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestC.decl: %TestC.type = fn_decl @TestC [concrete = constants.%TestC] {
// CHECK:STDOUT:     %c.patt: %pattern_type.ad6 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.ad6 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc14_34.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc14_34.2: type = converted %Bool.call, %.loc14_34.1 [concrete = bool]
// CHECK:STDOUT:     %c.param: %.273 = value_param call_param0
// CHECK:STDOUT:     %.loc14_28.1: type = splice_block %.loc14_28.3 [concrete = constants.%.273] {
// CHECK:STDOUT:       %MakeUnformed.ref: %MakeUnformed.type = name_ref MakeUnformed, file.%MakeUnformed.decl [concrete = constants.%MakeUnformed]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:       %MakeUnformed.call: init type = call %MakeUnformed.ref(%ptr) [concrete = constants.%.273]
// CHECK:STDOUT:       %.loc14_28.2: type = value_of_initializer %MakeUnformed.call [concrete = constants.%.273]
// CHECK:STDOUT:       %.loc14_28.3: type = converted %MakeUnformed.call, %.loc14_28.2 [concrete = constants.%.273]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %.273 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEmptyStruct(%s.param: %.b2d) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IsNull.ref: %IsNull.type = name_ref IsNull, file.%IsNull.decl [concrete = constants.%IsNull]
// CHECK:STDOUT:   %s.ref: %.b2d = name_ref s, %s
// CHECK:STDOUT:   %IsNull.specific_fn: <specific function> = specific_function %IsNull.ref, @IsNull(constants.%empty_struct_type) [concrete = constants.%IsNull.specific_fn.c98]
// CHECK:STDOUT:   %IsNull.call: init bool = call %IsNull.specific_fn(%s.ref)
// CHECK:STDOUT:   return %IsNull.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestC(%c.param: %.273) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IsNull.ref: %IsNull.type = name_ref IsNull, file.%IsNull.decl [concrete = constants.%IsNull]
// CHECK:STDOUT:   %c.ref: %.273 = name_ref c, %c
// CHECK:STDOUT:   %IsNull.specific_fn: <specific function> = specific_function %IsNull.ref, @IsNull(constants.%C) [concrete = constants.%IsNull.specific_fn.87c]
// CHECK:STDOUT:   %IsNull.call: init bool = call %IsNull.specific_fn(%c.ref)
// CHECK:STDOUT:   return %IsNull.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
