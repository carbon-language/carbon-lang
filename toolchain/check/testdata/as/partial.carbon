// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/partial.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/partial.carbon

// --- add_partial.carbon

library "[[@TEST_NAME]]";

base class X {}

fn Init() -> X;
let value: X = Init();
var reference: X;
let ptr: X* = &reference;

fn Use() {
  // TODO: Should some of these be valid without the `as`?
  //@dump-sem-ir-begin
  var i: partial X = Init() as partial X;
  let v: partial X = value as partial X;
  let a: partial X* = &(reference as partial X);
  let b: partial X* = ptr as partial X*;
  //@dump-sem-ir-end
}

// --- fail_todo_remove_partial_in_init.carbon

library "[[@TEST_NAME]]";

base class X {}

fn Init() -> partial X;

fn Use() {
  //@dump-sem-ir-begin
  // TODO: This should be valid, and should initialize the vptr. The explicit `as` should probably not be necessary.
  // CHECK:STDERR: fail_todo_remove_partial_in_init.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `partial X` to `X` [ConversionFailure]
  // CHECK:STDERR:   var i: X = Init();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_todo_remove_partial_in_init.carbon:[[@LINE+4]]:3: note: type `partial X` does not implement interface `Core.ImplicitAs(X)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var i: X = Init();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  var i: X = Init();
  // CHECK:STDERR: fail_todo_remove_partial_in_init.carbon:[[@LINE+7]]:14: error: cannot convert expression of type `partial X` to `X` with `as` [ConversionFailure]
  // CHECK:STDERR:   var j: X = Init() as X;
  // CHECK:STDERR:              ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_remove_partial_in_init.carbon:[[@LINE+4]]:14: note: type `partial X` does not implement interface `Core.As(X)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var j: X = Init() as X;
  // CHECK:STDERR:              ^~~~~~~~~~~
  // CHECK:STDERR:
  var j: X = Init() as X;
  // CHECK:STDERR: fail_todo_remove_partial_in_init.carbon:[[@LINE+7]]:14: error: cannot convert expression of type `partial X` to `X` with `unsafe as` [ConversionFailure]
  // CHECK:STDERR:   var k: X = Init() unsafe as X;
  // CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_remove_partial_in_init.carbon:[[@LINE+4]]:14: note: type `partial X` does not implement interface `Core.UnsafeAs(X)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var k: X = Init() unsafe as X;
  // CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var k: X = Init() unsafe as X;
  //@dump-sem-ir-end
}

// --- fail_cannot_remove_partial.carbon

library "[[@TEST_NAME]]";

base class X {}

fn Init() -> partial X;

let value: partial X = Init();
var reference: partial X;
let ptr: partial X* = &reference;

fn Use() {
  // CHECK:STDERR: fail_cannot_remove_partial.carbon:[[@LINE+7]]:14: error: cannot convert expression of type `partial X` to `X` with `as` [ConversionFailure]
  // CHECK:STDERR:   let v: X = value as X;
  // CHECK:STDERR:              ^~~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_partial.carbon:[[@LINE+4]]:14: note: type `partial X` does not implement interface `Core.As(X)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let v: X = value as X;
  // CHECK:STDERR:              ^~~~~~~~~~
  // CHECK:STDERR:
  let v: X = value as X;
  // CHECK:STDERR: fail_cannot_remove_partial.carbon:[[@LINE+7]]:17: error: cannot convert expression of type `partial X` to `X` with `as` [ConversionFailure]
  // CHECK:STDERR:   let a: X* = &(reference as X);
  // CHECK:STDERR:                 ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_partial.carbon:[[@LINE+4]]:17: note: type `partial X` does not implement interface `Core.As(X)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let a: X* = &(reference as X);
  // CHECK:STDERR:                 ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: X* = &(reference as X);
  // CHECK:STDERR: fail_cannot_remove_partial.carbon:[[@LINE+7]]:15: error: cannot convert expression of type `partial X*` to `X*` with `as` [ConversionFailure]
  // CHECK:STDERR:   let b: X* = ptr as X*;
  // CHECK:STDERR:               ^~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_partial.carbon:[[@LINE+4]]:15: note: type `partial X*` does not implement interface `Core.As(X*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let b: X* = ptr as X*;
  // CHECK:STDERR:               ^~~~~~~~~
  // CHECK:STDERR:
  let b: X* = ptr as X*;
}

// --- unsafe_remove_partial.carbon

library "[[@TEST_NAME]]";

base class X {}

fn Init() -> partial X;

let value: partial X = Init();
var reference: partial X;
let ptr: partial X* = &reference;

fn Use() {
  //@dump-sem-ir-begin
  let v: X = value unsafe as X;
  let a: X* = &(reference unsafe as X);
  let b: X* = ptr unsafe as X*;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- add_partial.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %Init.type: type = fn_type @Init [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Init: %Init.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.2a9: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %.4b5: type = partial_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.bc5: type = pattern_type %.4b5 [concrete]
// CHECK:STDOUT:   %ptr.314: type = ptr_type %.4b5 [concrete]
// CHECK:STDOUT:   %pattern_type.408: type = pattern_type %ptr.314 [concrete]
// CHECK:STDOUT:   %reference.var: ref %.4b5 = var file.%reference.var_patt [concrete]
// CHECK:STDOUT:   %addr.315: %ptr.314 = addr_of %reference.var [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.bc5 = ref_binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.bc5 = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %.4b5 = var %i.var_patt
// CHECK:STDOUT:   %Init.ref: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc14_3: ref %.4b5 = splice_block %i.var {}
// CHECK:STDOUT:   %Init.call: init %X to %.loc14_3 = call %Init.ref()
// CHECK:STDOUT:   %X.ref.loc14_40: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc14_32: type = partial_type %X.ref.loc14_40 [concrete = constants.%.4b5]
// CHECK:STDOUT:   %.loc14_29.1: init %.4b5 = as_compatible %Init.call
// CHECK:STDOUT:   %.loc14_29.2: init %.4b5 = converted %Init.call, %.loc14_29.1
// CHECK:STDOUT:   assign %i.var, %.loc14_29.2
// CHECK:STDOUT:   %.loc14_10.1: type = splice_block %.loc14_10.2 [concrete = constants.%.4b5] {
// CHECK:STDOUT:     %X.ref.loc14_18: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc14_10.2: type = partial_type %X.ref.loc14_18 [concrete = constants.%.4b5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: ref %.4b5 = ref_binding i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.bc5 = value_binding_pattern v [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %value.ref: %X = name_ref value, file.%value
// CHECK:STDOUT:   %X.ref.loc15_39: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc15_31: type = partial_type %X.ref.loc15_39 [concrete = constants.%.4b5]
// CHECK:STDOUT:   %.loc15_28.1: %.4b5 = as_compatible %value.ref
// CHECK:STDOUT:   %.loc15_28.2: %.4b5 = converted %value.ref, %.loc15_28.1
// CHECK:STDOUT:   %.loc15_10.1: type = splice_block %.loc15_10.2 [concrete = constants.%.4b5] {
// CHECK:STDOUT:     %X.ref.loc15_18: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc15_10.2: type = partial_type %X.ref.loc15_18 [concrete = constants.%.4b5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: %.4b5 = value_binding v, %.loc15_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.408 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %reference.ref: ref %X = name_ref reference, file.%reference [concrete = file.%reference.var]
// CHECK:STDOUT:   %X.ref.loc16_46: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc16_38: type = partial_type %X.ref.loc16_46 [concrete = constants.%.4b5]
// CHECK:STDOUT:   %.loc16_35.1: ref %.4b5 = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc16_35.2: ref %.4b5 = converted %reference.ref, %.loc16_35.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr: %ptr.314 = addr_of %.loc16_35.2 [concrete = constants.%addr.315]
// CHECK:STDOUT:   %.loc16_19: type = splice_block %ptr.loc16 [concrete = constants.%ptr.314] {
// CHECK:STDOUT:     %X.ref.loc16_18: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc16_10: type = partial_type %X.ref.loc16_18 [concrete = constants.%.4b5]
// CHECK:STDOUT:     %ptr.loc16: type = ptr_type %.loc16_10 [concrete = constants.%ptr.314]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.314 = value_binding a, %addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.408 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.2a9 = name_ref ptr, file.%ptr.loc9_5
// CHECK:STDOUT:   %X.ref.loc17_38: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc17_30: type = partial_type %X.ref.loc17_38 [concrete = constants.%.4b5]
// CHECK:STDOUT:   %ptr.loc17_39: type = ptr_type %.loc17_30 [concrete = constants.%ptr.314]
// CHECK:STDOUT:   %.loc17_27.1: %ptr.314 = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc17_27.2: %ptr.314 = converted %ptr.ref, %.loc17_27.1
// CHECK:STDOUT:   %.loc17_19: type = splice_block %ptr.loc17_19 [concrete = constants.%ptr.314] {
// CHECK:STDOUT:     %X.ref.loc17_18: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc17_10: type = partial_type %X.ref.loc17_18 [concrete = constants.%.4b5]
// CHECK:STDOUT:     %ptr.loc17_19: type = ptr_type %.loc17_10 [concrete = constants.%ptr.314]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.314 = value_binding b, %.loc17_27.2
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %i.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%i.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %.4b5) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_remove_partial_in_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %.4b5: type = partial_type %X [concrete]
// CHECK:STDOUT:   %Init.type: type = fn_type @Init [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Init: %Init.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.05f: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.05f = ref_binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.05f = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %X = var %i.var_patt
// CHECK:STDOUT:   %Init.ref.loc18: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc18_19: ref %.4b5 = temporary_storage
// CHECK:STDOUT:   %Init.call.loc18: init %.4b5 to %.loc18_19 = call %Init.ref.loc18()
// CHECK:STDOUT:   %.loc18_3: %X = converted %Init.call.loc18, <error> [concrete = <error>]
// CHECK:STDOUT:   assign %i.var, <error>
// CHECK:STDOUT:   %X.ref.loc18: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %i: ref %X = ref_binding i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %j.patt: %pattern_type.05f = ref_binding_pattern j [concrete]
// CHECK:STDOUT:     %j.var_patt: %pattern_type.05f = var_pattern %j.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %j.var: ref %X = var %j.var_patt
// CHECK:STDOUT:   %Init.ref.loc26: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc26_19: ref %.4b5 = temporary_storage
// CHECK:STDOUT:   %Init.call.loc26: init %.4b5 to %.loc26_19 = call %Init.ref.loc26()
// CHECK:STDOUT:   %X.ref.loc26_24: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc26_21: %X = converted %Init.call.loc26, <error> [concrete = <error>]
// CHECK:STDOUT:   assign %j.var, <error>
// CHECK:STDOUT:   %X.ref.loc26_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %j: ref %X = ref_binding j, %j.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %k.patt: %pattern_type.05f = ref_binding_pattern k [concrete]
// CHECK:STDOUT:     %k.var_patt: %pattern_type.05f = var_pattern %k.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %k.var: ref %X = var %k.var_patt
// CHECK:STDOUT:   %Init.ref.loc34: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc34_19: ref %.4b5 = temporary_storage
// CHECK:STDOUT:   %Init.call.loc34: init %.4b5 to %.loc34_19 = call %Init.ref.loc34()
// CHECK:STDOUT:   %X.ref.loc34_31: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc34_28: %X = converted %Init.call.loc34, <error> [concrete = <error>]
// CHECK:STDOUT:   assign %k.var, <error>
// CHECK:STDOUT:   %X.ref.loc34_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %k: ref %X = ref_binding k, %k.var
// CHECK:STDOUT:   %DestroyOp.bound.loc34: <bound method> = bound_method %k.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc34: init %empty_tuple.type = call %DestroyOp.bound.loc34(%k.var)
// CHECK:STDOUT:   %DestroyOp.bound.loc26: <bound method> = bound_method %j.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc26: init %empty_tuple.type = call %DestroyOp.bound.loc26(%j.var)
// CHECK:STDOUT:   %DestroyOp.bound.loc18: <bound method> = bound_method %i.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call.loc18: init %empty_tuple.type = call %DestroyOp.bound.loc18(%i.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %X) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsafe_remove_partial.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %.4b5: type = partial_type %X [concrete]
// CHECK:STDOUT:   %ptr.314: type = ptr_type %.4b5 [concrete]
// CHECK:STDOUT:   %pattern_type.05f: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %ptr.2a9: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.37f: type = pattern_type %ptr.2a9 [concrete]
// CHECK:STDOUT:   %reference.var: ref %X = var file.%reference.var_patt [concrete]
// CHECK:STDOUT:   %addr.9cd: %ptr.2a9 = addr_of %reference.var [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.05f = value_binding_pattern v [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %value.ref: %.4b5 = name_ref value, file.%value
// CHECK:STDOUT:   %X.ref.loc14_30: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc14_27.1: %X = as_compatible %value.ref
// CHECK:STDOUT:   %.loc14_27.2: %X = converted %value.ref, %.loc14_27.1
// CHECK:STDOUT:   %X.ref.loc14_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %v: %X = value_binding v, %.loc14_27.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.37f = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %reference.ref: ref %.4b5 = name_ref reference, file.%reference [concrete = file.%reference.var]
// CHECK:STDOUT:   %X.ref.loc15_37: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc15_34.1: ref %X = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc15_34.2: ref %X = converted %reference.ref, %.loc15_34.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr: %ptr.2a9 = addr_of %.loc15_34.2 [concrete = constants.%addr.9cd]
// CHECK:STDOUT:   %.loc15_11: type = splice_block %ptr.loc15 [concrete = constants.%ptr.2a9] {
// CHECK:STDOUT:     %X.ref.loc15_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc15: type = ptr_type %X.ref.loc15_10 [concrete = constants.%ptr.2a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.2a9 = value_binding a, %addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.37f = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.314 = name_ref ptr, file.%ptr.loc10_5
// CHECK:STDOUT:   %X.ref.loc16_29: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %ptr.loc16_30: type = ptr_type %X.ref.loc16_29 [concrete = constants.%ptr.2a9]
// CHECK:STDOUT:   %.loc16_26.1: %ptr.2a9 = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc16_26.2: %ptr.2a9 = converted %ptr.ref, %.loc16_26.1
// CHECK:STDOUT:   %.loc16_11: type = splice_block %ptr.loc16_11 [concrete = constants.%ptr.2a9] {
// CHECK:STDOUT:     %X.ref.loc16_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc16_11: type = ptr_type %X.ref.loc16_10 [concrete = constants.%ptr.2a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.2a9 = value_binding b, %.loc16_26.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
