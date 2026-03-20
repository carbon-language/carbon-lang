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
  var unused i: partial X = Init() as partial X;
  let unused v: partial X = value as partial X;
  let unused a: partial X* = &(reference as partial X);
  let unused b: partial X* = ptr as partial X*;
  //@dump-sem-ir-end
}

// --- remove_partial_in_init.carbon

library "[[@TEST_NAME]]";

base class X {}

fn Init() -> partial X;

fn Use() {
  //@dump-sem-ir-begin
  var unused i: X = Init();
  var unused j: X = Init() as X;
  var unused k: X = Init() unsafe as X;
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
  // CHECK:STDERR: fail_cannot_remove_partial.carbon:[[@LINE+7]]:21: error: cannot convert expression of type `partial X` to `X` with `as` [ConversionFailure]
  // CHECK:STDERR:   let unused v: X = value as X;
  // CHECK:STDERR:                     ^~~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_partial.carbon:[[@LINE+4]]:21: note: type `partial X` does not implement interface `Core.As(X)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused v: X = value as X;
  // CHECK:STDERR:                     ^~~~~~~~~~
  // CHECK:STDERR:
  let unused v: X = value as X;
  // CHECK:STDERR: fail_cannot_remove_partial.carbon:[[@LINE+7]]:24: error: cannot convert expression of type `partial X` to `X` with `as` [ConversionFailure]
  // CHECK:STDERR:   let unused a: X* = &(reference as X);
  // CHECK:STDERR:                        ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_partial.carbon:[[@LINE+4]]:24: note: type `partial X` does not implement interface `Core.As(X)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused a: X* = &(reference as X);
  // CHECK:STDERR:                        ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused a: X* = &(reference as X);
  // CHECK:STDERR: fail_cannot_remove_partial.carbon:[[@LINE+7]]:22: error: cannot convert expression of type `partial X*` to `X*` with `as` [ConversionFailure]
  // CHECK:STDERR:   let unused b: X* = ptr as X*;
  // CHECK:STDERR:                      ^~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_partial.carbon:[[@LINE+4]]:22: note: type `partial X*` does not implement interface `Core.As(X*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused b: X* = ptr as X*;
  // CHECK:STDERR:                      ^~~~~~~~~
  // CHECK:STDERR:
  let unused b: X* = ptr as X*;
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
  let unused v: X = value unsafe as X;
  let unused a: X* = &(reference unsafe as X);
  let unused b: X* = ptr unsafe as X*;
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
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %.loc44: ref %.4b5 = splice_block %i.var {}
// CHECK:STDOUT:   %Init.call: init %X to %.loc44 = call %Init.ref()
// CHECK:STDOUT:   %X.ref.loc56: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc55: type = partial_type %X.ref.loc56 [concrete = constants.%.4b5]
// CHECK:STDOUT:   %.loc54.1: init %.4b5 = as_compatible %Init.call
// CHECK:STDOUT:   %.loc54.2: init %.4b5 = converted %Init.call, %.loc54.1
// CHECK:STDOUT:   assign %i.var, %.loc54.2
// CHECK:STDOUT:   %.loc48.1: type = splice_block %.loc48.2 [concrete = constants.%.4b5] {
// CHECK:STDOUT:     %X.ref.loc49: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc48.2: type = partial_type %X.ref.loc49 [concrete = constants.%.4b5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: ref %.4b5 = ref_binding i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.bc5 = value_binding_pattern v [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %value.ref: %X = name_ref value, file.%value
// CHECK:STDOUT:   %X.ref.loc68: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc67: type = partial_type %X.ref.loc68 [concrete = constants.%.4b5]
// CHECK:STDOUT:   %.loc66.1: %.4b5 = as_compatible %value.ref
// CHECK:STDOUT:   %.loc66.2: %.4b5 = converted %value.ref, %.loc66.1
// CHECK:STDOUT:   %.loc62.1: type = splice_block %.loc62.2 [concrete = constants.%.4b5] {
// CHECK:STDOUT:     %X.ref.loc63: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc62.2: type = partial_type %X.ref.loc63 [concrete = constants.%.4b5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: %.4b5 = value_binding v, %.loc66.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.408 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %reference.ref: ref %X = name_ref reference, file.%reference [concrete = file.%reference.var]
// CHECK:STDOUT:   %X.ref.loc83: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc82: type = partial_type %X.ref.loc83 [concrete = constants.%.4b5]
// CHECK:STDOUT:   %.loc81.1: ref %.4b5 = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc81.2: ref %.4b5 = converted %reference.ref, %.loc81.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr: %ptr.314 = addr_of %.loc81.2 [concrete = constants.%addr.315]
// CHECK:STDOUT:   %.loc76: type = splice_block %ptr.loc76 [concrete = constants.%ptr.314] {
// CHECK:STDOUT:     %X.ref.loc75: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc74: type = partial_type %X.ref.loc75 [concrete = constants.%.4b5]
// CHECK:STDOUT:     %ptr.loc76: type = ptr_type %.loc74 [concrete = constants.%ptr.314]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.314 = value_binding a, %addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.408 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.2a9 = name_ref ptr, file.%ptr.loc31
// CHECK:STDOUT:   %X.ref.loc97: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc96: type = partial_type %X.ref.loc97 [concrete = constants.%.4b5]
// CHECK:STDOUT:   %ptr.loc98: type = ptr_type %.loc96 [concrete = constants.%ptr.314]
// CHECK:STDOUT:   %.loc95.1: %ptr.314 = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc95.2: %ptr.314 = converted %ptr.ref, %.loc95.1
// CHECK:STDOUT:   %.loc92: type = splice_block %ptr.loc92 [concrete = constants.%ptr.314] {
// CHECK:STDOUT:     %X.ref.loc91: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc90: type = partial_type %X.ref.loc91 [concrete = constants.%.4b5]
// CHECK:STDOUT:     %ptr.loc92: type = ptr_type %.loc90 [concrete = constants.%ptr.314]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.314 = value_binding b, %.loc95.2
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %i.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%i.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %.4b5) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- remove_partial_in_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %.4b5: type = partial_type %X [concrete]
// CHECK:STDOUT:   %Init.type: type = fn_type @Init [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Init: %Init.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.05f: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.05f = ref_binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.05f = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %X = var %i.var_patt
// CHECK:STDOUT:   %Init.ref.loc28: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc22.1: ref %X = splice_block %i.var {}
// CHECK:STDOUT:   %Init.call.loc30: init %.4b5 to %.loc22.1 = call %Init.ref.loc28()
// CHECK:STDOUT:   %.loc22.2: init %X = as_compatible %Init.call.loc30
// CHECK:STDOUT:   %.loc22.3: init %X = converted %Init.call.loc30, %.loc22.2
// CHECK:STDOUT:   assign %i.var, %.loc22.3
// CHECK:STDOUT:   %X.ref.loc26: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %i: ref %X = ref_binding i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %j.patt: %pattern_type.05f = ref_binding_pattern j [concrete]
// CHECK:STDOUT:     %j.var_patt: %pattern_type.05f = var_pattern %j.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %j.var: ref %X = var %j.var_patt
// CHECK:STDOUT:   %Init.ref.loc38: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc32: ref %X = splice_block %j.var {}
// CHECK:STDOUT:   %Init.call.loc40: init %.4b5 to %.loc32 = call %Init.ref.loc38()
// CHECK:STDOUT:   %X.ref.loc42: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc41.1: init %X = as_compatible %Init.call.loc40
// CHECK:STDOUT:   %.loc41.2: init %X = converted %Init.call.loc40, %.loc41.1
// CHECK:STDOUT:   assign %j.var, %.loc41.2
// CHECK:STDOUT:   %X.ref.loc36: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %j: ref %X = ref_binding j, %j.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %k.patt: %pattern_type.05f = ref_binding_pattern k [concrete]
// CHECK:STDOUT:     %k.var_patt: %pattern_type.05f = var_pattern %k.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %k.var: ref %X = var %k.var_patt
// CHECK:STDOUT:   %Init.ref.loc50: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc44: ref %X = splice_block %k.var {}
// CHECK:STDOUT:   %Init.call.loc52: init %.4b5 to %.loc44 = call %Init.ref.loc50()
// CHECK:STDOUT:   %X.ref.loc55: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc54.1: init %X = as_compatible %Init.call.loc52
// CHECK:STDOUT:   %.loc54.2: init %X = converted %Init.call.loc52, %.loc54.1
// CHECK:STDOUT:   assign %k.var, %.loc54.2
// CHECK:STDOUT:   %X.ref.loc48: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %k: ref %X = ref_binding k, %k.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc44: <bound method> = bound_method %k.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc44: init %empty_tuple.type = call %Destroy.Op.bound.loc44(%k.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc32: <bound method> = bound_method %j.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc32: init %empty_tuple.type = call %Destroy.Op.bound.loc32(%j.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc22: <bound method> = bound_method %i.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc22: init %empty_tuple.type = call %Destroy.Op.bound.loc22(%i.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %X) = "no_op";
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
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.05f = value_binding_pattern v [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %value.ref: %.4b5 = name_ref value, file.%value
// CHECK:STDOUT:   %X.ref.loc57: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc56.1: %X = as_compatible %value.ref
// CHECK:STDOUT:   %.loc56.2: %X = converted %value.ref, %.loc56.1
// CHECK:STDOUT:   %X.ref.loc52: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %v: %X = value_binding v, %.loc56.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.37f = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %reference.ref: ref %.4b5 = name_ref reference, file.%reference [concrete = file.%reference.var]
// CHECK:STDOUT:   %X.ref.loc71: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc70.1: ref %X = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc70.2: ref %X = converted %reference.ref, %.loc70.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr: %ptr.2a9 = addr_of %.loc70.2 [concrete = constants.%addr.9cd]
// CHECK:STDOUT:   %.loc64: type = splice_block %ptr.loc64 [concrete = constants.%ptr.2a9] {
// CHECK:STDOUT:     %X.ref.loc63: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc64: type = ptr_type %X.ref.loc63 [concrete = constants.%ptr.2a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.2a9 = value_binding a, %addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.37f = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.314 = name_ref ptr, file.%ptr.loc34
// CHECK:STDOUT:   %X.ref.loc84: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %ptr.loc85: type = ptr_type %X.ref.loc84 [concrete = constants.%ptr.2a9]
// CHECK:STDOUT:   %.loc83.1: %ptr.2a9 = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc83.2: %ptr.2a9 = converted %ptr.ref, %.loc83.1
// CHECK:STDOUT:   %.loc79: type = splice_block %ptr.loc79 [concrete = constants.%ptr.2a9] {
// CHECK:STDOUT:     %X.ref.loc78: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc79: type = ptr_type %X.ref.loc78 [concrete = constants.%ptr.2a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.2a9 = value_binding b, %.loc83.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
