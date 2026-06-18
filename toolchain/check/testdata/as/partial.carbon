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
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Init.type: type = fn_type @Init [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Init: %Init.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.db8: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %.d74: type = partial_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.53b: type = pattern_type %.d74 [concrete]
// CHECK:STDOUT:   %i.patt: %pattern_type.53b = ref_binding_pattern i [concrete]
// CHECK:STDOUT:   %i.var_patt: %pattern_type.53b = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   %v.patt: %pattern_type.53b = value_binding_pattern v [concrete]
// CHECK:STDOUT:   %ptr.44a: type = ptr_type %.d74 [concrete]
// CHECK:STDOUT:   %pattern_type.12c: type = pattern_type %ptr.44a [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.12c = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %reference.var: ref %.d74 = var file.%reference.var_patt [concrete]
// CHECK:STDOUT:   %addr.b93: %ptr.44a = addr_of %reference.var [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.12c = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc14_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.53b = ref_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.53b = var_pattern %i.patt [concrete = constants.%i.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %.d74 = var %i.var_patt
// CHECK:STDOUT:   %Init.ref: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc14_3: ref %.d74 = splice_block %i.var {}
// CHECK:STDOUT:   %Init.call: init %X to %.loc14_3 = call %Init.ref()
// CHECK:STDOUT:   %X.ref.loc14_47: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc14_39: type = partial_type %X.ref.loc14_47 [concrete = constants.%.d74]
// CHECK:STDOUT:   %.loc14_36.1: init %.d74 = as_compatible %Init.call
// CHECK:STDOUT:   %.loc14_36.2: init %.d74 = converted %Init.call, %.loc14_36.1
// CHECK:STDOUT:   assign %i.var, %.loc14_36.2
// CHECK:STDOUT:   %.loc14_17.1: type = splice_block %.loc14_17.2 [concrete = constants.%.d74] {
// CHECK:STDOUT:     %X.ref.loc14_25: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc14_17.2: type = partial_type %X.ref.loc14_25 [concrete = constants.%.d74]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: ref %.d74 = ref_binding i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.53b = value_binding_pattern v [concrete = constants.%v.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %value.ref: %X = name_ref value, file.%value
// CHECK:STDOUT:   %X.ref.loc15_46: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc15_38: type = partial_type %X.ref.loc15_46 [concrete = constants.%.d74]
// CHECK:STDOUT:   %.loc15_35.1: %.d74 = as_compatible %value.ref
// CHECK:STDOUT:   %.loc15_35.2: %.d74 = converted %value.ref, %.loc15_35.1
// CHECK:STDOUT:   %.loc15_17.1: type = splice_block %.loc15_17.2 [concrete = constants.%.d74] {
// CHECK:STDOUT:     %X.ref.loc15_25: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc15_17.2: type = partial_type %X.ref.loc15_25 [concrete = constants.%.d74]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: %.d74 = value_binding v, %.loc15_35.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.12c = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %reference.ref: ref %X = name_ref reference, file.%reference [concrete = file.%reference.var]
// CHECK:STDOUT:   %X.ref.loc16_53: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc16_45: type = partial_type %X.ref.loc16_53 [concrete = constants.%.d74]
// CHECK:STDOUT:   %.loc16_42.1: ref %.d74 = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc16_42.2: ref %.d74 = converted %reference.ref, %.loc16_42.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr: %ptr.44a = addr_of %.loc16_42.2 [concrete = constants.%addr.b93]
// CHECK:STDOUT:   %.loc16_26: type = splice_block %ptr.loc16 [concrete = constants.%ptr.44a] {
// CHECK:STDOUT:     %X.ref.loc16_25: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc16_17: type = partial_type %X.ref.loc16_25 [concrete = constants.%.d74]
// CHECK:STDOUT:     %ptr.loc16: type = ptr_type %.loc16_17 [concrete = constants.%ptr.44a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.44a = value_binding a, %addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.12c = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.db8 = name_ref ptr, file.%ptr.loc9_8
// CHECK:STDOUT:   %X.ref.loc17_45: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc17_37: type = partial_type %X.ref.loc17_45 [concrete = constants.%.d74]
// CHECK:STDOUT:   %ptr.loc17_46: type = ptr_type %.loc17_37 [concrete = constants.%ptr.44a]
// CHECK:STDOUT:   %.loc17_34.1: %ptr.44a = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc17_34.2: %ptr.44a = converted %ptr.ref, %.loc17_34.1
// CHECK:STDOUT:   %.loc17_26: type = splice_block %ptr.loc17_26 [concrete = constants.%ptr.44a] {
// CHECK:STDOUT:     %X.ref.loc17_25: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc17_17: type = partial_type %X.ref.loc17_25 [concrete = constants.%.d74]
// CHECK:STDOUT:     %ptr.loc17_26: type = ptr_type %.loc17_17 [concrete = constants.%ptr.44a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.44a = value_binding b, %.loc17_34.2
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %i.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%i.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_3.2(%self.param: ref %.d74) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.d74: type = partial_type %X [concrete]
// CHECK:STDOUT:   %Init.type: type = fn_type @Init [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Init: %Init.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %i.patt: %pattern_type.6cb = ref_binding_pattern i [concrete]
// CHECK:STDOUT:   %i.var_patt: %pattern_type.6cb = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   %j.patt: %pattern_type.6cb = ref_binding_pattern j [concrete]
// CHECK:STDOUT:   %j.var_patt: %pattern_type.6cb = var_pattern %j.patt [concrete]
// CHECK:STDOUT:   %k.patt: %pattern_type.6cb = ref_binding_pattern k [concrete]
// CHECK:STDOUT:   %k.var_patt: %pattern_type.6cb = var_pattern %k.patt [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc12_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.6cb = ref_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.6cb = var_pattern %i.patt [concrete = constants.%i.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %X = var %i.var_patt
// CHECK:STDOUT:   %Init.ref.loc10: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc10_3.1: ref %X = splice_block %i.var {}
// CHECK:STDOUT:   %Init.call.loc10: init %.d74 to %.loc10_3.1 = call %Init.ref.loc10()
// CHECK:STDOUT:   %.loc10_3.2: init %X = as_compatible %Init.call.loc10
// CHECK:STDOUT:   %.loc10_3.3: init %X = converted %Init.call.loc10, %.loc10_3.2
// CHECK:STDOUT:   assign %i.var, %.loc10_3.3
// CHECK:STDOUT:   %X.ref.loc10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %i: ref %X = ref_binding i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %j.patt: %pattern_type.6cb = ref_binding_pattern j [concrete = constants.%j.patt]
// CHECK:STDOUT:     %j.var_patt: %pattern_type.6cb = var_pattern %j.patt [concrete = constants.%j.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %j.var: ref %X = var %j.var_patt
// CHECK:STDOUT:   %Init.ref.loc11: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc11_3: ref %X = splice_block %j.var {}
// CHECK:STDOUT:   %Init.call.loc11: init %.d74 to %.loc11_3 = call %Init.ref.loc11()
// CHECK:STDOUT:   %X.ref.loc11_31: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc11_28.1: init %X = as_compatible %Init.call.loc11
// CHECK:STDOUT:   %.loc11_28.2: init %X = converted %Init.call.loc11, %.loc11_28.1
// CHECK:STDOUT:   assign %j.var, %.loc11_28.2
// CHECK:STDOUT:   %X.ref.loc11_17: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %j: ref %X = ref_binding j, %j.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %k.patt: %pattern_type.6cb = ref_binding_pattern k [concrete = constants.%k.patt]
// CHECK:STDOUT:     %k.var_patt: %pattern_type.6cb = var_pattern %k.patt [concrete = constants.%k.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %k.var: ref %X = var %k.var_patt
// CHECK:STDOUT:   %Init.ref.loc12: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc12_3: ref %X = splice_block %k.var {}
// CHECK:STDOUT:   %Init.call.loc12: init %.d74 to %.loc12_3 = call %Init.ref.loc12()
// CHECK:STDOUT:   %X.ref.loc12_38: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc12_35.1: init %X = as_compatible %Init.call.loc12
// CHECK:STDOUT:   %.loc12_35.2: init %X = converted %Init.call.loc12, %.loc12_35.1
// CHECK:STDOUT:   assign %k.var, %.loc12_35.2
// CHECK:STDOUT:   %X.ref.loc12_17: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %k: ref %X = ref_binding k, %k.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc12: <bound method> = bound_method %k.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc12: init %empty_tuple.type = call %Destroy.Op.bound.loc12(%k.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %j.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%j.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10: <bound method> = bound_method %i.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc10: init %empty_tuple.type = call %Destroy.Op.bound.loc10(%i.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc12_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc12_3.2(%self.param: ref %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsafe_remove_partial.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %.d74: type = partial_type %X [concrete]
// CHECK:STDOUT:   %ptr.44a: type = ptr_type %.d74 [concrete]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %v.patt: %pattern_type.6cb = value_binding_pattern v [concrete]
// CHECK:STDOUT:   %ptr.db8: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.990: type = pattern_type %ptr.db8 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.990 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %reference.var: ref %X = var file.%reference.var_patt [concrete]
// CHECK:STDOUT:   %addr.433: %ptr.db8 = addr_of %reference.var [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.990 = value_binding_pattern b [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.6cb = value_binding_pattern v [concrete = constants.%v.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %value.ref: %.d74 = name_ref value, file.%value
// CHECK:STDOUT:   %X.ref.loc14_37: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc14_34.1: %X = as_compatible %value.ref
// CHECK:STDOUT:   %.loc14_34.2: %X = converted %value.ref, %.loc14_34.1
// CHECK:STDOUT:   %X.ref.loc14_17: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %v: %X = value_binding v, %.loc14_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.990 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %reference.ref: ref %.d74 = name_ref reference, file.%reference [concrete = file.%reference.var]
// CHECK:STDOUT:   %X.ref.loc15_44: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc15_41.1: ref %X = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc15_41.2: ref %X = converted %reference.ref, %.loc15_41.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr: %ptr.db8 = addr_of %.loc15_41.2 [concrete = constants.%addr.433]
// CHECK:STDOUT:   %.loc15_18: type = splice_block %ptr.loc15 [concrete = constants.%ptr.db8] {
// CHECK:STDOUT:     %X.ref.loc15_17: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc15: type = ptr_type %X.ref.loc15_17 [concrete = constants.%ptr.db8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.db8 = value_binding a, %addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.990 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.44a = name_ref ptr, file.%ptr.loc10_8
// CHECK:STDOUT:   %X.ref.loc16_36: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %ptr.loc16_37: type = ptr_type %X.ref.loc16_36 [concrete = constants.%ptr.db8]
// CHECK:STDOUT:   %.loc16_33.1: %ptr.db8 = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc16_33.2: %ptr.db8 = converted %ptr.ref, %.loc16_33.1
// CHECK:STDOUT:   %.loc16_18: type = splice_block %ptr.loc16_18 [concrete = constants.%ptr.db8] {
// CHECK:STDOUT:     %X.ref.loc16_17: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc16_18: type = ptr_type %X.ref.loc16_17 [concrete = constants.%ptr.db8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.db8 = value_binding b, %.loc16_33.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
