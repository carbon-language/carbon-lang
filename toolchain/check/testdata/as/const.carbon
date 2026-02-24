// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/const.carbon

// --- add_const.carbon

library "[[@TEST_NAME]]";

class X {}

fn Init() -> X;
let value: X = Init();
var reference: X;
let ptr: X* = &reference;

fn Use() {
  // TODO: Should some of these be valid without the `as`?
  //@dump-sem-ir-begin
  var unused i: const X = Init() as const X;
  let unused v: const X = value as const X;
  let unused a: const X* = &(reference as const X);
  let unused b: const X* = ptr as const X*;
  //@dump-sem-ir-end
}

// --- remove_const.carbon

library "[[@TEST_NAME]]";

class X {}

fn Init() -> const X;
let value: const X = Init();

fn Use() {
  // TODO: Should some of these be valid without the `as`?
  //@dump-sem-ir-begin
  var unused i: X = Init() as X;
  let unused v: X = value as X;
  //@dump-sem-ir-end
}

// --- fail_cannot_remove_const.carbon

library "[[@TEST_NAME]]";

class X {}

var reference: const X;
let ptr: const X* = &reference;

fn Use() {
  // CHECK:STDERR: fail_cannot_remove_const.carbon:[[@LINE+7]]:24: error: cannot convert expression of type `const X` to `X` with `as` [ConversionFailure]
  // CHECK:STDERR:   let unused a: X* = &(reference as X);
  // CHECK:STDERR:                        ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_const.carbon:[[@LINE+4]]:24: note: type `const X` does not implement interface `Core.As(X)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let unused a: X* = &(reference as X);
  // CHECK:STDERR:                        ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused a: X* = &(reference as X);
  // CHECK:STDERR: fail_cannot_remove_const.carbon:[[@LINE+7]]:22: error: cannot convert expression of type `const X*` to `X*` with `as` [ConversionFailure]
  // CHECK:STDERR:   let unused b: X* = ptr as X*;
  // CHECK:STDERR:                      ^~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_const.carbon:[[@LINE+4]]:22: note: type `const X*` does not implement interface `Core.As(X*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let unused b: X* = ptr as X*;
  // CHECK:STDERR:                      ^~~~~~~~~
  // CHECK:STDERR:
  let unused b: X* = ptr as X*;
}

// --- unsafe_remove_const.carbon

library "[[@TEST_NAME]]";

class X {}

var reference: const X;
let ptr: const X* = &reference;

fn Use() {
  //@dump-sem-ir-begin
  let unused a: X* = &(reference unsafe as X);
  let unused b: X* = ptr unsafe as X*;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- add_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %Init.type: type = fn_type @Init [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Init: %Init.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.2a9: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %const: type = const_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.c9e: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %ptr.d4c: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %pattern_type.bf1: type = pattern_type %ptr.d4c [concrete]
// CHECK:STDOUT:   %reference.var: ref %const = var file.%reference.var_patt [concrete]
// CHECK:STDOUT:   %addr.160: %ptr.d4c = addr_of %reference.var [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.c9e = ref_binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.c9e = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %const = var %i.var_patt
// CHECK:STDOUT:   %Init.ref: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc14_3: ref %const = splice_block %i.var {}
// CHECK:STDOUT:   %Init.call: init %X to %.loc14_3 = call %Init.ref()
// CHECK:STDOUT:   %X.ref.loc14_43: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %const.loc14_37: type = const_type %X.ref.loc14_43 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc14_34.1: init %const = as_compatible %Init.call
// CHECK:STDOUT:   %.loc14_34.2: init %const = converted %Init.call, %.loc14_34.1
// CHECK:STDOUT:   assign %i.var, %.loc14_34.2
// CHECK:STDOUT:   %.loc14_17: type = splice_block %const.loc14_17 [concrete = constants.%const] {
// CHECK:STDOUT:     %X.ref.loc14_23: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %const.loc14_17: type = const_type %X.ref.loc14_23 [concrete = constants.%const]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: ref %const = ref_binding i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.c9e = value_binding_pattern v [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %value.ref: %X = name_ref value, file.%value
// CHECK:STDOUT:   %X.ref.loc15_42: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %const.loc15_36: type = const_type %X.ref.loc15_42 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc15_33.1: %const = as_compatible %value.ref
// CHECK:STDOUT:   %.loc15_33.2: %const = converted %value.ref, %.loc15_33.1
// CHECK:STDOUT:   %.loc15_17: type = splice_block %const.loc15_17 [concrete = constants.%const] {
// CHECK:STDOUT:     %X.ref.loc15_23: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %const.loc15_17: type = const_type %X.ref.loc15_23 [concrete = constants.%const]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: %const = value_binding v, %.loc15_33.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.bf1 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %reference.ref: ref %X = name_ref reference, file.%reference [concrete = file.%reference.var]
// CHECK:STDOUT:   %X.ref.loc16_49: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %const.loc16_43: type = const_type %X.ref.loc16_49 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc16_40.1: ref %const = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc16_40.2: ref %const = converted %reference.ref, %.loc16_40.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr: %ptr.d4c = addr_of %.loc16_40.2 [concrete = constants.%addr.160]
// CHECK:STDOUT:   %.loc16_24: type = splice_block %ptr.loc16 [concrete = constants.%ptr.d4c] {
// CHECK:STDOUT:     %X.ref.loc16_23: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %const.loc16_17: type = const_type %X.ref.loc16_23 [concrete = constants.%const]
// CHECK:STDOUT:     %ptr.loc16: type = ptr_type %const.loc16_17 [concrete = constants.%ptr.d4c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.d4c = value_binding a, %addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.bf1 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.2a9 = name_ref ptr, file.%ptr.loc9_5
// CHECK:STDOUT:   %X.ref.loc17_41: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %const.loc17_35: type = const_type %X.ref.loc17_41 [concrete = constants.%const]
// CHECK:STDOUT:   %ptr.loc17_42: type = ptr_type %const.loc17_35 [concrete = constants.%ptr.d4c]
// CHECK:STDOUT:   %.loc17_32.1: %ptr.d4c = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc17_32.2: %ptr.d4c = converted %ptr.ref, %.loc17_32.1
// CHECK:STDOUT:   %.loc17_24: type = splice_block %ptr.loc17_24 [concrete = constants.%ptr.d4c] {
// CHECK:STDOUT:     %X.ref.loc17_23: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %const.loc17_17: type = const_type %X.ref.loc17_23 [concrete = constants.%const]
// CHECK:STDOUT:     %ptr.loc17_24: type = ptr_type %const.loc17_17 [concrete = constants.%ptr.d4c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.d4c = value_binding b, %.loc17_32.2
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %i.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%i.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: ref %const) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- remove_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %const: type = const_type %X [concrete]
// CHECK:STDOUT:   %Init.type: type = fn_type @Init [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Init: %Init.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.05f: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.05f = ref_binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.05f = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %X = var %i.var_patt
// CHECK:STDOUT:   %Init.ref: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc12_3: ref %X = splice_block %i.var {}
// CHECK:STDOUT:   %Init.call: init %const to %.loc12_3 = call %Init.ref()
// CHECK:STDOUT:   %X.ref.loc12_31: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc12_28.1: init %X = as_compatible %Init.call
// CHECK:STDOUT:   %.loc12_28.2: init %X = converted %Init.call, %.loc12_28.1
// CHECK:STDOUT:   assign %i.var, %.loc12_28.2
// CHECK:STDOUT:   %X.ref.loc12_17: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %i: ref %X = ref_binding i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.05f = value_binding_pattern v [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %value.ref: %const = name_ref value, file.%value
// CHECK:STDOUT:   %X.ref.loc13_30: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc13_27.1: %X = as_compatible %value.ref
// CHECK:STDOUT:   %.loc13_27.2: %X = converted %value.ref, %.loc13_27.1
// CHECK:STDOUT:   %X.ref.loc13_17: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %v: %X = value_binding v, %.loc13_27.2
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %i.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%i.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: ref %X) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsafe_remove_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %const: type = const_type %X [concrete]
// CHECK:STDOUT:   %ptr.d4c: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %ptr.2a9: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.37f: type = pattern_type %ptr.2a9 [concrete]
// CHECK:STDOUT:   %reference.var: ref %X = var file.%reference.var_patt [concrete]
// CHECK:STDOUT:   %addr.806: %ptr.2a9 = addr_of %reference.var [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.37f = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %reference.ref: ref %const = name_ref reference, file.%reference [concrete = file.%reference.var]
// CHECK:STDOUT:   %X.ref.loc11_44: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc11_41.1: ref %X = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc11_41.2: ref %X = converted %reference.ref, %.loc11_41.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr: %ptr.2a9 = addr_of %.loc11_41.2 [concrete = constants.%addr.806]
// CHECK:STDOUT:   %.loc11_18: type = splice_block %ptr.loc11 [concrete = constants.%ptr.2a9] {
// CHECK:STDOUT:     %X.ref.loc11_17: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc11: type = ptr_type %X.ref.loc11_17 [concrete = constants.%ptr.2a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.2a9 = value_binding a, %addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.37f = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.d4c = name_ref ptr, file.%ptr.loc7_5
// CHECK:STDOUT:   %X.ref.loc12_36: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %ptr.loc12_37: type = ptr_type %X.ref.loc12_36 [concrete = constants.%ptr.2a9]
// CHECK:STDOUT:   %.loc12_33.1: %ptr.2a9 = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc12_33.2: %ptr.2a9 = converted %ptr.ref, %.loc12_33.1
// CHECK:STDOUT:   %.loc12_18: type = splice_block %ptr.loc12_18 [concrete = constants.%ptr.2a9] {
// CHECK:STDOUT:     %X.ref.loc12_17: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc12_18: type = ptr_type %X.ref.loc12_17 [concrete = constants.%ptr.2a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.2a9 = value_binding b, %.loc12_33.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
