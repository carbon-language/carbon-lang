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
  // CHECK:STDERR: fail_cannot_remove_const.carbon:[[@LINE+4]]:24: note: type `const X` does not implement interface `Core.As(X)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused a: X* = &(reference as X);
  // CHECK:STDERR:                        ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused a: X* = &(reference as X);
  // CHECK:STDERR: fail_cannot_remove_const.carbon:[[@LINE+7]]:22: error: cannot convert expression of type `const X*` to `X*` with `as` [ConversionFailure]
  // CHECK:STDERR:   let unused b: X* = ptr as X*;
  // CHECK:STDERR:                      ^~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_const.carbon:[[@LINE+4]]:22: note: type `const X*` does not implement interface `Core.As(X*)` [MissingImplInMemberAccessInContext]
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
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %.loc43: ref %const = splice_block %i.var {}
// CHECK:STDOUT:   %Init.call: init %X to %.loc43 = call %Init.ref()
// CHECK:STDOUT:   %X.ref.loc55: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %const.loc54: type = const_type %X.ref.loc55 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc53.1: init %const = as_compatible %Init.call
// CHECK:STDOUT:   %.loc53.2: init %const = converted %Init.call, %.loc53.1
// CHECK:STDOUT:   assign %i.var, %.loc53.2
// CHECK:STDOUT:   %.loc47: type = splice_block %const.loc47 [concrete = constants.%const] {
// CHECK:STDOUT:     %X.ref.loc48: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %const.loc47: type = const_type %X.ref.loc48 [concrete = constants.%const]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: ref %const = ref_binding i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.c9e = value_binding_pattern v [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %value.ref: %X = name_ref value, file.%value
// CHECK:STDOUT:   %X.ref.loc67: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %const.loc66: type = const_type %X.ref.loc67 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc65.1: %const = as_compatible %value.ref
// CHECK:STDOUT:   %.loc65.2: %const = converted %value.ref, %.loc65.1
// CHECK:STDOUT:   %.loc61: type = splice_block %const.loc61 [concrete = constants.%const] {
// CHECK:STDOUT:     %X.ref.loc62: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %const.loc61: type = const_type %X.ref.loc62 [concrete = constants.%const]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: %const = value_binding v, %.loc65.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.bf1 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %reference.ref: ref %X = name_ref reference, file.%reference [concrete = file.%reference.var]
// CHECK:STDOUT:   %X.ref.loc82: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %const.loc81: type = const_type %X.ref.loc82 [concrete = constants.%const]
// CHECK:STDOUT:   %.loc80.1: ref %const = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc80.2: ref %const = converted %reference.ref, %.loc80.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr: %ptr.d4c = addr_of %.loc80.2 [concrete = constants.%addr.160]
// CHECK:STDOUT:   %.loc75: type = splice_block %ptr.loc75 [concrete = constants.%ptr.d4c] {
// CHECK:STDOUT:     %X.ref.loc74: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %const.loc73: type = const_type %X.ref.loc74 [concrete = constants.%const]
// CHECK:STDOUT:     %ptr.loc75: type = ptr_type %const.loc73 [concrete = constants.%ptr.d4c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.d4c = value_binding a, %addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.bf1 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.2a9 = name_ref ptr, file.%ptr.loc30
// CHECK:STDOUT:   %X.ref.loc96: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %const.loc95: type = const_type %X.ref.loc96 [concrete = constants.%const]
// CHECK:STDOUT:   %ptr.loc97: type = ptr_type %const.loc95 [concrete = constants.%ptr.d4c]
// CHECK:STDOUT:   %.loc94.1: %ptr.d4c = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc94.2: %ptr.d4c = converted %ptr.ref, %.loc94.1
// CHECK:STDOUT:   %.loc91: type = splice_block %ptr.loc91 [concrete = constants.%ptr.d4c] {
// CHECK:STDOUT:     %X.ref.loc90: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %const.loc89: type = const_type %X.ref.loc90 [concrete = constants.%const]
// CHECK:STDOUT:     %ptr.loc91: type = ptr_type %const.loc89 [concrete = constants.%ptr.d4c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.d4c = value_binding b, %.loc94.2
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %i.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%i.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %const) = "no_op";
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
// CHECK:STDOUT:   %Init.ref: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc31: ref %X = splice_block %i.var {}
// CHECK:STDOUT:   %Init.call: init %const to %.loc31 = call %Init.ref()
// CHECK:STDOUT:   %X.ref.loc41: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc40.1: init %X = as_compatible %Init.call
// CHECK:STDOUT:   %.loc40.2: init %X = converted %Init.call, %.loc40.1
// CHECK:STDOUT:   assign %i.var, %.loc40.2
// CHECK:STDOUT:   %X.ref.loc35: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %i: ref %X = ref_binding i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.05f = value_binding_pattern v [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %value.ref: %const = name_ref value, file.%value
// CHECK:STDOUT:   %X.ref.loc51: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc50.1: %X = as_compatible %value.ref
// CHECK:STDOUT:   %.loc50.2: %X = converted %value.ref, %.loc50.1
// CHECK:STDOUT:   %X.ref.loc47: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %v: %X = value_binding v, %.loc50.2
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %i.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%i.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %X) = "no_op";
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
// CHECK:STDOUT:   %X.ref.loc41: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc40.1: ref %X = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc40.2: ref %X = converted %reference.ref, %.loc40.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr: %ptr.2a9 = addr_of %.loc40.2 [concrete = constants.%addr.806]
// CHECK:STDOUT:   %.loc34: type = splice_block %ptr.loc34 [concrete = constants.%ptr.2a9] {
// CHECK:STDOUT:     %X.ref.loc33: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc34: type = ptr_type %X.ref.loc33 [concrete = constants.%ptr.2a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.2a9 = value_binding a, %addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.37f = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.d4c = name_ref ptr, file.%ptr.loc15
// CHECK:STDOUT:   %X.ref.loc54: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %ptr.loc55: type = ptr_type %X.ref.loc54 [concrete = constants.%ptr.2a9]
// CHECK:STDOUT:   %.loc53.1: %ptr.2a9 = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc53.2: %ptr.2a9 = converted %ptr.ref, %.loc53.1
// CHECK:STDOUT:   %.loc49: type = splice_block %ptr.loc49 [concrete = constants.%ptr.2a9] {
// CHECK:STDOUT:     %X.ref.loc48: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc49: type = ptr_type %X.ref.loc48 [concrete = constants.%ptr.2a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.2a9 = value_binding b, %.loc53.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
