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

// --- fail_todo_add_partial.carbon

library "[[@TEST_NAME]]";

base class X {}

fn Init() -> X;
let value: X = Init();
var reference: X;
let ptr: X* = &reference;

fn Use() {
  // TODO: Should some of these be valid without the `as`?
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_add_partial.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.Destroy` in type `partial X` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   var i: partial X = Init() as partial X;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
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

// CHECK:STDOUT: --- fail_todo_add_partial.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %Init.type: type = fn_type @Init [concrete]
// CHECK:STDOUT:   %Init: %Init.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %.e71: type = partial_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.a53: type = pattern_type %.e71 [concrete]
// CHECK:STDOUT:   %ptr.7b2: type = ptr_type %.e71 [concrete]
// CHECK:STDOUT:   %pattern_type.46e: type = pattern_type %ptr.7b2 [concrete]
// CHECK:STDOUT:   %reference.var: ref %.e71 = var file.%reference.var_patt [concrete]
// CHECK:STDOUT:   %addr.e01: %ptr.7b2 = addr_of %reference.var [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.a53 = binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.a53 = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %.e71 = var %i.var_patt
// CHECK:STDOUT:   %Init.ref: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc18_3: ref %.e71 = splice_block %i.var {}
// CHECK:STDOUT:   %Init.call: init %X = call %Init.ref() to %.loc18_3
// CHECK:STDOUT:   %X.ref.loc18_40: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc18_32: type = partial_type %X.ref.loc18_40 [concrete = constants.%.e71]
// CHECK:STDOUT:   %.loc18_29.1: init %.e71 = as_compatible %Init.call
// CHECK:STDOUT:   %.loc18_29.2: init %.e71 = converted %Init.call, %.loc18_29.1
// CHECK:STDOUT:   assign %i.var, %.loc18_29.2
// CHECK:STDOUT:   %.loc18_10.1: type = splice_block %.loc18_10.2 [concrete = constants.%.e71] {
// CHECK:STDOUT:     %X.ref.loc18_18: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc18_10.2: type = partial_type %X.ref.loc18_18 [concrete = constants.%.e71]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: ref %.e71 = bind_name i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.a53 = binding_pattern v [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %value.ref: %X = name_ref value, file.%value
// CHECK:STDOUT:   %X.ref.loc19_39: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc19_31: type = partial_type %X.ref.loc19_39 [concrete = constants.%.e71]
// CHECK:STDOUT:   %.loc19_28.1: %.e71 = as_compatible %value.ref
// CHECK:STDOUT:   %.loc19_28.2: %.e71 = converted %value.ref, %.loc19_28.1
// CHECK:STDOUT:   %.loc19_10.1: type = splice_block %.loc19_10.2 [concrete = constants.%.e71] {
// CHECK:STDOUT:     %X.ref.loc19_18: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc19_10.2: type = partial_type %X.ref.loc19_18 [concrete = constants.%.e71]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: %.e71 = bind_name v, %.loc19_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.46e = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %reference.ref: ref %X = name_ref reference, file.%reference [concrete = file.%reference.var]
// CHECK:STDOUT:   %X.ref.loc20_46: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc20_38: type = partial_type %X.ref.loc20_46 [concrete = constants.%.e71]
// CHECK:STDOUT:   %.loc20_35.1: ref %.e71 = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc20_35.2: ref %.e71 = converted %reference.ref, %.loc20_35.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr: %ptr.7b2 = addr_of %.loc20_35.2 [concrete = constants.%addr.e01]
// CHECK:STDOUT:   %.loc20_19: type = splice_block %ptr.loc20 [concrete = constants.%ptr.7b2] {
// CHECK:STDOUT:     %X.ref.loc20_18: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc20_10: type = partial_type %X.ref.loc20_18 [concrete = constants.%.e71]
// CHECK:STDOUT:     %ptr.loc20: type = ptr_type %.loc20_10 [concrete = constants.%ptr.7b2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.7b2 = bind_name a, %addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.46e = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.d17 = name_ref ptr, file.%ptr.loc9_5
// CHECK:STDOUT:   %X.ref.loc21_38: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc21_30: type = partial_type %X.ref.loc21_38 [concrete = constants.%.e71]
// CHECK:STDOUT:   %ptr.loc21_39: type = ptr_type %.loc21_30 [concrete = constants.%ptr.7b2]
// CHECK:STDOUT:   %.loc21_27.1: %ptr.7b2 = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc21_27.2: %ptr.7b2 = converted %ptr.ref, %.loc21_27.1
// CHECK:STDOUT:   %.loc21_19: type = splice_block %ptr.loc21_19 [concrete = constants.%ptr.7b2] {
// CHECK:STDOUT:     %X.ref.loc21_18: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc21_10: type = partial_type %X.ref.loc21_18 [concrete = constants.%.e71]
// CHECK:STDOUT:     %ptr.loc21_19: type = ptr_type %.loc21_10 [concrete = constants.%ptr.7b2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.7b2 = bind_name b, %.loc21_27.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %.e71: type = partial_type %X [concrete]
// CHECK:STDOUT:   %Init.type: type = fn_type @Init [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Init: %Init.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.019: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %X, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.dc1: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.177: %AggregateT.as_type.as.Destroy.impl.Op.type.dc1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.019 = binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.019 = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %X = var %i.var_patt
// CHECK:STDOUT:   %Init.ref.loc18: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc18_19: ref %.e71 = temporary_storage
// CHECK:STDOUT:   %Init.call.loc18: init %.e71 = call %Init.ref.loc18() to %.loc18_19
// CHECK:STDOUT:   %.loc18_3.1: %X = converted %Init.call.loc18, <error> [concrete = <error>]
// CHECK:STDOUT:   assign %i.var, <error>
// CHECK:STDOUT:   %X.ref.loc18: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %i: ref %X = bind_name i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %j.patt: %pattern_type.019 = binding_pattern j [concrete]
// CHECK:STDOUT:     %j.var_patt: %pattern_type.019 = var_pattern %j.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %j.var: ref %X = var %j.var_patt
// CHECK:STDOUT:   %Init.ref.loc26: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc26_19: ref %.e71 = temporary_storage
// CHECK:STDOUT:   %Init.call.loc26: init %.e71 = call %Init.ref.loc26() to %.loc26_19
// CHECK:STDOUT:   %X.ref.loc26_24: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc26_21: %X = converted %Init.call.loc26, <error> [concrete = <error>]
// CHECK:STDOUT:   assign %j.var, <error>
// CHECK:STDOUT:   %X.ref.loc26_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %j: ref %X = bind_name j, %j.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %k.patt: %pattern_type.019 = binding_pattern k [concrete]
// CHECK:STDOUT:     %k.var_patt: %pattern_type.019 = var_pattern %k.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %k.var: ref %X = var %k.var_patt
// CHECK:STDOUT:   %Init.ref.loc34: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc34_19: ref %.e71 = temporary_storage
// CHECK:STDOUT:   %Init.call.loc34: init %.e71 = call %Init.ref.loc34() to %.loc34_19
// CHECK:STDOUT:   %X.ref.loc34_31: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc34_28: %X = converted %Init.call.loc34, <error> [concrete = <error>]
// CHECK:STDOUT:   assign %k.var, <error>
// CHECK:STDOUT:   %X.ref.loc34_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %k: ref %X = bind_name k, %k.var
// CHECK:STDOUT:   %facet_value.loc34: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc34_3: %type_where = converted constants.%X, %facet_value.loc34 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc34: <bound method> = bound_method %k.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.177
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc34: <bound method> = bound_method %k.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc34: %ptr.d17 = addr_of %k.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc34: init %empty_tuple.type = call %bound_method.loc34(%addr.loc34)
// CHECK:STDOUT:   %facet_value.loc26: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc26_3: %type_where = converted constants.%X, %facet_value.loc26 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc26: <bound method> = bound_method %j.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.177
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc26: <bound method> = bound_method %j.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc26: %ptr.d17 = addr_of %j.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc26: init %empty_tuple.type = call %bound_method.loc26(%addr.loc26)
// CHECK:STDOUT:   %facet_value.loc18: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc18_3.2: %type_where = converted constants.%X, %facet_value.loc18 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc18: <bound method> = bound_method %i.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.177
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc18: <bound method> = bound_method %i.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc18: %ptr.d17 = addr_of %i.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc18: init %empty_tuple.type = call %bound_method.loc18(%addr.loc18)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsafe_remove_partial.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %.e71: type = partial_type %X [concrete]
// CHECK:STDOUT:   %ptr.7b2: type = ptr_type %.e71 [concrete]
// CHECK:STDOUT:   %pattern_type.019: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.1c6: type = pattern_type %ptr.d17 [concrete]
// CHECK:STDOUT:   %reference.var: ref %X = var file.%reference.var_patt [concrete]
// CHECK:STDOUT:   %addr.c6b: %ptr.d17 = addr_of %reference.var [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.019 = binding_pattern v [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %value.ref: %.e71 = name_ref value, file.%value
// CHECK:STDOUT:   %X.ref.loc14_30: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc14_27.1: %X = as_compatible %value.ref
// CHECK:STDOUT:   %.loc14_27.2: %X = converted %value.ref, %.loc14_27.1
// CHECK:STDOUT:   %X.ref.loc14_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %v: %X = bind_name v, %.loc14_27.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.1c6 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %reference.ref: ref %.e71 = name_ref reference, file.%reference [concrete = file.%reference.var]
// CHECK:STDOUT:   %X.ref.loc15_37: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc15_34.1: ref %X = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc15_34.2: ref %X = converted %reference.ref, %.loc15_34.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr: %ptr.d17 = addr_of %.loc15_34.2 [concrete = constants.%addr.c6b]
// CHECK:STDOUT:   %.loc15_11: type = splice_block %ptr.loc15 [concrete = constants.%ptr.d17] {
// CHECK:STDOUT:     %X.ref.loc15_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc15: type = ptr_type %X.ref.loc15_10 [concrete = constants.%ptr.d17]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.d17 = bind_name a, %addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.1c6 = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.7b2 = name_ref ptr, file.%ptr.loc10_5
// CHECK:STDOUT:   %X.ref.loc16_29: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %ptr.loc16_30: type = ptr_type %X.ref.loc16_29 [concrete = constants.%ptr.d17]
// CHECK:STDOUT:   %.loc16_26.1: %ptr.d17 = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc16_26.2: %ptr.d17 = converted %ptr.ref, %.loc16_26.1
// CHECK:STDOUT:   %.loc16_11: type = splice_block %ptr.loc16_11 [concrete = constants.%ptr.d17] {
// CHECK:STDOUT:     %X.ref.loc16_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc16_11: type = ptr_type %X.ref.loc16_10 [concrete = constants.%ptr.d17]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.d17 = bind_name b, %.loc16_26.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
