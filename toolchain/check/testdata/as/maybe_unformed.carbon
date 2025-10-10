// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/unformed.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/maybe_unformed.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/maybe_unformed.carbon

// --- add_unformed.carbon

library "[[@TEST_NAME]]";

class X {}

var reference: X;
let ptr: X* = &reference;

fn Use() {
  // TODO: Should some of these be valid without the `as`?
  //@dump-sem-ir-begin
  let a: Core.MaybeUnformed(X)* = &(reference as Core.MaybeUnformed(X));
  let b: Core.MaybeUnformed(X)* = ptr as Core.MaybeUnformed(X)*;
  //@dump-sem-ir-end
}

// --- fail_todo_add_unformed_nonref.carbon

library "[[@TEST_NAME]]";

class X {}

fn Init() -> X;
let value: X = Init();

fn Use() {
  // TODO: These should probably be valid.
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_add_unformed_nonref.carbon:[[@LINE+7]]:34: error: cannot convert expression of type `X` to `Core.MaybeUnformed(X)` with `as` [ConversionFailure]
  // CHECK:STDERR:   var i: Core.MaybeUnformed(X) = Init() as Core.MaybeUnformed(X);
  // CHECK:STDERR:                                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_add_unformed_nonref.carbon:[[@LINE+4]]:34: note: type `X` does not implement interface `Core.As(Core.MaybeUnformed(X))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var i: Core.MaybeUnformed(X) = Init() as Core.MaybeUnformed(X);
  // CHECK:STDERR:                                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var i: Core.MaybeUnformed(X) = Init() as Core.MaybeUnformed(X);
  // CHECK:STDERR: fail_todo_add_unformed_nonref.carbon:[[@LINE+7]]:34: error: cannot convert expression of type `X` to `Core.MaybeUnformed(X)` with `as` [ConversionFailure]
  // CHECK:STDERR:   let v: Core.MaybeUnformed(X) = value as Core.MaybeUnformed(X);
  // CHECK:STDERR:                                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_add_unformed_nonref.carbon:[[@LINE+4]]:34: note: type `X` does not implement interface `Core.As(Core.MaybeUnformed(X))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let v: Core.MaybeUnformed(X) = value as Core.MaybeUnformed(X);
  // CHECK:STDERR:                                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let v: Core.MaybeUnformed(X) = value as Core.MaybeUnformed(X);
  //@dump-sem-ir-end
}

// --- fail_cannot_remove_unformed.carbon

library "[[@TEST_NAME]]";

class X {}

fn Init() -> Core.MaybeUnformed(X);
let value: Core.MaybeUnformed(X) = Init();
var reference: Core.MaybeUnformed(X);
let ptr: Core.MaybeUnformed(X)* = &reference;

fn Use() {
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+7]]:14: error: cannot convert expression of type `Core.MaybeUnformed(X)` to `X` with `as` [ConversionFailure]
  // CHECK:STDERR:   var i: X = Init() as X;
  // CHECK:STDERR:              ^~~~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+4]]:14: note: type `Core.MaybeUnformed(X)` does not implement interface `Core.As(X)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var i: X = Init() as X;
  // CHECK:STDERR:              ^~~~~~~~~~~
  // CHECK:STDERR:
  var i: X = Init() as X;
  // TODO: The diagnostic should explain that the reason we can't perform this
  // conversion is due to the expression category.
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+7]]:14: error: cannot convert expression of type `Core.MaybeUnformed(X)` to `X` with `unsafe as` [ConversionFailure]
  // CHECK:STDERR:   var j: X = Init() unsafe as X;
  // CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+4]]:14: note: type `Core.MaybeUnformed(X)` does not implement interface `Core.UnsafeAs(X)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var j: X = Init() unsafe as X;
  // CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var j: X = Init() unsafe as X;
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+7]]:14: error: cannot convert expression of type `Core.MaybeUnformed(X)` to `X` with `as` [ConversionFailure]
  // CHECK:STDERR:   let v: X = value as X;
  // CHECK:STDERR:              ^~~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+4]]:14: note: type `Core.MaybeUnformed(X)` does not implement interface `Core.As(X)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let v: X = value as X;
  // CHECK:STDERR:              ^~~~~~~~~~
  // CHECK:STDERR:
  let v: X = value as X;
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+7]]:17: error: cannot convert expression of type `Core.MaybeUnformed(X)` to `X` with `as` [ConversionFailure]
  // CHECK:STDERR:   let a: X* = &(reference as X);
  // CHECK:STDERR:                 ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+4]]:17: note: type `Core.MaybeUnformed(X)` does not implement interface `Core.As(X)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let a: X* = &(reference as X);
  // CHECK:STDERR:                 ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: X* = &(reference as X);
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+7]]:15: error: cannot convert expression of type `Core.MaybeUnformed(X)*` to `X*` with `as` [ConversionFailure]
  // CHECK:STDERR:   let b: X* = ptr as X*;
  // CHECK:STDERR:               ^~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+4]]:15: note: type `Core.MaybeUnformed(X)*` does not implement interface `Core.As(X*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let b: X* = ptr as X*;
  // CHECK:STDERR:               ^~~~~~~~~
  // CHECK:STDERR:
  let b: X* = ptr as X*;
}

// --- unsafe_remove_unformed.carbon

library "[[@TEST_NAME]]";

class X {}

fn Init() -> Core.MaybeUnformed(X);
let value: Core.MaybeUnformed(X) = Init();
var reference: Core.MaybeUnformed(X);
let ptr: Core.MaybeUnformed(X)* = &reference;

fn Use() {
  //@dump-sem-ir-begin
  let v: X = value unsafe as X;
  let a: X* = &(reference unsafe as X);
  let b: X* = ptr unsafe as X*;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- add_unformed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %MaybeUnformed.type: type = generic_class_type @MaybeUnformed [concrete]
// CHECK:STDOUT:   %MaybeUnformed.generic: %MaybeUnformed.type = struct_value () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.275: type = class_type @MaybeUnformed, @MaybeUnformed(%X) [concrete]
// CHECK:STDOUT:   %ptr.58e: type = ptr_type %MaybeUnformed.275 [concrete]
// CHECK:STDOUT:   %pattern_type.460: type = pattern_type %ptr.58e [concrete]
// CHECK:STDOUT:   %reference.var: ref %MaybeUnformed.275 = var file.%reference.var_patt [concrete]
// CHECK:STDOUT:   %addr.10a: %ptr.58e = addr_of %reference.var [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .MaybeUnformed = %Core.MaybeUnformed
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.MaybeUnformed: %MaybeUnformed.type = import_ref Core//prelude/parts/maybe_unformed, MaybeUnformed, loaded [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.460 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %reference.ref: ref %X = name_ref reference, file.%reference [concrete = file.%reference.var]
// CHECK:STDOUT:   %Core.ref.loc12_50: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %MaybeUnformed.ref.loc12_54: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:   %X.ref.loc12_69: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %MaybeUnformed.loc12_70: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.275]
// CHECK:STDOUT:   %.loc12_47.1: ref %MaybeUnformed.275 = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc12_47.2: ref %MaybeUnformed.275 = converted %reference.ref, %.loc12_47.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr: %ptr.58e = addr_of %.loc12_47.2 [concrete = constants.%addr.10a]
// CHECK:STDOUT:   %.loc12_31: type = splice_block %ptr.loc12 [concrete = constants.%ptr.58e] {
// CHECK:STDOUT:     %Core.ref.loc12_10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %MaybeUnformed.ref.loc12_14: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:     %X.ref.loc12_29: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %MaybeUnformed.loc12_30: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.275]
// CHECK:STDOUT:     %ptr.loc12: type = ptr_type %MaybeUnformed.loc12_30 [concrete = constants.%ptr.58e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.58e = bind_name a, %addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.460 = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.d17 = name_ref ptr, file.%ptr.loc7_5
// CHECK:STDOUT:   %Core.ref.loc13_42: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %MaybeUnformed.ref.loc13_46: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:   %X.ref.loc13_61: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %MaybeUnformed.loc13_62: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.275]
// CHECK:STDOUT:   %ptr.loc13_63: type = ptr_type %MaybeUnformed.loc13_62 [concrete = constants.%ptr.58e]
// CHECK:STDOUT:   %.loc13_39.1: %ptr.58e = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc13_39.2: %ptr.58e = converted %ptr.ref, %.loc13_39.1
// CHECK:STDOUT:   %.loc13_31: type = splice_block %ptr.loc13_31 [concrete = constants.%ptr.58e] {
// CHECK:STDOUT:     %Core.ref.loc13_10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %MaybeUnformed.ref.loc13_14: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:     %X.ref.loc13_29: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %MaybeUnformed.loc13_30: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.275]
// CHECK:STDOUT:     %ptr.loc13_31: type = ptr_type %MaybeUnformed.loc13_30 [concrete = constants.%ptr.58e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.58e = bind_name b, %.loc13_39.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_add_unformed_nonref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %Init.type: type = fn_type @Init [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Init: %Init.type = struct_value () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.type: type = generic_class_type @MaybeUnformed [concrete]
// CHECK:STDOUT:   %MaybeUnformed.generic: %MaybeUnformed.type = struct_value () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.275: type = class_type @MaybeUnformed, @MaybeUnformed(%X) [concrete]
// CHECK:STDOUT:   %pattern_type.eb8: type = pattern_type %MaybeUnformed.275 [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %MaybeUnformed.275, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.bef: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.60f: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.bef = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.58e: type = ptr_type %MaybeUnformed.275 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .MaybeUnformed = %Core.MaybeUnformed
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.MaybeUnformed: %MaybeUnformed.type = import_ref Core//prelude/parts/maybe_unformed, MaybeUnformed, loaded [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.eb8 = binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.eb8 = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %MaybeUnformed.275 = var %i.var_patt
// CHECK:STDOUT:   %Init.ref: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc19_39: ref %X = temporary_storage
// CHECK:STDOUT:   %Init.call: init %X = call %Init.ref() to %.loc19_39
// CHECK:STDOUT:   %Core.ref.loc19_44: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %MaybeUnformed.ref.loc19_48: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:   %X.ref.loc19_63: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %MaybeUnformed.loc19_64: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.275]
// CHECK:STDOUT:   %.loc19_41: %MaybeUnformed.275 = converted %Init.call, <error> [concrete = <error>]
// CHECK:STDOUT:   assign %i.var, <error>
// CHECK:STDOUT:   %.loc19_30: type = splice_block %MaybeUnformed.loc19_30 [concrete = constants.%MaybeUnformed.275] {
// CHECK:STDOUT:     %Core.ref.loc19_10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %MaybeUnformed.ref.loc19_14: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:     %X.ref.loc19_29: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %MaybeUnformed.loc19_30: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.275]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: ref %MaybeUnformed.275 = bind_name i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.eb8 = binding_pattern v [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %value.ref: %X = name_ref value, file.%value
// CHECK:STDOUT:   %Core.ref.loc27_43: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %MaybeUnformed.ref.loc27_47: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:   %X.ref.loc27_62: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %MaybeUnformed.loc27_63: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.275]
// CHECK:STDOUT:   %.loc27_40: %MaybeUnformed.275 = converted %value.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc27_30: type = splice_block %MaybeUnformed.loc27_30 [concrete = constants.%MaybeUnformed.275] {
// CHECK:STDOUT:     %Core.ref.loc27_10: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %MaybeUnformed.ref.loc27_14: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:     %X.ref.loc27_29: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %MaybeUnformed.loc27_30: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.275]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: %MaybeUnformed.275 = bind_name v, <error> [concrete = <error>]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %i.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.60f
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %i.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.58e = addr_of %i.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsafe_remove_unformed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %MaybeUnformed.275: type = class_type @MaybeUnformed, @MaybeUnformed(%X) [concrete]
// CHECK:STDOUT:   %ptr.58e: type = ptr_type %MaybeUnformed.275 [concrete]
// CHECK:STDOUT:   %pattern_type.019: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.1c6: type = pattern_type %ptr.d17 [concrete]
// CHECK:STDOUT:   %reference.var: ref %X = var file.%reference.var_patt [concrete]
// CHECK:STDOUT:   %addr.a46: %ptr.d17 = addr_of %reference.var [concrete]
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
// CHECK:STDOUT:   %value.ref: %MaybeUnformed.275 = name_ref value, file.%value
// CHECK:STDOUT:   %X.ref.loc13_30: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc13_27.1: ref %MaybeUnformed.275 = value_as_ref %value.ref
// CHECK:STDOUT:   %.loc13_27.2: ref %X = as_compatible %.loc13_27.1
// CHECK:STDOUT:   %.loc13_27.3: %X = bind_value %.loc13_27.2
// CHECK:STDOUT:   %.loc13_27.4: %X = converted %value.ref, %.loc13_27.3
// CHECK:STDOUT:   %X.ref.loc13_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %v: %X = bind_name v, %.loc13_27.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.1c6 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %reference.ref: ref %MaybeUnformed.275 = name_ref reference, file.%reference [concrete = file.%reference.var]
// CHECK:STDOUT:   %X.ref.loc14_37: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc14_34.1: ref %X = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc14_34.2: ref %X = converted %reference.ref, %.loc14_34.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr: %ptr.d17 = addr_of %.loc14_34.2 [concrete = constants.%addr.a46]
// CHECK:STDOUT:   %.loc14_11: type = splice_block %ptr.loc14 [concrete = constants.%ptr.d17] {
// CHECK:STDOUT:     %X.ref.loc14_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc14: type = ptr_type %X.ref.loc14_10 [concrete = constants.%ptr.d17]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.d17 = bind_name a, %addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.1c6 = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.58e = name_ref ptr, file.%ptr.loc9_5
// CHECK:STDOUT:   %X.ref.loc15_29: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %ptr.loc15_30: type = ptr_type %X.ref.loc15_29 [concrete = constants.%ptr.d17]
// CHECK:STDOUT:   %.loc15_26.1: %ptr.d17 = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc15_26.2: %ptr.d17 = converted %ptr.ref, %.loc15_26.1
// CHECK:STDOUT:   %.loc15_11: type = splice_block %ptr.loc15_11 [concrete = constants.%ptr.d17] {
// CHECK:STDOUT:     %X.ref.loc15_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc15_11: type = ptr_type %X.ref.loc15_10 [concrete = constants.%ptr.d17]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.d17 = bind_name b, %.loc15_26.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
