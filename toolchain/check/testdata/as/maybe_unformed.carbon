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
  let unused a: Core.MaybeUnformed(X)* = &(reference as Core.MaybeUnformed(X));
  let unused b: Core.MaybeUnformed(X)* = ptr as Core.MaybeUnformed(X)*;
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
  // CHECK:STDERR: fail_todo_add_unformed_nonref.carbon:[[@LINE+7]]:41: error: cannot convert expression of type `X` to `Core.MaybeUnformed(X)` with `as` [ConversionFailure]
  // CHECK:STDERR:   var unused i: Core.MaybeUnformed(X) = Init() as Core.MaybeUnformed(X);
  // CHECK:STDERR:                                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_add_unformed_nonref.carbon:[[@LINE+4]]:41: note: type `X` does not implement interface `Core.As(Core.MaybeUnformed(X))` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused i: Core.MaybeUnformed(X) = Init() as Core.MaybeUnformed(X);
  // CHECK:STDERR:                                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused i: Core.MaybeUnformed(X) = Init() as Core.MaybeUnformed(X);
  // CHECK:STDERR: fail_todo_add_unformed_nonref.carbon:[[@LINE+7]]:41: error: cannot convert expression of type `X` to `Core.MaybeUnformed(X)` with `as` [ConversionFailure]
  // CHECK:STDERR:   let unused v: Core.MaybeUnformed(X) = value as Core.MaybeUnformed(X);
  // CHECK:STDERR:                                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_add_unformed_nonref.carbon:[[@LINE+4]]:41: note: type `X` does not implement interface `Core.As(Core.MaybeUnformed(X))` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused v: Core.MaybeUnformed(X) = value as Core.MaybeUnformed(X);
  // CHECK:STDERR:                                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused v: Core.MaybeUnformed(X) = value as Core.MaybeUnformed(X);
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
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+7]]:21: error: cannot convert expression of type `Core.MaybeUnformed(X)` to `X` with `as` [ConversionFailure]
  // CHECK:STDERR:   var unused i: X = Init() as X;
  // CHECK:STDERR:                     ^~~~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+4]]:21: note: type `Core.MaybeUnformed(X)` does not implement interface `Core.As(X)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused i: X = Init() as X;
  // CHECK:STDERR:                     ^~~~~~~~~~~
  // CHECK:STDERR:
  var unused i: X = Init() as X;
  // TODO: The diagnostic should explain that the reason we can't perform this
  // conversion is due to the expression category.
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+7]]:21: error: cannot convert expression of type `Core.MaybeUnformed(X)` to `X` with `unsafe as` [ConversionFailure]
  // CHECK:STDERR:   var unused j: X = Init() unsafe as X;
  // CHECK:STDERR:                     ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+4]]:21: note: type `Core.MaybeUnformed(X)` does not implement interface `Core.UnsafeAs(X)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused j: X = Init() unsafe as X;
  // CHECK:STDERR:                     ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused j: X = Init() unsafe as X;
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+7]]:21: error: cannot convert expression of type `Core.MaybeUnformed(X)` to `X` with `as` [ConversionFailure]
  // CHECK:STDERR:   let unused v: X = value as X;
  // CHECK:STDERR:                     ^~~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+4]]:21: note: type `Core.MaybeUnformed(X)` does not implement interface `Core.As(X)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused v: X = value as X;
  // CHECK:STDERR:                     ^~~~~~~~~~
  // CHECK:STDERR:
  let unused v: X = value as X;
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+7]]:24: error: cannot convert expression of type `Core.MaybeUnformed(X)` to `X` with `as` [ConversionFailure]
  // CHECK:STDERR:   let unused a: X* = &(reference as X);
  // CHECK:STDERR:                        ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+4]]:24: note: type `Core.MaybeUnformed(X)` does not implement interface `Core.As(X)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused a: X* = &(reference as X);
  // CHECK:STDERR:                        ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused a: X* = &(reference as X);
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+7]]:22: error: cannot convert expression of type `Core.MaybeUnformed(X)*` to `X*` with `as` [ConversionFailure]
  // CHECK:STDERR:   let unused b: X* = ptr as X*;
  // CHECK:STDERR:                      ^~~~~~~~~
  // CHECK:STDERR: fail_cannot_remove_unformed.carbon:[[@LINE+4]]:22: note: type `Core.MaybeUnformed(X)*` does not implement interface `Core.As(X*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused b: X* = ptr as X*;
  // CHECK:STDERR:                      ^~~~~~~~~
  // CHECK:STDERR:
  let unused b: X* = ptr as X*;
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
  let unused v: X = value unsafe as X;
  let unused a: X* = &(reference unsafe as X);
  let unused b: X* = ptr unsafe as X*;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- add_unformed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %ptr.2a9: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %MaybeUnformed.type: type = generic_class_type @MaybeUnformed [concrete]
// CHECK:STDOUT:   %MaybeUnformed.generic: %MaybeUnformed.type = struct_value () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.b49: type = class_type @MaybeUnformed, @MaybeUnformed(%X) [concrete]
// CHECK:STDOUT:   %ptr.045: type = ptr_type %MaybeUnformed.b49 [concrete]
// CHECK:STDOUT:   %pattern_type.396: type = pattern_type %ptr.045 [concrete]
// CHECK:STDOUT:   %reference.var: ref %MaybeUnformed.b49 = var file.%reference.var_patt [concrete]
// CHECK:STDOUT:   %addr.25c: %ptr.045 = addr_of %reference.var [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     .MaybeUnformed = %Core.MaybeUnformed
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.MaybeUnformed: %MaybeUnformed.type = import_ref Core//prelude/parts/maybe_unformed, MaybeUnformed, loaded [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.396 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %reference.ref: ref %X = name_ref reference, file.%reference [concrete = file.%reference.var]
// CHECK:STDOUT:   %Core.ref.loc43: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %MaybeUnformed.ref.loc44: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:   %X.ref.loc47: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %MaybeUnformed.loc48: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.b49]
// CHECK:STDOUT:   %.loc42.1: ref %MaybeUnformed.b49 = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc42.2: ref %MaybeUnformed.b49 = converted %reference.ref, %.loc42.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr: %ptr.045 = addr_of %.loc42.2 [concrete = constants.%addr.25c]
// CHECK:STDOUT:   %.loc37: type = splice_block %ptr.loc37 [concrete = constants.%ptr.045] {
// CHECK:STDOUT:     %Core.ref.loc31: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %MaybeUnformed.ref.loc32: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:     %X.ref.loc35: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %MaybeUnformed.loc36: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.b49]
// CHECK:STDOUT:     %ptr.loc37: type = ptr_type %MaybeUnformed.loc36 [concrete = constants.%ptr.045]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.045 = value_binding a, %addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.396 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.2a9 = name_ref ptr, file.%ptr.loc14
// CHECK:STDOUT:   %Core.ref.loc65: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %MaybeUnformed.ref.loc66: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:   %X.ref.loc69: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %MaybeUnformed.loc70: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.b49]
// CHECK:STDOUT:   %ptr.loc71: type = ptr_type %MaybeUnformed.loc70 [concrete = constants.%ptr.045]
// CHECK:STDOUT:   %.loc64.1: %ptr.045 = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc64.2: %ptr.045 = converted %ptr.ref, %.loc64.1
// CHECK:STDOUT:   %.loc61: type = splice_block %ptr.loc61 [concrete = constants.%ptr.045] {
// CHECK:STDOUT:     %Core.ref.loc55: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %MaybeUnformed.ref.loc56: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:     %X.ref.loc59: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %MaybeUnformed.loc60: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.b49]
// CHECK:STDOUT:     %ptr.loc61: type = ptr_type %MaybeUnformed.loc60 [concrete = constants.%ptr.045]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.045 = value_binding b, %.loc64.2
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
// CHECK:STDOUT:   %MaybeUnformed.b49: type = class_type @MaybeUnformed, @MaybeUnformed(%X) [concrete]
// CHECK:STDOUT:   %pattern_type.de9: type = pattern_type %MaybeUnformed.b49 [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:     %i.patt: %pattern_type.de9 = ref_binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.de9 = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %MaybeUnformed.b49 = var %i.var_patt
// CHECK:STDOUT:   %Init.ref: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc42: ref %X = temporary_storage
// CHECK:STDOUT:   %Init.call: init %X to %.loc42 = call %Init.ref()
// CHECK:STDOUT:   %Core.ref.loc44: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %MaybeUnformed.ref.loc45: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:   %X.ref.loc48: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %MaybeUnformed.loc49: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.b49]
// CHECK:STDOUT:   %.loc43: %MaybeUnformed.b49 = converted %Init.call, <error> [concrete = <error>]
// CHECK:STDOUT:   assign %i.var, <error>
// CHECK:STDOUT:   %.loc38: type = splice_block %MaybeUnformed.loc38 [concrete = constants.%MaybeUnformed.b49] {
// CHECK:STDOUT:     %Core.ref.loc33: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %MaybeUnformed.ref.loc34: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:     %X.ref.loc37: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %MaybeUnformed.loc38: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.b49]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: ref %MaybeUnformed.b49 = ref_binding i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.de9 = value_binding_pattern v [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %value.ref: %X = name_ref value, file.%value
// CHECK:STDOUT:   %Core.ref.loc64: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %MaybeUnformed.ref.loc65: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:   %X.ref.loc68: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %MaybeUnformed.loc69: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.b49]
// CHECK:STDOUT:   %.loc63: %MaybeUnformed.b49 = converted %value.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc60: type = splice_block %MaybeUnformed.loc60 [concrete = constants.%MaybeUnformed.b49] {
// CHECK:STDOUT:     %Core.ref.loc55: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %MaybeUnformed.ref.loc56: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:     %X.ref.loc59: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %MaybeUnformed.loc60: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.b49]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: %MaybeUnformed.b49 = value_binding v, <error> [concrete = <error>]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %i.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%i.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %MaybeUnformed.b49) = "no_op";
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
// CHECK:STDOUT:   %MaybeUnformed.b49: type = class_type @MaybeUnformed, @MaybeUnformed(%X) [concrete]
// CHECK:STDOUT:   %ptr.045: type = ptr_type %MaybeUnformed.b49 [concrete]
// CHECK:STDOUT:   %pattern_type.05f: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %ptr.2a9: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.37f: type = pattern_type %ptr.2a9 [concrete]
// CHECK:STDOUT:   %reference.var: ref %X = var file.%reference.var_patt [concrete]
// CHECK:STDOUT:   %addr.6f7: %ptr.2a9 = addr_of %reference.var [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.05f = value_binding_pattern v [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %value.ref: %MaybeUnformed.b49 = name_ref value, file.%value
// CHECK:STDOUT:   %X.ref.loc72: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc71.1: %X = as_compatible %value.ref
// CHECK:STDOUT:   %.loc71.2: %X = converted %value.ref, %.loc71.1
// CHECK:STDOUT:   %X.ref.loc67: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %v: %X = value_binding v, %.loc71.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.37f = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %reference.ref: ref %MaybeUnformed.b49 = name_ref reference, file.%reference [concrete = file.%reference.var]
// CHECK:STDOUT:   %X.ref.loc86: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc85.1: ref %X = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc85.2: ref %X = converted %reference.ref, %.loc85.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr: %ptr.2a9 = addr_of %.loc85.2 [concrete = constants.%addr.6f7]
// CHECK:STDOUT:   %.loc79: type = splice_block %ptr.loc79 [concrete = constants.%ptr.2a9] {
// CHECK:STDOUT:     %X.ref.loc78: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc79: type = ptr_type %X.ref.loc78 [concrete = constants.%ptr.2a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.2a9 = value_binding a, %addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.37f = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.045 = name_ref ptr, file.%ptr.loc45
// CHECK:STDOUT:   %X.ref.loc99: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %ptr.loc100: type = ptr_type %X.ref.loc99 [concrete = constants.%ptr.2a9]
// CHECK:STDOUT:   %.loc98.1: %ptr.2a9 = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc98.2: %ptr.2a9 = converted %ptr.ref, %.loc98.1
// CHECK:STDOUT:   %.loc94: type = splice_block %ptr.loc94 [concrete = constants.%ptr.2a9] {
// CHECK:STDOUT:     %X.ref.loc93: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc94: type = ptr_type %X.ref.loc93 [concrete = constants.%ptr.2a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.2a9 = value_binding b, %.loc98.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
