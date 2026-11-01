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
  // CHECK:STDERR:        ╭─┤ fail_todo_add_unformed_nonref.carbon:[[@LINE+14]]:41
  // CHECK:STDERR: ->     │   var unused i: Core.MaybeUnformed(X) = Init() as Core.MaybeUnformed(X);
  // CHECK:STDERR:        ·                                         ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·                                         ┗━━┫ error: cannot convert expression of
  // CHECK:STDERR:        ·                                         │  ┃        type `X` to
  // CHECK:STDERR:        ·                                         │  ┃        `Core.MaybeUnformed(X)` with
  // CHECK:STDERR:        ·                                         │  ┃        `as`
  // CHECK:STDERR:        ·                                         │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                                         ╰────┤ type `X` does not implement
  // CHECK:STDERR:        ·                                              │ interface
  // CHECK:STDERR:        ·                                              │ `Core.As(Core.MaybeUnformed(X))`
  // CHECK:STDERR:        ·                                              │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var unused i: Core.MaybeUnformed(X) = Init() as Core.MaybeUnformed(X);
  // CHECK:STDERR:        ╭─┤ fail_todo_add_unformed_nonref.carbon:[[@LINE+14]]:41
  // CHECK:STDERR: ->     │   let unused v: Core.MaybeUnformed(X) = value as Core.MaybeUnformed(X);
  // CHECK:STDERR:        ·                                         ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·                                         ┗━━┫ error: cannot convert expression of
  // CHECK:STDERR:        ·                                         │  ┃        type `X` to
  // CHECK:STDERR:        ·                                         │  ┃        `Core.MaybeUnformed(X)` with
  // CHECK:STDERR:        ·                                         │  ┃        `as`
  // CHECK:STDERR:        ·                                         │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                                         ╰────┤ type `X` does not implement
  // CHECK:STDERR:        ·                                              │ interface
  // CHECK:STDERR:        ·                                              │ `Core.As(Core.MaybeUnformed(X))`
  // CHECK:STDERR:        ·                                              │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
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
  // CHECK:STDERR:        ╭─┤ fail_cannot_remove_unformed.carbon:[[@LINE+11]]:21
  // CHECK:STDERR: ->     │   var unused i: X = Init() as X;
  // CHECK:STDERR:        ·                     ┳━━━━━━━━━━
  // CHECK:STDERR:        ·                     ┗━━┫ error: cannot convert expression of type
  // CHECK:STDERR:        ·                     │  ┃        `Core.MaybeUnformed(X)` to `X` with `as`
  // CHECK:STDERR:        ·                     │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                     ╰────┤ type `Core.MaybeUnformed(X)` does not implement
  // CHECK:STDERR:        ·                          │ interface `Core.As(X)`
  // CHECK:STDERR:        ·                          │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var unused i: X = Init() as X;
  // TODO: The diagnostic should explain that the reason we can't perform this
  // conversion is due to the expression category.
  // CHECK:STDERR:        ╭─┤ fail_cannot_remove_unformed.carbon:[[@LINE+11]]:21
  // CHECK:STDERR: ->     │   var unused j: X = Init() unsafe as X;
  // CHECK:STDERR:        ·                     ┳━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·                     ┗━━┫ error: cannot convert expression of type
  // CHECK:STDERR:        ·                     │  ┃        `Core.MaybeUnformed(X)` to `X` with `unsafe as`
  // CHECK:STDERR:        ·                     │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                     ╰────┤ type `Core.MaybeUnformed(X)` does not implement
  // CHECK:STDERR:        ·                          │ interface `Core.UnsafeAs(X)`
  // CHECK:STDERR:        ·                          │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var unused j: X = Init() unsafe as X;
  // CHECK:STDERR:        ╭─┤ fail_cannot_remove_unformed.carbon:[[@LINE+11]]:21
  // CHECK:STDERR: ->     │   let unused v: X = value as X;
  // CHECK:STDERR:        ·                     ┳━━━━━━━━━
  // CHECK:STDERR:        ·                     ┗━━┫ error: cannot convert expression of type
  // CHECK:STDERR:        ·                     │  ┃        `Core.MaybeUnformed(X)` to `X` with `as`
  // CHECK:STDERR:        ·                     │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                     ╰────┤ type `Core.MaybeUnformed(X)` does not implement
  // CHECK:STDERR:        ·                          │ interface `Core.As(X)`
  // CHECK:STDERR:        ·                          │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let unused v: X = value as X;
  // CHECK:STDERR:        ╭─┤ fail_cannot_remove_unformed.carbon:[[@LINE+11]]:24
  // CHECK:STDERR: ->     │   let unused a: X* = &(reference as X);
  // CHECK:STDERR:        ·                        ┳━━━━━━━━━━━━━
  // CHECK:STDERR:        ·                        ┗━━┫ error: cannot convert expression of type
  // CHECK:STDERR:        ·                        │  ┃        `Core.MaybeUnformed(X)` to `X` with `as`
  // CHECK:STDERR:        ·                        │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                        ╰────┤ type `Core.MaybeUnformed(X)` does not implement
  // CHECK:STDERR:        ·                             │ interface `Core.As(X)`
  // CHECK:STDERR:        ·                             │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let unused a: X* = &(reference as X);
  // CHECK:STDERR:        ╭─┤ fail_cannot_remove_unformed.carbon:[[@LINE+11]]:22
  // CHECK:STDERR: ->     │   let unused b: X* = ptr as X*;
  // CHECK:STDERR:        ·                      ┳━━━━━━━━
  // CHECK:STDERR:        ·                      ┗━━┫ error: cannot convert expression of type
  // CHECK:STDERR:        ·                      │  ┃        `Core.MaybeUnformed(X)*` to `X*` with `as`
  // CHECK:STDERR:        ·                      │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                      ╰────┤ type `Core.MaybeUnformed(X)*` does not implement
  // CHECK:STDERR:        ·                           │ interface `Core.As(X*)`
  // CHECK:STDERR:        ·                           │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %ptr.db8: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %MaybeUnformed.type: type = parameterized_class_name_type @MaybeUnformed [concrete]
// CHECK:STDOUT:   %MaybeUnformed.generic: %MaybeUnformed.type = struct_value () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.67f: type = class_type @MaybeUnformed, @MaybeUnformed(%X) [concrete]
// CHECK:STDOUT:   %ptr.6d8: type = ptr_type %MaybeUnformed.67f [concrete]
// CHECK:STDOUT:   %pattern_type.d77: type = pattern_type %ptr.6d8 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d77 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %reference.var: ref %MaybeUnformed.67f = var_storage file.%reference.var_patt [concrete]
// CHECK:STDOUT:   %addr.d96: %ptr.6d8 = addr_of %reference.var [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.d77 = value_binding_pattern b [concrete]
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
// CHECK:STDOUT:   %reference.ref: ref %X = name_ref reference, file.%reference [concrete = file.%reference.var]
// CHECK:STDOUT:   %Core.ref.loc12_57: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %MaybeUnformed.ref.loc12_61: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:   %X.ref.loc12_76: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %MaybeUnformed.loc12_77: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.67f]
// CHECK:STDOUT:   %.loc12_54.1: ref %MaybeUnformed.67f = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc12_54.2: ref %MaybeUnformed.67f = converted %reference.ref, %.loc12_54.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr: %ptr.6d8 = addr_of %.loc12_54.2 [concrete = constants.%addr.d96]
// CHECK:STDOUT:   %.loc12_38: type = splice_block %ptr.loc12 [concrete = constants.%ptr.6d8] {
// CHECK:STDOUT:     %Core.ref.loc12_17: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %MaybeUnformed.ref.loc12_21: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:     %X.ref.loc12_36: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %MaybeUnformed.loc12_37: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.67f]
// CHECK:STDOUT:     %ptr.loc12: type = ptr_type %MaybeUnformed.loc12_37 [concrete = constants.%ptr.6d8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.6d8 = wrapper_binding a, %addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d77 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.db8 = name_ref ptr, file.%ptr.loc7_8
// CHECK:STDOUT:   %Core.ref.loc13_49: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %MaybeUnformed.ref.loc13_53: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:   %X.ref.loc13_68: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %MaybeUnformed.loc13_69: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.67f]
// CHECK:STDOUT:   %ptr.loc13_70: type = ptr_type %MaybeUnformed.loc13_69 [concrete = constants.%ptr.6d8]
// CHECK:STDOUT:   %.loc13_46.1: %ptr.6d8 = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc13_46.2: %ptr.6d8 = converted %ptr.ref, %.loc13_46.1
// CHECK:STDOUT:   %.loc13_38: type = splice_block %ptr.loc13_38 [concrete = constants.%ptr.6d8] {
// CHECK:STDOUT:     %Core.ref.loc13_17: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %MaybeUnformed.ref.loc13_21: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:     %X.ref.loc13_36: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %MaybeUnformed.loc13_37: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.67f]
// CHECK:STDOUT:     %ptr.loc13_38: type = ptr_type %MaybeUnformed.loc13_37 [concrete = constants.%ptr.6d8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.6d8 = wrapper_binding b, %.loc13_46.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.d77 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Init.type: type = fn_type @Init [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Init: %Init.type = struct_value () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.type: type = parameterized_class_name_type @MaybeUnformed [concrete]
// CHECK:STDOUT:   %MaybeUnformed.generic: %MaybeUnformed.type = struct_value () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.67f: type = class_type @MaybeUnformed, @MaybeUnformed(%X) [concrete]
// CHECK:STDOUT:   %.f01: type = maybe_unformed_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.55d: type = pattern_type %MaybeUnformed.67f [concrete]
// CHECK:STDOUT:   %i.patt: %pattern_type.55d = ref_binding_pattern i [concrete]
// CHECK:STDOUT:   %i.var_patt: %pattern_type.55d = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %v.patt: %pattern_type.55d = value_binding_pattern v [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc26_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.4df: %SubobjectDestroy.type = facet_value %MaybeUnformed.67f, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.8e6: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.4df) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %MaybeUnformed.67f, (%DestroyImpl.impl_witness.8e6, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .MaybeUnformed = %Core.MaybeUnformed
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.MaybeUnformed: %MaybeUnformed.type = import_ref Core//prelude/parts/maybe_unformed, MaybeUnformed, loaded [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:   %Core.As: %As.type.116 = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %i.var: ref %MaybeUnformed.67f = var_storage %i.var_patt
// CHECK:STDOUT:   %Init.ref: %Init.type = name_ref Init, file.%Init.decl [concrete = constants.%Init]
// CHECK:STDOUT:   %.loc26_46: ref %X = temporary_storage
// CHECK:STDOUT:   %Init.call: init %X to %.loc26_46 = call %Init.ref()
// CHECK:STDOUT:   %Core.ref.loc26_51: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %MaybeUnformed.ref.loc26_55: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:   %X.ref.loc26_70: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %MaybeUnformed.loc26_71: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.67f]
// CHECK:STDOUT:   %.loc26_48: %MaybeUnformed.67f = converted %Init.call, <error> [concrete = <error>]
// CHECK:STDOUT:   assign %i.var, <error>
// CHECK:STDOUT:   %.loc26_37: type = splice_block %MaybeUnformed.loc26_37 [concrete = constants.%MaybeUnformed.67f] {
// CHECK:STDOUT:     %Core.ref.loc26_17: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %MaybeUnformed.ref.loc26_21: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:     %X.ref.loc26_36: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %MaybeUnformed.loc26_37: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.67f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: ref %MaybeUnformed.67f = wrapper_binding i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.55d = ref_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.55d = var_pattern %i.patt [concrete = constants.%i.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %value.ref: %X = name_ref value, file.%value
// CHECK:STDOUT:   %Core.ref.loc41_50: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %MaybeUnformed.ref.loc41_54: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:   %X.ref.loc41_69: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %MaybeUnformed.loc41_70: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.67f]
// CHECK:STDOUT:   %.loc41_47: %MaybeUnformed.67f = converted %value.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc41_37: type = splice_block %MaybeUnformed.loc41_37 [concrete = constants.%MaybeUnformed.67f] {
// CHECK:STDOUT:     %Core.ref.loc41_17: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %MaybeUnformed.ref.loc41_21: %MaybeUnformed.type = name_ref MaybeUnformed, imports.%Core.MaybeUnformed [concrete = constants.%MaybeUnformed.generic]
// CHECK:STDOUT:     %X.ref.loc41_36: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %MaybeUnformed.loc41_37: type = class_type @MaybeUnformed, @MaybeUnformed(constants.%X) [concrete = constants.%MaybeUnformed.67f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: %MaybeUnformed.67f = wrapper_binding v, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.55d = value_binding_pattern v [concrete = constants.%v.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc26_3.1: %Destroy.type = facet_value constants.%MaybeUnformed.67f, (constants.%DestroyImpl.impl_witness.8e6, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc26_3.1: %Destroy.type = converted constants.%MaybeUnformed.67f, %Destroy.facet.loc26_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc26_3.2: %Destroy.type = facet_value constants.%MaybeUnformed.67f, (constants.%DestroyImpl.impl_witness.8e6, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc26_3.2: %Destroy.type = converted constants.%MaybeUnformed.67f, %Destroy.facet.loc26_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%i.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_3.2(%self.param: ref %X) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_3.3(%self.param: ref %.f01) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_3.4(%self.param: ref %MaybeUnformed.67f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unsafe_remove_unformed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %MaybeUnformed.67f: type = class_type @MaybeUnformed, @MaybeUnformed(%X) [concrete]
// CHECK:STDOUT:   %ptr.6d8: type = ptr_type %MaybeUnformed.67f [concrete]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %v.patt: %pattern_type.6cb = value_binding_pattern v [concrete]
// CHECK:STDOUT:   %ptr.db8: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.990: type = pattern_type %ptr.db8 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.990 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %reference.var: ref %X = var_storage file.%reference.var_patt [concrete]
// CHECK:STDOUT:   %addr.4ff: %ptr.db8 = addr_of %reference.var [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.990 = value_binding_pattern b [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %value.ref: %MaybeUnformed.67f = name_ref value, file.%value
// CHECK:STDOUT:   %X.ref.loc13_37: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc13_34.1: %X = as_compatible %value.ref
// CHECK:STDOUT:   %.loc13_34.2: %X = converted %value.ref, %.loc13_34.1
// CHECK:STDOUT:   %X.ref.loc13_17: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %v: %X = wrapper_binding v, %.loc13_34.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.6cb = value_binding_pattern v [concrete = constants.%v.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %reference.ref: ref %MaybeUnformed.67f = name_ref reference, file.%reference [concrete = file.%reference.var]
// CHECK:STDOUT:   %X.ref.loc14_44: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc14_41.1: ref %X = as_compatible %reference.ref [concrete = constants.%reference.var]
// CHECK:STDOUT:   %.loc14_41.2: ref %X = converted %reference.ref, %.loc14_41.1 [concrete = constants.%reference.var]
// CHECK:STDOUT:   %addr: %ptr.db8 = addr_of %.loc14_41.2 [concrete = constants.%addr.4ff]
// CHECK:STDOUT:   %.loc14_18: type = splice_block %ptr.loc14 [concrete = constants.%ptr.db8] {
// CHECK:STDOUT:     %X.ref.loc14_17: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc14: type = ptr_type %X.ref.loc14_17 [concrete = constants.%ptr.db8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %ptr.db8 = wrapper_binding a, %addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.990 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr.ref: %ptr.6d8 = name_ref ptr, file.%ptr.loc9_8
// CHECK:STDOUT:   %X.ref.loc15_36: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %ptr.loc15_37: type = ptr_type %X.ref.loc15_36 [concrete = constants.%ptr.db8]
// CHECK:STDOUT:   %.loc15_33.1: %ptr.db8 = as_compatible %ptr.ref
// CHECK:STDOUT:   %.loc15_33.2: %ptr.db8 = converted %ptr.ref, %.loc15_33.1
// CHECK:STDOUT:   %.loc15_18: type = splice_block %ptr.loc15_18 [concrete = constants.%ptr.db8] {
// CHECK:STDOUT:     %X.ref.loc15_17: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc15_18: type = ptr_type %X.ref.loc15_17 [concrete = constants.%ptr.db8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.db8 = wrapper_binding b, %.loc15_33.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.990 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
