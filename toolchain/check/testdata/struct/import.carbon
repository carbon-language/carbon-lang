// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/import.carbon

// --- implicit.carbon

package Implicit;

var a_ref: {.a: i32} = {.a = 0};
var b_ref: {.a: {.b: i32, .c: (i32,)}, .d: i32} =
    {.a = {.b = 0, .c = (0,)}, .d = 0};

class C(S:! {.a: i32, .b: i32}) {}
fn F() -> C({.a = 1, .b = 2});

// --- implicit.impl.carbon

impl package Implicit;

var a: {.a: i32} = a_ref;
var b: {.a: {.b: i32, .c: (i32,)}, .d: i32} = b_ref;
var c: C({.a = 1, .b = 2}) = F();

// --- fail_bad_type.impl.carbon

impl package Implicit;
// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE+8]]:14: error: missing value for field `a` in struct initialization [StructInitMissingFieldInLiteral]
// CHECK:STDERR: var c_bad: C({.c = 1, .d = 2}) = F();
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_type.impl.carbon:[[@LINE-4]]:1: in import [InImport]
// CHECK:STDERR: implicit.carbon:8:9: note: initializing generic parameter `S` declared here [InitializingGenericParam]
// CHECK:STDERR: class C(S:! {.a: i32, .b: i32}) {}
// CHECK:STDERR:         ^
// CHECK:STDERR:
var c_bad: C({.c = 1, .d = 2}) = F();

// --- fail_bad_value.impl.carbon

impl package Implicit;
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+7]]:1: error: cannot implicitly convert expression of type `C({.a = 1, .b = 2})` to `C({.a = 3, .b = 4})` [ConversionFailure]
// CHECK:STDERR: var c_bad: C({.a = 3, .b = 4}) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_value.impl.carbon:[[@LINE+4]]:1: note: type `C({.a = 1, .b = 2})` does not implement interface `Core.ImplicitAs(C({.a = 3, .b = 4}))` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var c_bad: C({.a = 3, .b = 4}) = F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var c_bad: C({.a = 3, .b = 4}) = F();

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.ba9: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.268: type = pattern_type %struct_type.a.ba9 [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %struct_type.a.a6c: type = struct_type {.a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.956: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d04: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.956, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %struct.92d: %struct_type.a.ba9 = struct_value (%int_0.6a9) [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %struct_type.b.c.929: type = struct_type {.b: %i32, .c: %tuple.type.a1c} [concrete]
// CHECK:STDOUT:   %struct_type.a.d.3d9: type = struct_type {.a: %struct_type.b.c.929, .d: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.665: type = pattern_type %struct_type.a.d.3d9 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %struct_type.b.c.9af: type = struct_type {.b: Core.IntLiteral, .c: %tuple.type.985} [concrete]
// CHECK:STDOUT:   %struct_type.a.d.b82: type = struct_type {.a: %struct_type.b.c.9af, .d: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %.042: ref %struct_type.b.c.929 = struct_access file.%b_ref.var, element0 [concrete]
// CHECK:STDOUT:   %.643: ref %i32 = struct_access %.042, element0 [concrete]
// CHECK:STDOUT:   %.66d: ref %tuple.type.a1c = struct_access %.042, element1 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.a1c = tuple_value (%int_0.6a9) [concrete]
// CHECK:STDOUT:   %struct.381: %struct_type.b.c.929 = struct_value (%int_0.6a9, %tuple) [concrete]
// CHECK:STDOUT:   %.c6b: ref %i32 = struct_access file.%b_ref.var, element1 [concrete]
// CHECK:STDOUT:   %struct.20d: %struct_type.a.d.3d9 = struct_value (%struct.381, %int_0.6a9) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.501: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %S: %struct_type.a.b.501 = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.851: type = pattern_type %struct_type.a.b.501 [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.775: type = class_type @C, @C(%S) [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.d9c: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%S) [symbolic]
// CHECK:STDOUT:   %ptr.e4c: type = ptr_type %C.775 [symbolic]
// CHECK:STDOUT:   %pattern_type.d5f: type = pattern_type %ptr.e4c [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%S) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %struct.ed5: %struct_type.a.b.501 = struct_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %C.c8f: type = class_type @C, @C(%struct.ed5) [concrete]
// CHECK:STDOUT:   %pattern_type.4f9: type = pattern_type %C.c8f [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:     .b_ref = %b_ref
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_ref.patt: %pattern_type.268 = binding_pattern a_ref [concrete]
// CHECK:STDOUT:     %a_ref.var_patt: %pattern_type.268 = var_pattern %a_ref.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ref.var: ref %struct_type.a.ba9 = var %a_ref.var_patt [concrete]
// CHECK:STDOUT:   %.loc4: type = splice_block %struct_type.a [concrete = constants.%struct_type.a.ba9] {
// CHECK:STDOUT:     %int_32.loc4: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a.ba9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ref: ref %struct_type.a.ba9 = bind_name a_ref, %a_ref.var [concrete = %a_ref.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_ref.patt: %pattern_type.665 = binding_pattern b_ref [concrete]
// CHECK:STDOUT:     %b_ref.var_patt: %pattern_type.665 = var_pattern %b_ref.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_ref.var: ref %struct_type.a.d.3d9 = var %b_ref.var_patt [concrete]
// CHECK:STDOUT:   %.loc5_47: type = splice_block %struct_type.a.d [concrete = constants.%struct_type.a.d.3d9] {
// CHECK:STDOUT:     %int_32.loc5_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_36.1: %tuple.type.85c = tuple_literal (%i32.loc5_32)
// CHECK:STDOUT:     %.loc5_36.2: type = converted %.loc5_36.1, constants.%tuple.type.a1c [concrete = constants.%tuple.type.a1c]
// CHECK:STDOUT:     %struct_type.b.c: type = struct_type {.b: %i32, .c: %tuple.type.a1c} [concrete = constants.%struct_type.b.c.929]
// CHECK:STDOUT:     %int_32.loc5_44: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_44: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.d: type = struct_type {.a: %struct_type.b.c.929, .d: %i32} [concrete = constants.%struct_type.a.d.3d9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_ref: ref %struct_type.a.d.3d9 = bind_name b_ref, %b_ref.var [concrete = %b_ref.var]
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %S.patt: %pattern_type.851 = symbolic_binding_pattern S, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b.501] {
// CHECK:STDOUT:       %int_32.loc8_18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc8_18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_32.loc8_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc8_27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.501]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %S.loc8_9.2: %struct_type.a.b.501 = bind_symbolic_name S, 0 [symbolic = %S.loc8_9.1 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.4f9 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.4f9 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc9_28.1: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:     %impl.elem0.loc9_28.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:     %bound_method.loc9_28.1: <bound method> = bound_method %int_1, %impl.elem0.loc9_28.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ab5]
// CHECK:STDOUT:     %specific_fn.loc9_28.1: <specific function> = specific_function %impl.elem0.loc9_28.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_28.2: <bound method> = bound_method %int_1, %specific_fn.loc9_28.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_28.1: init %i32 = call %bound_method.loc9_28.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc9_28.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_28.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc9_28.3: %i32 = converted %int_1, %.loc9_28.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %impl.elem0.loc9_28.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:     %bound_method.loc9_28.3: <bound method> = bound_method %int_2, %impl.elem0.loc9_28.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ef9]
// CHECK:STDOUT:     %specific_fn.loc9_28.2: <specific function> = specific_function %impl.elem0.loc9_28.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_28.4: <bound method> = bound_method %int_2, %specific_fn.loc9_28.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_28.2: init %i32 = call %bound_method.loc9_28.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc9_28.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_28.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc9_28.5: %i32 = converted %int_2, %.loc9_28.4 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %struct: %struct_type.a.b.501 = struct_value (%.loc9_28.3, %.loc9_28.5) [concrete = constants.%struct.ed5]
// CHECK:STDOUT:     %.loc9_29: %struct_type.a.b.501 = converted %.loc9_28.1, %struct [concrete = constants.%struct.ed5]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%struct.ed5) [concrete = constants.%C.c8f]
// CHECK:STDOUT:     %return.param: ref %C.c8f = out_param call_param0
// CHECK:STDOUT:     %return: ref %C.c8f = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Destroy.impl(@C.%S.loc8_9.2: %struct_type.a.b.501) {
// CHECK:STDOUT:   %S: %struct_type.a.b.501 = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%S) [symbolic = %C (constants.%C.775)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%S) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.d9c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%S) [symbolic = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = struct_value () [symbolic = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %C.as.Destroy.impl.Op.decl: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = fn_decl @C.as.Destroy.impl.Op [symbolic = @C.as.Destroy.impl.%C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.d5f) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.d5f) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc8_33.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.e4c) = value_param call_param0
// CHECK:STDOUT:       %.loc8_33.2: type = splice_block %Self.ref [symbolic = %C (constants.%C.775)] {
// CHECK:STDOUT:         %.loc8_33.3: type = specific_constant constants.%C.775, @C(constants.%S) [symbolic = %C (constants.%C.775)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc8_33.3 [symbolic = %C (constants.%C.775)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Destroy.impl.Op.%ptr (%ptr.e4c) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @C.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%S.loc8_9.2: %struct_type.a.b.501) {
// CHECK:STDOUT:   %S.loc8_9.1: %struct_type.a.b.501 = bind_symbolic_name S, 0 [symbolic = %S.loc8_9.1 (constants.%S)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C.775 [symbolic = @C.as.Destroy.impl.%C (constants.%C.775)]
// CHECK:STDOUT:     impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @C.as.Destroy.impl(constants.%S) [symbolic = @C.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.d9c)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.775
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Destroy.impl.Op(@C.%S.loc8_9.2: %struct_type.a.b.501) {
// CHECK:STDOUT:   %S: %struct_type.a.b.501 = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%S) [symbolic = %C (constants.%C.775)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [symbolic = %ptr (constants.%ptr.e4c)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.d5f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.e4c)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C.c8f;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc4: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc4_31.1: %struct_type.a.a6c = struct_literal (%int_0.loc4)
// CHECK:STDOUT:   %impl.elem0.loc4: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc4_31.1: <bound method> = bound_method %int_0.loc4, %impl.elem0.loc4 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc4: <specific function> = specific_function %impl.elem0.loc4, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_31.2: <bound method> = bound_method %int_0.loc4, %specific_fn.loc4 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4: init %i32 = call %bound_method.loc4_31.2(%int_0.loc4) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc4_31.2: init %i32 = converted %int_0.loc4, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc4_31.3: init %struct_type.a.ba9 = struct_init (%.loc4_31.2) to file.%a_ref.var [concrete = constants.%struct.92d]
// CHECK:STDOUT:   %.loc4_1: init %struct_type.a.ba9 = converted %.loc4_31.1, %.loc4_31.3 [concrete = constants.%struct.92d]
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc4_1
// CHECK:STDOUT:   %int_0.loc6_17: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_0.loc6_26: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc6_28.1: %tuple.type.985 = tuple_literal (%int_0.loc6_26)
// CHECK:STDOUT:   %.loc6_29.1: %struct_type.b.c.9af = struct_literal (%int_0.loc6_17, %.loc6_28.1)
// CHECK:STDOUT:   %int_0.loc6_37: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc6_38.1: %struct_type.a.d.b82 = struct_literal (%.loc6_29.1, %int_0.loc6_37)
// CHECK:STDOUT:   %impl.elem0.loc6_29: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc6_29.1: <bound method> = bound_method %int_0.loc6_17, %impl.elem0.loc6_29 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc6_29: <specific function> = specific_function %impl.elem0.loc6_29, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_29.2: <bound method> = bound_method %int_0.loc6_17, %specific_fn.loc6_29 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_29: init %i32 = call %bound_method.loc6_29.2(%int_0.loc6_17) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_29.2: init %i32 = converted %int_0.loc6_17, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_29 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_38.2: ref %struct_type.b.c.929 = struct_access file.%b_ref.var, element0 [concrete = constants.%.042]
// CHECK:STDOUT:   %.loc6_29.3: ref %i32 = struct_access %.loc6_38.2, element0 [concrete = constants.%.643]
// CHECK:STDOUT:   %.loc6_29.4: init %i32 = initialize_from %.loc6_29.2 to %.loc6_29.3 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %impl.elem0.loc6_28: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc6_28.1: <bound method> = bound_method %int_0.loc6_26, %impl.elem0.loc6_28 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc6_28: <specific function> = specific_function %impl.elem0.loc6_28, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_28.2: <bound method> = bound_method %int_0.loc6_26, %specific_fn.loc6_28 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_28: init %i32 = call %bound_method.loc6_28.2(%int_0.loc6_26) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_28.2: init %i32 = converted %int_0.loc6_26, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_28 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_29.5: ref %tuple.type.a1c = struct_access %.loc6_38.2, element1 [concrete = constants.%.66d]
// CHECK:STDOUT:   %.loc6_28.3: init %tuple.type.a1c = tuple_init (%.loc6_28.2) to %.loc6_29.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.6: init %tuple.type.a1c = converted %.loc6_28.1, %.loc6_28.3 [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.7: init %tuple.type.a1c = initialize_from %.loc6_29.6 to %.loc6_29.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.8: init %struct_type.b.c.929 = struct_init (%.loc6_29.4, %.loc6_29.7) to %.loc6_38.2 [concrete = constants.%struct.381]
// CHECK:STDOUT:   %.loc6_38.3: init %struct_type.b.c.929 = converted %.loc6_29.1, %.loc6_29.8 [concrete = constants.%struct.381]
// CHECK:STDOUT:   %impl.elem0.loc6_38: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc6_38.1: <bound method> = bound_method %int_0.loc6_37, %impl.elem0.loc6_38 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc6_38: <specific function> = specific_function %impl.elem0.loc6_38, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_38.2: <bound method> = bound_method %int_0.loc6_37, %specific_fn.loc6_38 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_38: init %i32 = call %bound_method.loc6_38.2(%int_0.loc6_37) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_38.4: init %i32 = converted %int_0.loc6_37, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_38 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_38.5: ref %i32 = struct_access file.%b_ref.var, element1 [concrete = constants.%.c6b]
// CHECK:STDOUT:   %.loc6_38.6: init %i32 = initialize_from %.loc6_38.4 to %.loc6_38.5 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_38.7: init %struct_type.a.d.3d9 = struct_init (%.loc6_38.3, %.loc6_38.6) to file.%b_ref.var [concrete = constants.%struct.20d]
// CHECK:STDOUT:   %.loc5: init %struct_type.a.d.3d9 = converted %.loc6_38.1, %.loc6_38.7 [concrete = constants.%struct.20d]
// CHECK:STDOUT:   assign file.%b_ref.var, %.loc5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S.loc8_9.1 => constants.%S
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT:   %C => constants.%C.775
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.d9c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT:   %C => constants.%C.775
// CHECK:STDOUT:   %ptr => constants.%ptr.e4c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d5f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct.ed5) {
// CHECK:STDOUT:   %S.loc8_9.1 => constants.%struct.ed5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.595: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.595 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9a6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.458: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.5ca: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %S: %struct_type.a.b.5ca = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C.720: type = class_type @C, @C(%S) [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.abc: <witness> = impl_witness imports.%Destroy.impl_witness_table.2ab, @C.as.Destroy.impl(%S) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%S) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.9e0: type = ptr_type %C.720 [symbolic]
// CHECK:STDOUT:   %pattern_type.d77: type = pattern_type %ptr.9e0 [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.b54: type = pattern_type %struct_type.a [concrete]
// CHECK:STDOUT:   %.c27: ref %i32 = struct_access imports.%a_ref.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.type.dd4: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %struct_type.b.c: type = struct_type {.b: %i32, .c: %tuple.type.dd4} [concrete]
// CHECK:STDOUT:   %struct_type.a.d.9db: type = struct_type {.a: %struct_type.b.c, .d: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.020: type = pattern_type %struct_type.a.d.9db [concrete]
// CHECK:STDOUT:   %.769: ref %struct_type.b.c = struct_access imports.%b_ref.var, element0 [concrete]
// CHECK:STDOUT:   %.92c: ref %struct_type.b.c = struct_access file.%b.var, element0 [concrete]
// CHECK:STDOUT:   %.d8a: ref %i32 = struct_access %.769, element0 [concrete]
// CHECK:STDOUT:   %.9dd: ref %i32 = struct_access %.92c, element0 [concrete]
// CHECK:STDOUT:   %.e5f: ref %tuple.type.dd4 = struct_access %.769, element1 [concrete]
// CHECK:STDOUT:   %.bbf: ref %tuple.type.dd4 = struct_access %.92c, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem0.aa5: ref %i32 = tuple_access %.e5f, element0 [concrete]
// CHECK:STDOUT:   %.dfe: ref %i32 = struct_access imports.%b_ref.var, element1 [concrete]
// CHECK:STDOUT:   %.607: ref %i32 = struct_access file.%b.var, element1 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b9e: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.ea0: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1ad, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e14: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.4cb: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e14 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.b9e = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.940: type = fn_type_with_self_type %ImplicitAs.Convert.type.ea0, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dc5: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.4cb [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.4cb, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.c37: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.47b: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.30f: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.4cb [concrete]
// CHECK:STDOUT:   %bound_method.62b: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.d0d: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.5ca = struct_value (%int_1.47b, %int_2.d0d) [concrete]
// CHECK:STDOUT:   %C.a8a: type = class_type @C, @C(%struct) [concrete]
// CHECK:STDOUT:   %pattern_type.a5a: type = pattern_type %C.a8a [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Implicit.a_ref: ref %struct_type.a = import_ref Implicit//default, a_ref, loaded [concrete = %a_ref.var]
// CHECK:STDOUT:   %Implicit.b_ref: ref %struct_type.a.d.9db = import_ref Implicit//default, b_ref, loaded [concrete = %b_ref.var]
// CHECK:STDOUT:   %Implicit.C: %C.type = import_ref Implicit//default, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Implicit.F: %F.type = import_ref Implicit//default, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import_ref.773: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9a6) = import_ref Implicit//default, inst151 [indirect], loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.458)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1ad = impl_witness_table (%Implicit.import_ref.773), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Implicit.import_ref.48d = import_ref Implicit//default, loc8_33, unloaded
// CHECK:STDOUT:   %Implicit.import_ref.c81f8f.1: %struct_type.a.b.5ca = import_ref Implicit//default, loc8_9, loaded [symbolic = @C.%S (constants.%S)]
// CHECK:STDOUT:   %Implicit.import_ref.8f2: <witness> = import_ref Implicit//default, loc8_34, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Implicit.import_ref.b8b = import_ref Implicit//default, inst490 [no loc], unloaded
// CHECK:STDOUT:   %Implicit.import_ref.c81f8f.2: %struct_type.a.b.5ca = import_ref Implicit//default, loc8_9, loaded [symbolic = @C.%S (constants.%S)]
// CHECK:STDOUT:   %Implicit.import_ref.ba3: type = import_ref Implicit//default, loc8_33, loaded [symbolic = @C.as.Destroy.impl.%C (constants.%C.720)]
// CHECK:STDOUT:   %Implicit.import_ref.cb9: type = import_ref Implicit//default, inst109 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Implicit.import_ref.7f2ca0.1 = import_ref Implicit//default, loc8_33, unloaded
// CHECK:STDOUT:   %Implicit.import_ref.7f2ca0.2 = import_ref Implicit//default, loc8_33, unloaded
// CHECK:STDOUT:   %Destroy.impl_witness_table.2ab = impl_witness_table (%Implicit.import_ref.7f2ca0.2), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Implicit.import_ref.c81f8f.3: %struct_type.a.b.5ca = import_ref Implicit//default, loc8_9, loaded [symbolic = @C.%S (constants.%S)]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %a_ref.patt: %pattern_type.b54 = binding_pattern a_ref [concrete]
// CHECK:STDOUT:   %a_ref.var_patt: %pattern_type.b54 = var_pattern %a_ref.patt [concrete]
// CHECK:STDOUT:   %a_ref.var: ref %struct_type.a = var %a_ref.var_patt [concrete]
// CHECK:STDOUT:   %b_ref.patt: %pattern_type.020 = binding_pattern b_ref [concrete]
// CHECK:STDOUT:   %b_ref.var_patt: %pattern_type.020 = var_pattern %b_ref.patt [concrete]
// CHECK:STDOUT:   %b_ref.var: ref %struct_type.a.d.9db = var %b_ref.var_patt [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.595 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .a_ref = imports.%Implicit.a_ref
// CHECK:STDOUT:     .b_ref = imports.%Implicit.b_ref
// CHECK:STDOUT:     .C = imports.%Implicit.C
// CHECK:STDOUT:     .F = imports.%Implicit.F
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b54 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b54 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %struct_type.a = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc4: type = splice_block %struct_type.a [concrete = constants.%struct_type.a] {
// CHECK:STDOUT:     %int_32.loc4: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %struct_type.a = bind_name a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.020 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.020 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %struct_type.a.d.9db = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc5_43: type = splice_block %struct_type.a.d [concrete = constants.%struct_type.a.d.9db] {
// CHECK:STDOUT:     %int_32.loc5_18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_28: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_28: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_32.1: %tuple.type.85c = tuple_literal (%i32.loc5_28)
// CHECK:STDOUT:     %.loc5_32.2: type = converted %.loc5_32.1, constants.%tuple.type.dd4 [concrete = constants.%tuple.type.dd4]
// CHECK:STDOUT:     %struct_type.b.c: type = struct_type {.b: %i32, .c: %tuple.type.dd4} [concrete = constants.%struct_type.b.c]
// CHECK:STDOUT:     %int_32.loc5_40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_40: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.d: type = struct_type {.a: %struct_type.b.c, .d: %i32} [concrete = constants.%struct_type.a.d.9db]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %struct_type.a.d.9db = bind_name b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.a5a = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.a5a = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.a8a = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc6_26.1: type = splice_block %C [concrete = constants.%C.a8a] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Implicit.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc6_25.1: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:     %impl.elem0.loc6_25.1: %.940 = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.4cb]
// CHECK:STDOUT:     %bound_method.loc6_25.1: <bound method> = bound_method %int_1, %impl.elem0.loc6_25.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dc5]
// CHECK:STDOUT:     %specific_fn.loc6_25.1: <specific function> = specific_function %impl.elem0.loc6_25.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_25.2: <bound method> = bound_method %int_1, %specific_fn.loc6_25.1 [concrete = constants.%bound_method.c37]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_25.1: init %i32 = call %bound_method.loc6_25.2(%int_1) [concrete = constants.%int_1.47b]
// CHECK:STDOUT:     %.loc6_25.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_25.1 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:     %.loc6_25.3: %i32 = converted %int_1, %.loc6_25.2 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:     %impl.elem0.loc6_25.2: %.940 = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.4cb]
// CHECK:STDOUT:     %bound_method.loc6_25.3: <bound method> = bound_method %int_2, %impl.elem0.loc6_25.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.30f]
// CHECK:STDOUT:     %specific_fn.loc6_25.2: <specific function> = specific_function %impl.elem0.loc6_25.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_25.4: <bound method> = bound_method %int_2, %specific_fn.loc6_25.2 [concrete = constants.%bound_method.62b]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_25.2: init %i32 = call %bound_method.loc6_25.4(%int_2) [concrete = constants.%int_2.d0d]
// CHECK:STDOUT:     %.loc6_25.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_25.2 [concrete = constants.%int_2.d0d]
// CHECK:STDOUT:     %.loc6_25.5: %i32 = converted %int_2, %.loc6_25.4 [concrete = constants.%int_2.d0d]
// CHECK:STDOUT:     %struct: %struct_type.a.b.5ca = struct_value (%.loc6_25.3, %.loc6_25.5) [concrete = constants.%struct]
// CHECK:STDOUT:     %.loc6_26.2: %struct_type.a.b.5ca = converted %.loc6_25.1, %struct [concrete = constants.%struct]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%struct) [concrete = constants.%C.a8a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.a8a = bind_name c, %c.var [concrete = %c.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Destroy.impl(imports.%Implicit.import_ref.c81f8f.2: %struct_type.a.b.5ca) [from "implicit.carbon"] {
// CHECK:STDOUT:   %S: %struct_type.a.b.5ca = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%S) [symbolic = %C (constants.%C.720)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table.2ab, @C.as.Destroy.impl(%S) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.abc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%S) [symbolic = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = struct_value () [symbolic = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Implicit.import_ref.ba3 as imports.%Implicit.import_ref.cb9 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = imports.%Implicit.import_ref.7f2ca0.1
// CHECK:STDOUT:     witness = imports.%Implicit.import_ref.48d
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Implicit.import_ref.c81f8f.1: %struct_type.a.b.5ca) [from "implicit.carbon"] {
// CHECK:STDOUT:   %S: %struct_type.a.b.5ca = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Implicit.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Implicit.import_ref.b8b
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Destroy.impl.Op(imports.%Implicit.import_ref.c81f8f.3: %struct_type.a.b.5ca) [from "implicit.carbon"] {
// CHECK:STDOUT:   %S: %struct_type.a.b.5ca = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%S) [symbolic = %C (constants.%C.720)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [symbolic = %ptr (constants.%ptr.9e0)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.d77)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %struct_type.a = name_ref a_ref, imports.%Implicit.a_ref [concrete = imports.%a_ref.var]
// CHECK:STDOUT:   %.loc4_20.1: ref %i32 = struct_access %a_ref.ref, element0 [concrete = constants.%.c27]
// CHECK:STDOUT:   %.loc4_20.2: %i32 = bind_value %.loc4_20.1
// CHECK:STDOUT:   %.loc4_20.3: init %struct_type.a = struct_init (%.loc4_20.2) to file.%a.var
// CHECK:STDOUT:   %.loc4_1: init %struct_type.a = converted %a_ref.ref, %.loc4_20.3
// CHECK:STDOUT:   assign file.%a.var, %.loc4_1
// CHECK:STDOUT:   %b_ref.ref: ref %struct_type.a.d.9db = name_ref b_ref, imports.%Implicit.b_ref [concrete = imports.%b_ref.var]
// CHECK:STDOUT:   %.loc5_47.1: ref %struct_type.b.c = struct_access %b_ref.ref, element0 [concrete = constants.%.769]
// CHECK:STDOUT:   %.loc5_47.2: ref %i32 = struct_access %.loc5_47.1, element0 [concrete = constants.%.d8a]
// CHECK:STDOUT:   %.loc5_47.3: %i32 = bind_value %.loc5_47.2
// CHECK:STDOUT:   %.loc5_47.4: ref %struct_type.b.c = struct_access file.%b.var, element0 [concrete = constants.%.92c]
// CHECK:STDOUT:   %.loc5_47.5: ref %i32 = struct_access %.loc5_47.4, element0 [concrete = constants.%.9dd]
// CHECK:STDOUT:   %.loc5_47.6: init %i32 = initialize_from %.loc5_47.3 to %.loc5_47.5
// CHECK:STDOUT:   %.loc5_47.7: ref %tuple.type.dd4 = struct_access %.loc5_47.1, element1 [concrete = constants.%.e5f]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %.loc5_47.7, element0 [concrete = constants.%tuple.elem0.aa5]
// CHECK:STDOUT:   %.loc5_47.8: %i32 = bind_value %tuple.elem0
// CHECK:STDOUT:   %.loc5_47.9: ref %tuple.type.dd4 = struct_access %.loc5_47.4, element1 [concrete = constants.%.bbf]
// CHECK:STDOUT:   %.loc5_47.10: init %tuple.type.dd4 = tuple_init (%.loc5_47.8) to %.loc5_47.9
// CHECK:STDOUT:   %.loc5_47.11: init %tuple.type.dd4 = converted %.loc5_47.7, %.loc5_47.10
// CHECK:STDOUT:   %.loc5_47.12: init %tuple.type.dd4 = initialize_from %.loc5_47.11 to %.loc5_47.9
// CHECK:STDOUT:   %.loc5_47.13: init %struct_type.b.c = struct_init (%.loc5_47.6, %.loc5_47.12) to %.loc5_47.4
// CHECK:STDOUT:   %.loc5_47.14: init %struct_type.b.c = converted %.loc5_47.1, %.loc5_47.13
// CHECK:STDOUT:   %.loc5_47.15: ref %i32 = struct_access %b_ref.ref, element1 [concrete = constants.%.dfe]
// CHECK:STDOUT:   %.loc5_47.16: %i32 = bind_value %.loc5_47.15
// CHECK:STDOUT:   %.loc5_47.17: ref %i32 = struct_access file.%b.var, element1 [concrete = constants.%.607]
// CHECK:STDOUT:   %.loc5_47.18: init %i32 = initialize_from %.loc5_47.16 to %.loc5_47.17
// CHECK:STDOUT:   %.loc5_47.19: init %struct_type.a.d.9db = struct_init (%.loc5_47.14, %.loc5_47.18) to file.%b.var
// CHECK:STDOUT:   %.loc5_1: init %struct_type.a.d.9db = converted %b_ref.ref, %.loc5_47.19
// CHECK:STDOUT:   assign file.%b.var, %.loc5_1
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Implicit.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc6: ref %C.a8a = splice_block file.%c.var [concrete = file.%c.var] {}
// CHECK:STDOUT:   %F.call: init %C.a8a = call %F.ref() to %.loc6
// CHECK:STDOUT:   assign file.%c.var, %F.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT:   %C => constants.%C.720
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.abc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT:   %C => constants.%C.720
// CHECK:STDOUT:   %ptr => constants.%ptr.9e0
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct) {
// CHECK:STDOUT:   %S => constants.%struct
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_type.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %S: %struct_type.a.b = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C.720: type = class_type @C, @C(%S) [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.abc: <witness> = impl_witness imports.%Destroy.impl_witness_table.2ab, @C.as.Destroy.impl(%S) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%S) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.9e0: type = ptr_type %C.720 [symbolic]
// CHECK:STDOUT:   %pattern_type.d77: type = pattern_type %ptr.9e0 [symbolic]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.c.d: type = struct_type {.c: Core.IntLiteral, .d: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.d0d: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_1.47b: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b = struct_value (%int_1.47b, %int_2.d0d) [concrete]
// CHECK:STDOUT:   %C.a8a: type = class_type @C, @C(%struct) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Implicit.a_ref = import_ref Implicit//default, a_ref, unloaded
// CHECK:STDOUT:   %Implicit.b_ref = import_ref Implicit//default, b_ref, unloaded
// CHECK:STDOUT:   %Implicit.C: %C.type = import_ref Implicit//default, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Implicit.F: %F.type = import_ref Implicit//default, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import_ref.48d = import_ref Implicit//default, loc8_33, unloaded
// CHECK:STDOUT:   %Implicit.import_ref.c81f8f.1: %struct_type.a.b = import_ref Implicit//default, loc8_9, loaded [symbolic = @C.%S (constants.%S)]
// CHECK:STDOUT:   %Implicit.import_ref.8f2: <witness> = import_ref Implicit//default, loc8_34, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Implicit.import_ref.b8b = import_ref Implicit//default, inst490 [no loc], unloaded
// CHECK:STDOUT:   %Implicit.import_ref.c81f8f.2: %struct_type.a.b = import_ref Implicit//default, loc8_9, loaded [symbolic = @C.%S (constants.%S)]
// CHECK:STDOUT:   %Implicit.import_ref.ba3: type = import_ref Implicit//default, loc8_33, loaded [symbolic = @C.as.Destroy.impl.%C (constants.%C.720)]
// CHECK:STDOUT:   %Implicit.import_ref.cb9: type = import_ref Implicit//default, inst109 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Implicit.import_ref.7f2ca0.1 = import_ref Implicit//default, loc8_33, unloaded
// CHECK:STDOUT:   %Implicit.import_ref.7f2ca0.2 = import_ref Implicit//default, loc8_33, unloaded
// CHECK:STDOUT:   %Destroy.impl_witness_table.2ab = impl_witness_table (%Implicit.import_ref.7f2ca0.2), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Implicit.import_ref.c81f8f.3: %struct_type.a.b = import_ref Implicit//default, loc8_9, loaded [symbolic = @C.%S (constants.%S)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .a_ref = imports.%Implicit.a_ref
// CHECK:STDOUT:     .b_ref = imports.%Implicit.b_ref
// CHECK:STDOUT:     .C = imports.%Implicit.C
// CHECK:STDOUT:     .F = imports.%Implicit.F
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .c_bad = %c_bad
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c_bad.patt: <error> = binding_pattern c_bad [concrete]
// CHECK:STDOUT:     %c_bad.var_patt: <error> = var_pattern %c_bad.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_bad.var: ref <error> = var %c_bad.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Implicit.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc11: %struct_type.c.d = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_bad: <error> = bind_name c_bad, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Destroy.impl(imports.%Implicit.import_ref.c81f8f.2: %struct_type.a.b) [from "implicit.carbon"] {
// CHECK:STDOUT:   %S: %struct_type.a.b = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%S) [symbolic = %C (constants.%C.720)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table.2ab, @C.as.Destroy.impl(%S) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.abc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%S) [symbolic = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = struct_value () [symbolic = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Implicit.import_ref.ba3 as imports.%Implicit.import_ref.cb9 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = imports.%Implicit.import_ref.7f2ca0.1
// CHECK:STDOUT:     witness = imports.%Implicit.import_ref.48d
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Implicit.import_ref.c81f8f.1: %struct_type.a.b) [from "implicit.carbon"] {
// CHECK:STDOUT:   %S: %struct_type.a.b = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Implicit.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Implicit.import_ref.b8b
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Destroy.impl.Op(imports.%Implicit.import_ref.c81f8f.3: %struct_type.a.b) [from "implicit.carbon"] {
// CHECK:STDOUT:   %S: %struct_type.a.b = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%S) [symbolic = %C (constants.%C.720)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [symbolic = %ptr (constants.%ptr.9e0)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.d77)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Implicit.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc11: ref %C.a8a = temporary_storage
// CHECK:STDOUT:   %F.call: init %C.a8a = call %F.ref() to %.loc11
// CHECK:STDOUT:   assign file.%c_bad.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT:   %C => constants.%C.720
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.abc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT:   %C => constants.%C.720
// CHECK:STDOUT:   %ptr => constants.%ptr.9e0
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct) {
// CHECK:STDOUT:   %S => constants.%struct
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_value.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.595: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.595 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9a6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.458: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.5ca: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %S: %struct_type.a.b.5ca = bind_symbolic_name S, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C.720: type = class_type @C, @C(%S) [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.abc: <witness> = impl_witness imports.%Destroy.impl_witness_table.2ab, @C.as.Destroy.impl(%S) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%S) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.9e0: type = ptr_type %C.720 [symbolic]
// CHECK:STDOUT:   %pattern_type.d77: type = pattern_type %ptr.9e0 [symbolic]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b9e: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.ea0: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2ce: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1ad, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e14: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.4cb: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e14 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.b9e = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.2ce) [concrete]
// CHECK:STDOUT:   %.940: type = fn_type_with_self_type %ImplicitAs.Convert.type.ea0, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a00: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.4cb [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.4cb, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.fc5: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.d47: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.694: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.4cb [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.ea3: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %struct.8e6: %struct_type.a.b.5ca = struct_value (%int_3.d47, %int_4.ea3) [concrete]
// CHECK:STDOUT:   %C.5a7: type = class_type @C, @C(%struct.8e6) [concrete]
// CHECK:STDOUT:   %pattern_type.366: type = pattern_type %C.5a7 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_2: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_1: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct.cd9: %struct_type.a.b.5ca = struct_value (%int_1, %int_2) [concrete]
// CHECK:STDOUT:   %C.a8a: type = class_type @C, @C(%struct.cd9) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Implicit.a_ref = import_ref Implicit//default, a_ref, unloaded
// CHECK:STDOUT:   %Implicit.b_ref = import_ref Implicit//default, b_ref, unloaded
// CHECK:STDOUT:   %Implicit.C: %C.type = import_ref Implicit//default, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Implicit.F: %F.type = import_ref Implicit//default, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import_ref.773: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9a6) = import_ref Implicit//default, inst151 [indirect], loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.458)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1ad = impl_witness_table (%Implicit.import_ref.773), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Implicit.import_ref.48d = import_ref Implicit//default, loc8_33, unloaded
// CHECK:STDOUT:   %Implicit.import_ref.c81f8f.1: %struct_type.a.b.5ca = import_ref Implicit//default, loc8_9, loaded [symbolic = @C.%S (constants.%S)]
// CHECK:STDOUT:   %Implicit.import_ref.8f2: <witness> = import_ref Implicit//default, loc8_34, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Implicit.import_ref.b8b = import_ref Implicit//default, inst490 [no loc], unloaded
// CHECK:STDOUT:   %Implicit.import_ref.c81f8f.2: %struct_type.a.b.5ca = import_ref Implicit//default, loc8_9, loaded [symbolic = @C.%S (constants.%S)]
// CHECK:STDOUT:   %Implicit.import_ref.ba3: type = import_ref Implicit//default, loc8_33, loaded [symbolic = @C.as.Destroy.impl.%C (constants.%C.720)]
// CHECK:STDOUT:   %Implicit.import_ref.cb9: type = import_ref Implicit//default, inst109 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Implicit.import_ref.7f2ca0.1 = import_ref Implicit//default, loc8_33, unloaded
// CHECK:STDOUT:   %Implicit.import_ref.7f2ca0.2 = import_ref Implicit//default, loc8_33, unloaded
// CHECK:STDOUT:   %Destroy.impl_witness_table.2ab = impl_witness_table (%Implicit.import_ref.7f2ca0.2), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Implicit.import_ref.c81f8f.3: %struct_type.a.b.5ca = import_ref Implicit//default, loc8_9, loaded [symbolic = @C.%S (constants.%S)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.595 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .a_ref = imports.%Implicit.a_ref
// CHECK:STDOUT:     .b_ref = imports.%Implicit.b_ref
// CHECK:STDOUT:     .C = imports.%Implicit.C
// CHECK:STDOUT:     .F = imports.%Implicit.F
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .c_bad = %c_bad
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c_bad.patt: %pattern_type.366 = binding_pattern c_bad [concrete]
// CHECK:STDOUT:     %c_bad.var_patt: %pattern_type.366 = var_pattern %c_bad.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_bad.var: ref %C.5a7 = var %c_bad.var_patt [concrete]
// CHECK:STDOUT:   %.loc10_30.1: type = splice_block %C [concrete = constants.%C.5a7] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Implicit.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:     %.loc10_29.1: %struct_type.a.b.cfd = struct_literal (%int_3, %int_4)
// CHECK:STDOUT:     %impl.elem0.loc10_29.1: %.940 = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.4cb]
// CHECK:STDOUT:     %bound_method.loc10_29.1: <bound method> = bound_method %int_3, %impl.elem0.loc10_29.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a00]
// CHECK:STDOUT:     %specific_fn.loc10_29.1: <specific function> = specific_function %impl.elem0.loc10_29.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_29.2: <bound method> = bound_method %int_3, %specific_fn.loc10_29.1 [concrete = constants.%bound_method.fc5]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_29.1: init %i32 = call %bound_method.loc10_29.2(%int_3) [concrete = constants.%int_3.d47]
// CHECK:STDOUT:     %.loc10_29.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_29.1 [concrete = constants.%int_3.d47]
// CHECK:STDOUT:     %.loc10_29.3: %i32 = converted %int_3, %.loc10_29.2 [concrete = constants.%int_3.d47]
// CHECK:STDOUT:     %impl.elem0.loc10_29.2: %.940 = impl_witness_access constants.%ImplicitAs.impl_witness.2ce, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.4cb]
// CHECK:STDOUT:     %bound_method.loc10_29.3: <bound method> = bound_method %int_4, %impl.elem0.loc10_29.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.694]
// CHECK:STDOUT:     %specific_fn.loc10_29.2: <specific function> = specific_function %impl.elem0.loc10_29.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_29.4: <bound method> = bound_method %int_4, %specific_fn.loc10_29.2 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_29.2: init %i32 = call %bound_method.loc10_29.4(%int_4) [concrete = constants.%int_4.ea3]
// CHECK:STDOUT:     %.loc10_29.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_29.2 [concrete = constants.%int_4.ea3]
// CHECK:STDOUT:     %.loc10_29.5: %i32 = converted %int_4, %.loc10_29.4 [concrete = constants.%int_4.ea3]
// CHECK:STDOUT:     %struct: %struct_type.a.b.5ca = struct_value (%.loc10_29.3, %.loc10_29.5) [concrete = constants.%struct.8e6]
// CHECK:STDOUT:     %.loc10_30.2: %struct_type.a.b.5ca = converted %.loc10_29.1, %struct [concrete = constants.%struct.8e6]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%struct.8e6) [concrete = constants.%C.5a7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c_bad: ref %C.5a7 = bind_name c_bad, %c_bad.var [concrete = %c_bad.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Destroy.impl(imports.%Implicit.import_ref.c81f8f.2: %struct_type.a.b.5ca) [from "implicit.carbon"] {
// CHECK:STDOUT:   %S: %struct_type.a.b.5ca = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%S) [symbolic = %C (constants.%C.720)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table.2ab, @C.as.Destroy.impl(%S) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.abc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%S) [symbolic = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = struct_value () [symbolic = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Implicit.import_ref.ba3 as imports.%Implicit.import_ref.cb9 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = imports.%Implicit.import_ref.7f2ca0.1
// CHECK:STDOUT:     witness = imports.%Implicit.import_ref.48d
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Implicit.import_ref.c81f8f.1: %struct_type.a.b.5ca) [from "implicit.carbon"] {
// CHECK:STDOUT:   %S: %struct_type.a.b.5ca = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Implicit.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Implicit.import_ref.b8b
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Destroy.impl.Op(imports.%Implicit.import_ref.c81f8f.3: %struct_type.a.b.5ca) [from "implicit.carbon"] {
// CHECK:STDOUT:   %S: %struct_type.a.b.5ca = bind_symbolic_name S, 0 [symbolic = %S (constants.%S)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%S) [symbolic = %C (constants.%C.720)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [symbolic = %ptr (constants.%ptr.9e0)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.d77)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F [from "implicit.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Implicit.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc10_36: ref %C.a8a = temporary_storage
// CHECK:STDOUT:   %F.call: init %C.a8a = call %F.ref() to %.loc10_36
// CHECK:STDOUT:   %.loc10_1: %C.5a7 = converted %F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%c_bad.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT:   %C => constants.%C.720
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.abc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%S) {
// CHECK:STDOUT:   %S => constants.%S
// CHECK:STDOUT:   %C => constants.%C.720
// CHECK:STDOUT:   %ptr => constants.%ptr.9e0
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct.8e6) {
// CHECK:STDOUT:   %S => constants.%struct.8e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%struct.cd9) {
// CHECK:STDOUT:   %S => constants.%struct.cd9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
