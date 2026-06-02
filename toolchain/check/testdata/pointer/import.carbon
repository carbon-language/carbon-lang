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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/import.carbon

// --- implicit.carbon

package Implicit;

var a_orig: i32 = 0;
var a_ref: i32* = &a_orig;

// --- implicit.impl.carbon

impl package Implicit;

var a: i32* = a_ref;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.104: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.754: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.754 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.360: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3d6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0bd, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.360 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3d6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.474: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da5, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.82d: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.62d: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %ptr.847: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7b5: type = pattern_type %ptr.847 [concrete]
// CHECK:STDOUT:   %addr: %ptr.847 = addr_of file.%a_orig.var [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.aa4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.5a1: %ptr.as.Copy.impl.Op.type.aa4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.c37: <witness> = impl_witness imports.%Copy.impl_witness_table.6b9, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ae4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.6e4: %ptr.as.Copy.impl.Op.type.ae4 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.847, (%Copy.impl_witness.c37) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.165: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.cd7: type = fn_type_with_self_type %Copy.WithSelf.Op.type.165, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.6e4 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.6e4, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %bound_method.ea9: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.754 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.b42: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0bd = impl_witness_table (%Core.import_ref.b42), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.aa4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.aa4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.5a1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.6b9 = impl_witness_table (%Core.import_ref.aa4), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a_orig = %a_orig
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_orig.patt: %pattern_type.104 = ref_binding_pattern a_orig [concrete]
// CHECK:STDOUT:     %a_orig.var_patt: %pattern_type.104 = var_pattern %a_orig.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_orig.var: ref %i32 = var %a_orig.var_patt [concrete]
// CHECK:STDOUT:   %i32.loc4: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a_orig: ref %i32 = ref_binding a_orig, %a_orig.var [concrete = %a_orig.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_ref.patt: %pattern_type.7b5 = ref_binding_pattern a_ref [concrete]
// CHECK:STDOUT:     %a_ref.var_patt: %pattern_type.7b5 = var_pattern %a_ref.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ref.var: ref %ptr.847 = var %a_ref.var_patt [concrete]
// CHECK:STDOUT:   %.loc5: type = splice_block %ptr [concrete = constants.%ptr.847] {
// CHECK:STDOUT:     %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32.loc5 [concrete = constants.%ptr.847]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ref: ref %ptr.847 = ref_binding a_ref, %a_ref.var [concrete = %a_ref.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc4: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc4_1.1: <bound method> = bound_method %int_0, %impl.elem0.loc4 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc4: <specific function> = specific_function %impl.elem0.loc4, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_1.2: <bound method> = bound_method %int_0, %specific_fn.loc4 [concrete = constants.%bound_method.82d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc4_1.2(%int_0) [concrete = constants.%int_0.62d]
// CHECK:STDOUT:   %.loc4: init %i32 = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.62d]
// CHECK:STDOUT:   assign file.%a_orig.var, %.loc4
// CHECK:STDOUT:   %a_orig.ref: ref %i32 = name_ref a_orig, file.%a_orig [concrete = file.%a_orig.var]
// CHECK:STDOUT:   %addr: %ptr.847 = addr_of %a_orig.ref [concrete = constants.%addr]
// CHECK:STDOUT:   %impl.elem0.loc5: %.cd7 = impl_witness_access constants.%Copy.impl_witness.c37, element0 [concrete = constants.%ptr.as.Copy.impl.Op.6e4]
// CHECK:STDOUT:   %bound_method.loc5_19.1: <bound method> = bound_method %addr, %impl.elem0.loc5 [concrete = constants.%ptr.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc5: <specific function> = specific_function %impl.elem0.loc5, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_19.2: <bound method> = bound_method %addr, %specific_fn.loc5 [concrete = constants.%bound_method.ea9]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.847 = call %bound_method.loc5_19.2(%addr) [concrete = constants.%addr]
// CHECK:STDOUT:   assign file.%a_ref.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.05a: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.fad: %ptr.as.Copy.impl.Op.type.05a = struct_value () [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.b07: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c12: type = pattern_type %ptr.b07 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.22b: <witness> = impl_witness imports.%Copy.impl_witness_table.6eb, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.abd: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.958: %ptr.as.Copy.impl.Op.type.abd = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.b07, (%Copy.impl_witness.22b) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.5c2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.227: type = fn_type_with_self_type %Copy.WithSelf.Op.type.5c2, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.958, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Implicit.a_orig = import_ref Implicit//default, a_orig, unloaded
// CHECK:STDOUT:   %Implicit.a_ref: ref %ptr.b07 = import_ref Implicit//default, a_ref, loaded [concrete = %a_ref.var]
// CHECK:STDOUT:   %Core.979: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import_ref.33d: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.05a) = import_ref Implicit//default, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.fad)]
// CHECK:STDOUT:   %Copy.impl_witness_table.6eb = impl_witness_table (%Implicit.import_ref.33d), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %a_ref.patt: %pattern_type.c12 = ref_binding_pattern a_ref [concrete]
// CHECK:STDOUT:   %a_ref.var_patt: %pattern_type.c12 = var_pattern %a_ref.patt [concrete]
// CHECK:STDOUT:   %a_ref.var: ref %ptr.b07 = var %a_ref.var_patt [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .a_orig = imports.%Implicit.a_orig
// CHECK:STDOUT:     .a_ref = imports.%Implicit.a_ref
// CHECK:STDOUT:     .Core = imports.%Core.979
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Implicit.import = import Implicit
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c12 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.c12 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %ptr.b07 = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc4: type = splice_block %ptr [concrete = constants.%ptr.b07] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32 [concrete = constants.%ptr.b07]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %ptr.b07 = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %ptr.b07 = name_ref a_ref, imports.%Implicit.a_ref [concrete = imports.%a_ref.var]
// CHECK:STDOUT:   %.loc4: %ptr.b07 = acquire_value %a_ref.ref
// CHECK:STDOUT:   %impl.elem0: %.227 = impl_witness_access constants.%Copy.impl_witness.22b, element0 [concrete = constants.%ptr.as.Copy.impl.Op.958]
// CHECK:STDOUT:   %bound_method.loc4_15.1: <bound method> = bound_method %.loc4, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_15.2: <bound method> = bound_method %.loc4, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.b07 = call %bound_method.loc4_15.2(%.loc4)
// CHECK:STDOUT:   assign file.%a.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
