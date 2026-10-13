// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/param_in_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/param_in_type.carbon

fn F(generic N: i32, a: array(i32, N)*);

// CHECK:STDOUT: --- param_in_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.282: %pattern_type.df9 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N.06f: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.0ab: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.fab: %Int.as.ImplicitAs.impl.Convert.type.0ab = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.34b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9ad, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.f74: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.412: %Int.as.ImplicitAs.impl.Convert.type.f74 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.7cb = facet_value %i32, (%ImplicitAs.impl_witness.34b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.7e7: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.53e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.7e7, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.06f, %Int.as.ImplicitAs.impl.Convert.412 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.412, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %N.06f, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method(%N.06f) [symbolic]
// CHECK:STDOUT:   %Array.type: type = parameterized_class_name_type @Array [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.a68: type = class_type @Array, @Array(%i32, %Int.as.ImplicitAs.impl.Convert.call) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %Array.a68 [symbolic]
// CHECK:STDOUT:   %pattern_type.ade: type = pattern_type %ptr [symbolic]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.ade = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt: %pattern_type.ade = wrapper_binding_pattern a, %a.param_patt [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.03c: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.0ab) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.fab)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9ad = impl_witness_table (%Core.import_ref.03c), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc15_15.1: %pattern_type.df9 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc15_15.2 (constants.%N.patt.282)]
// CHECK:STDOUT:     %a.param_patt.loc15_23.1: @F.%pattern_type (%pattern_type.ade) = value_param_pattern [symbolic = %a.param_patt.loc15_23.2 (constants.%a.param_patt)]
// CHECK:STDOUT:     %a.patt.loc15_23.1: @F.%pattern_type (%pattern_type.ade) = wrapper_binding_pattern a, %a.param_patt.loc15_23.1 [symbolic = %a.patt.loc15_23.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_17: type = splice_block %i32.loc15_17 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %i32.loc15_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc15_15.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc15_15.1 (constants.%N.06f)]
// CHECK:STDOUT:     %a.param: @F.%ptr.loc15_38.1 (%ptr) = value_param call_param0
// CHECK:STDOUT:     %.loc15_38: type = splice_block %ptr.loc15_38.2 [symbolic = %ptr.loc15_38.1 (constants.%ptr)] {
// CHECK:STDOUT:       %i32.loc15_31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc15_15.2 [symbolic = %N.loc15_15.1 (constants.%N.06f)]
// CHECK:STDOUT:       %impl.elem0: %.53e = impl_witness_access constants.%ImplicitAs.impl_witness.34b, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.412]
// CHECK:STDOUT:       %bound_method.loc15_36.2: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc15_36.3: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc15_36.1 (constants.%bound_method)]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call.loc15_36.2: init Core.IntLiteral = call %bound_method.loc15_36.3(%N.ref) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc15_36.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc15_36.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc15_36.2 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc15_36.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc15_36.2: Core.IntLiteral = converted %N.ref, %.loc15_36.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc15_36.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %Array.loc15_37.2: type = class_type @Array, @Array(constants.%i32, constants.%Int.as.ImplicitAs.impl.Convert.call) [symbolic = %Array.loc15_37.1 (constants.%Array.a68)]
// CHECK:STDOUT:       %ptr.loc15_38.2: type = ptr_type %Array.loc15_37.2 [symbolic = %ptr.loc15_38.1 (constants.%ptr)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%ptr.loc15_38.1 (%ptr) = wrapper_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc15_15.2: %i32) {
// CHECK:STDOUT:   %N.patt.loc15_15.2: %pattern_type.df9 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc15_15.2 (constants.%N.patt.282)]
// CHECK:STDOUT:   %N.loc15_15.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc15_15.1 (constants.%N.06f)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc15_15.1, constants.%Int.as.ImplicitAs.impl.Convert.412 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc15_36.1: <bound method> = bound_method %N.loc15_15.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc15_36.1 (constants.%bound_method)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc15_36.1: init Core.IntLiteral = call %bound_method.loc15_36.1(%N.loc15_15.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc15_36.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %Array.loc15_37.1: type = class_type @Array, @Array(constants.%i32, %Int.as.ImplicitAs.impl.Convert.call.loc15_36.1) [symbolic = %Array.loc15_37.1 (constants.%Array.a68)]
// CHECK:STDOUT:   %ptr.loc15_38.1: type = ptr_type %Array.loc15_37.1 [symbolic = %ptr.loc15_38.1 (constants.%ptr)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc15_38.1 [symbolic = %pattern_type (constants.%pattern_type.ade)]
// CHECK:STDOUT:   %a.param_patt.loc15_23.2: @F.%pattern_type (%pattern_type.ade) = value_param_pattern [symbolic = %a.param_patt.loc15_23.2 (constants.%a.param_patt)]
// CHECK:STDOUT:   %a.patt.loc15_23.2: @F.%pattern_type (%pattern_type.ade) = wrapper_binding_pattern a, %a.param_patt.loc15_23.2 [symbolic = %a.patt.loc15_23.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%ptr.loc15_38.1 (%ptr));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.06f) {
// CHECK:STDOUT:   %N.patt.loc15_15.2 => constants.%N.patt.282
// CHECK:STDOUT:   %N.loc15_15.1 => constants.%N.06f
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound => constants.%Int.as.ImplicitAs.impl.Convert.bound
// CHECK:STDOUT:   %bound_method.loc15_36.1 => constants.%bound_method
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc15_36.1 => constants.%Int.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %Array.loc15_37.1 => constants.%Array.a68
// CHECK:STDOUT:   %ptr.loc15_38.1 => constants.%ptr
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ade
// CHECK:STDOUT:   %a.param_patt.loc15_23.2 => constants.%a.param_patt
// CHECK:STDOUT:   %a.patt.loc15_23.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
