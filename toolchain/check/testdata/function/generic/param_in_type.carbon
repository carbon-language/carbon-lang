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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/param_in_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/param_in_type.carbon

fn F(N:! i32, a: array(i32, N)*);

// CHECK:STDOUT: --- param_in_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.38a: %pattern_type.6b6 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N.37f: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.48d: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.0f9: %Int.as.ImplicitAs.impl.Convert.type.48d = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.416: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b8b, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.683: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.69e: %Int.as.ImplicitAs.impl.Convert.type.683 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.7cb = facet_value %i32, (%ImplicitAs.impl_witness.416) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.dc6: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b08: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.dc6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.37f, %Int.as.ImplicitAs.impl.Convert.69e [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.69e, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %N.37f, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method(%N.37f) [symbolic]
// CHECK:STDOUT:   %array_type: type = array_type %Int.as.ImplicitAs.impl.Convert.call, %i32 [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [symbolic]
// CHECK:STDOUT:   %pattern_type.5a4: type = pattern_type %ptr [symbolic]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.5a4 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt: %pattern_type.5a4 = at_binding_pattern a, %a.param_patt [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.32e: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.48d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.0f9)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b8b = impl_witness_table (%Core.import_ref.32e), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc15_7.1: %pattern_type.6b6 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc15_7.2 (constants.%N.patt.38a)]
// CHECK:STDOUT:     %a.param_patt.loc15_16.1: @F.%pattern_type (%pattern_type.5a4) = value_param_pattern [symbolic = %a.param_patt.loc15_16.2 (constants.%a.param_patt)]
// CHECK:STDOUT:     %a.patt.loc15_16.1: @F.%pattern_type (%pattern_type.5a4) = at_binding_pattern a, %a.param_patt.loc15_16.1 [symbolic = %a.patt.loc15_16.2 (constants.%a.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_10: type = splice_block %i32.loc15_10 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32.loc15_10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc15_7.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc15_7.1 (constants.%N.37f)]
// CHECK:STDOUT:     %a.param: @F.%ptr.loc15_31.1 (%ptr) = value_param call_param0
// CHECK:STDOUT:     %.loc15_31: type = splice_block %ptr.loc15_31.2 [symbolic = %ptr.loc15_31.1 (constants.%ptr)] {
// CHECK:STDOUT:       %i32.loc15_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc15_7.2 [symbolic = %N.loc15_7.1 (constants.%N.37f)]
// CHECK:STDOUT:       %impl.elem0: %.b08 = impl_witness_access constants.%ImplicitAs.impl_witness.416, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.69e]
// CHECK:STDOUT:       %bound_method.loc15_29.2: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc15_29.3: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc15_29.1 (constants.%bound_method)]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call.loc15_29.2: init Core.IntLiteral = call %bound_method.loc15_29.3(%N.ref) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc15_29.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc15_29.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc15_29.2 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc15_29.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc15_29.2: Core.IntLiteral = converted %N.ref, %.loc15_29.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc15_29.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %array_type.loc15_30.2: type = array_type %.loc15_29.2, %i32.loc15_24 [symbolic = %array_type.loc15_30.1 (constants.%array_type)]
// CHECK:STDOUT:       %ptr.loc15_31.2: type = ptr_type %array_type.loc15_30.2 [symbolic = %ptr.loc15_31.1 (constants.%ptr)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%ptr.loc15_31.1 (%ptr) = wrapper_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc15_7.2: %i32) {
// CHECK:STDOUT:   %N.patt.loc15_7.2: %pattern_type.6b6 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc15_7.2 (constants.%N.patt.38a)]
// CHECK:STDOUT:   %N.loc15_7.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc15_7.1 (constants.%N.37f)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc15_7.1, constants.%Int.as.ImplicitAs.impl.Convert.69e [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc15_29.1: <bound method> = bound_method %N.loc15_7.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc15_29.1 (constants.%bound_method)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc15_29.1: init Core.IntLiteral = call %bound_method.loc15_29.1(%N.loc15_7.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc15_29.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %array_type.loc15_30.1: type = array_type %Int.as.ImplicitAs.impl.Convert.call.loc15_29.1, constants.%i32 [symbolic = %array_type.loc15_30.1 (constants.%array_type)]
// CHECK:STDOUT:   %ptr.loc15_31.1: type = ptr_type %array_type.loc15_30.1 [symbolic = %ptr.loc15_31.1 (constants.%ptr)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc15_31.1 [symbolic = %pattern_type (constants.%pattern_type.5a4)]
// CHECK:STDOUT:   %a.param_patt.loc15_16.2: @F.%pattern_type (%pattern_type.5a4) = value_param_pattern [symbolic = %a.param_patt.loc15_16.2 (constants.%a.param_patt)]
// CHECK:STDOUT:   %a.patt.loc15_16.2: @F.%pattern_type (%pattern_type.5a4) = at_binding_pattern a, %a.param_patt.loc15_16.2 [symbolic = %a.patt.loc15_16.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%ptr.loc15_31.1 (%ptr));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.37f) {
// CHECK:STDOUT:   %N.patt.loc15_7.2 => constants.%N.patt.38a
// CHECK:STDOUT:   %N.loc15_7.1 => constants.%N.37f
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound => constants.%Int.as.ImplicitAs.impl.Convert.bound
// CHECK:STDOUT:   %bound_method.loc15_29.1 => constants.%bound_method
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc15_29.1 => constants.%Int.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %array_type.loc15_30.1 => constants.%array_type
// CHECK:STDOUT:   %ptr.loc15_31.1 => constants.%ptr
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.5a4
// CHECK:STDOUT:   %a.param_patt.loc15_16.2 => constants.%a.param_patt
// CHECK:STDOUT:   %a.patt.loc15_16.2 => constants.%a.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
