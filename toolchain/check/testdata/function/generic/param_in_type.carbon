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
// CHECK:STDOUT:   %.Self.eb1: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = bind_symbolic_name From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.eb9: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.958: %Int.as.ImplicitAs.impl.Convert.type.eb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6fb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.13c, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.d5d: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.87c: %Int.as.ImplicitAs.impl.Convert.type.d5d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.139 = facet_value %i32, (%ImplicitAs.impl_witness.6fb) [concrete]
// CHECK:STDOUT:   %.2d1: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.51e, %Int.as.ImplicitAs.impl.Convert.87c [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.87c, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %N.51e, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method(%N.51e) [symbolic]
// CHECK:STDOUT:   %array_type: type = array_type %Int.as.ImplicitAs.impl.Convert.call, %i32 [symbolic]
// CHECK:STDOUT:   %ptr.bce: type = ptr_type %array_type [symbolic]
// CHECK:STDOUT:   %pattern_type.82b: type = pattern_type %ptr.bce [symbolic]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.02e: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.eb9) = import_ref Core//prelude/parts/int, loc20_44, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.958)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.13c = impl_witness_table (%Core.import_ref.02e), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt: %pattern_type.7ce = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type (%pattern_type.82b) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type (%pattern_type.82b) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_10: type = splice_block %i32.loc15_10 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.eb1]
// CHECK:STDOUT:       %int_32.loc15_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc15_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc15_6.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc15_6.1 (constants.%N.51e)]
// CHECK:STDOUT:     %a.param: @F.%ptr.loc15_31.1 (%ptr.bce) = value_param call_param0
// CHECK:STDOUT:     %.loc15_31: type = splice_block %ptr.loc15_31.2 [symbolic = %ptr.loc15_31.1 (constants.%ptr.bce)] {
// CHECK:STDOUT:       %int_32.loc15_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc15_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc15_6.2 [symbolic = %N.loc15_6.1 (constants.%N.51e)]
// CHECK:STDOUT:       %impl.elem0: %.2d1 = impl_witness_access constants.%ImplicitAs.impl_witness.6fb, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.87c]
// CHECK:STDOUT:       %bound_method.loc15_29.2: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc15_29.3: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc15_29.1 (constants.%bound_method)]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call.loc15_29.2: init Core.IntLiteral = call %bound_method.loc15_29.3(%N.ref) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc15_29.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc15_29.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc15_29.2 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc15_29.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc15_29.2: Core.IntLiteral = converted %N.ref, %.loc15_29.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc15_29.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %array_type.loc15_30.2: type = array_type %.loc15_29.2, %i32.loc15_24 [symbolic = %array_type.loc15_30.1 (constants.%array_type)]
// CHECK:STDOUT:       %ptr.loc15_31.2: type = ptr_type %array_type.loc15_30.2 [symbolic = %ptr.loc15_31.1 (constants.%ptr.bce)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%ptr.loc15_31.1 (%ptr.bce) = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc15_6.2: %i32) {
// CHECK:STDOUT:   %N.loc15_6.1: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc15_6.1 (constants.%N.51e)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc15_6.1, constants.%Int.as.ImplicitAs.impl.Convert.87c [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc15_29.1: <bound method> = bound_method %N.loc15_6.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc15_29.1 (constants.%bound_method)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc15_29.1: init Core.IntLiteral = call %bound_method.loc15_29.1(%N.loc15_6.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc15_29.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %array_type.loc15_30.1: type = array_type %Int.as.ImplicitAs.impl.Convert.call.loc15_29.1, constants.%i32 [symbolic = %array_type.loc15_30.1 (constants.%array_type)]
// CHECK:STDOUT:   %ptr.loc15_31.1: type = ptr_type %array_type.loc15_30.1 [symbolic = %ptr.loc15_31.1 (constants.%ptr.bce)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc15_31.1 [symbolic = %pattern_type (constants.%pattern_type.82b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%ptr.loc15_31.1 (%ptr.bce));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc15_6.1 => constants.%N.51e
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound => constants.%Int.as.ImplicitAs.impl.Convert.bound
// CHECK:STDOUT:   %bound_method.loc15_29.1 => constants.%bound_method
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc15_29.1 => constants.%Int.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %array_type.loc15_30.1 => constants.%array_type
// CHECK:STDOUT:   %ptr.loc15_31.1 => constants.%ptr.bce
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.82b
// CHECK:STDOUT: }
// CHECK:STDOUT:
