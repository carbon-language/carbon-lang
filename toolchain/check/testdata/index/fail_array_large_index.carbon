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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/fail_array_large_index.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/fail_array_large_index.carbon

var a: array(i32, 1) = (12,);

// CHECK:STDERR: fail_array_large_index.carbon:[[@LINE+4]]:16: error: array index `1` is past the end of type `array(i32, 1)` [ArrayIndexOutOfBounds]
// CHECK:STDERR: var b: i32 = a[1];
// CHECK:STDERR:                ^
// CHECK:STDERR:
var b: i32 = a[1];

// CHECK:STDERR: fail_array_large_index.carbon:[[@LINE+4]]:16: error: array index `2147483647` is past the end of type `array(i32, 1)` [ArrayIndexOutOfBounds]
// CHECK:STDERR: var c: i32 = a[0x7FFF_FFFF];
// CHECK:STDERR:                ^~~~~~~~~~~
// CHECK:STDERR:
var c: i32 = a[0x7FFF_FFFF];

// CHECK:STDOUT: --- fail_array_large_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6f3: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %int_12.6a3: Core.IntLiteral = int_value 12 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.3bc: %tuple.type.985 = tuple_value (%int_12.6a3) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.3a1: <bound method> = bound_method %int_12.6a3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.839: <bound method> = bound_method %int_12.6a3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_12.f43: %i32 = int_value 12 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_12.f43) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.33d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d5e: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.32d: <witness> = impl_witness imports.%Copy.impl_witness_table.07a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.4a0: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c4a: %Int.as.Copy.impl.Op.type.4a0 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.32d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.afe: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a5f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.afe, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c4a, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %int_2147483647.d89: Core.IntLiteral = int_value 2147483647 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.565: <bound method> = bound_method %int_2147483647.d89, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.0b4: <bound method> = bound_method %int_2147483647.d89, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2147483647.cb2: %i32 = int_value 2147483647 [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.809: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c85)]
// CHECK:STDOUT:   %Copy.impl_witness_table.07a = impl_witness_table (%Core.import_ref.809), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.6f3 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.6f3 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc15: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %i32.loc15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %int_1, %i32.loc15 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.c6b = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.c6b = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var %b.var_patt [concrete]
// CHECK:STDOUT:   %i32.loc21: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: ref %i32 = ref_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.c6b = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.c6b = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %i32 = var %c.var_patt [concrete]
// CHECK:STDOUT:   %i32.loc27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %c: ref %i32 = ref_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete = constants.%int_12.6a3]
// CHECK:STDOUT:   %.loc15_28.1: %tuple.type.985 = tuple_literal (%int_12) [concrete = constants.%tuple.3bc]
// CHECK:STDOUT:   %impl.elem0.loc15: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc15_28.1: <bound method> = bound_method %int_12, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.3a1]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_28.2: <bound method> = bound_method %int_12, %specific_fn.loc15 [concrete = constants.%bound_method.839]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %i32 = call %bound_method.loc15_28.2(%int_12) [concrete = constants.%int_12.f43]
// CHECK:STDOUT:   %.loc15_28.2: init %i32 = converted %int_12, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_12.f43]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc15_28.3: ref %i32 = array_index file.%a.var, %int_0
// CHECK:STDOUT:   %.loc15_28.4: init %i32 to %.loc15_28.3 = in_place_init %.loc15_28.2 [concrete = constants.%int_12.f43]
// CHECK:STDOUT:   %.loc15_28.5: init %array_type to file.%a.var = array_init (%.loc15_28.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc15_1: init %array_type = converted %.loc15_28.1, %.loc15_28.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc15_1
// CHECK:STDOUT:   %a.ref.loc21: ref %array_type = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc21_16: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc21_16.1: <bound method> = bound_method %int_1, %impl.elem0.loc21_16 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d]
// CHECK:STDOUT:   %specific_fn.loc21_16: <specific function> = specific_function %impl.elem0.loc21_16, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_16.2: <bound method> = bound_method %int_1, %specific_fn.loc21_16 [concrete = constants.%bound_method.33d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %i32 = call %bound_method.loc21_16.2(%int_1) [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc21_16.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc21_16.2: %i32 = converted %int_1, %.loc21_16.1 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc21_17.1: ref %i32 = array_index %a.ref.loc21, %.loc21_16.2 [concrete = <error>]
// CHECK:STDOUT:   %.loc21_17.2: %i32 = acquire_value %.loc21_17.1 [concrete = <error>]
// CHECK:STDOUT:   %impl.elem0.loc21_17: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc21_17.1: <bound method> = bound_method %.loc21_17.2, %impl.elem0.loc21_17 [concrete = <error>]
// CHECK:STDOUT:   %specific_fn.loc21_17: <specific function> = specific_function %impl.elem0.loc21_17, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_17.2: <bound method> = bound_method %.loc21_17.2, %specific_fn.loc21_17 [concrete = <error>]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc21: init %i32 = call %bound_method.loc21_17.2(%.loc21_17.2) [concrete = <error>]
// CHECK:STDOUT:   assign file.%b.var, %Int.as.Copy.impl.Op.call.loc21
// CHECK:STDOUT:   %a.ref.loc27: ref %array_type = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [concrete = constants.%int_2147483647.d89]
// CHECK:STDOUT:   %impl.elem0.loc27_16: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc27_16.1: <bound method> = bound_method %int_2147483647, %impl.elem0.loc27_16 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.565]
// CHECK:STDOUT:   %specific_fn.loc27_16: <specific function> = specific_function %impl.elem0.loc27_16, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_16.2: <bound method> = bound_method %int_2147483647, %specific_fn.loc27_16 [concrete = constants.%bound_method.0b4]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc27: init %i32 = call %bound_method.loc27_16.2(%int_2147483647) [concrete = constants.%int_2147483647.cb2]
// CHECK:STDOUT:   %.loc27_16.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc27 [concrete = constants.%int_2147483647.cb2]
// CHECK:STDOUT:   %.loc27_16.2: %i32 = converted %int_2147483647, %.loc27_16.1 [concrete = constants.%int_2147483647.cb2]
// CHECK:STDOUT:   %.loc27_27.1: ref %i32 = array_index %a.ref.loc27, %.loc27_16.2 [concrete = <error>]
// CHECK:STDOUT:   %.loc27_27.2: %i32 = acquire_value %.loc27_27.1 [concrete = <error>]
// CHECK:STDOUT:   %impl.elem0.loc27_27: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc27_27.1: <bound method> = bound_method %.loc27_27.2, %impl.elem0.loc27_27 [concrete = <error>]
// CHECK:STDOUT:   %specific_fn.loc27_27: <specific function> = specific_function %impl.elem0.loc27_27, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_27.2: <bound method> = bound_method %.loc27_27.2, %specific_fn.loc27_27 [concrete = <error>]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc27: init %i32 = call %bound_method.loc27_27.2(%.loc27_27.2) [concrete = <error>]
// CHECK:STDOUT:   assign file.%c.var, %Int.as.Copy.impl.Op.call.loc27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
