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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/tuple_as_element.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/tuple_as_element.carbon

var x: {.a: i32, .b: (i32,)} = {.a = 1, .b = (2,)};
var y: {.a: i32, .b: (i32,)} = x;

// CHECK:STDOUT: --- tuple_as_element.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.4e6: %tuple.type.85c = tuple_value (%i32) [concrete]
// CHECK:STDOUT:   %tuple.type.cb3: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.96d: type = struct_type {.a: %i32, .b: %tuple.type.cb3} [concrete]
// CHECK:STDOUT:   %pattern_type.933: type = pattern_type %struct_type.a.b.96d [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.e7e: %tuple.type.985 = tuple_value (%int_2.ecc) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.057: type = struct_type {.a: Core.IntLiteral, .b: %tuple.type.985} [concrete]
// CHECK:STDOUT:   %struct.60a: %struct_type.a.b.057 = struct_value (%int_1.5b8, %tuple.e7e) [concrete]
// CHECK:STDOUT:   %.2f43: ref %i32 = struct_access file.%x.var, element0 [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c1a: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.e5b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d9c: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %.0f4: ref %tuple.type.cb3 = struct_access file.%x.var, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem0.c57: ref %i32 = tuple_access %.0f4, element0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.bf7: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %bound_method.7d5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.228: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.b3d: %tuple.type.cb3 = tuple_value (%int_2.228) [concrete]
// CHECK:STDOUT:   %struct.86e: %struct_type.a.b.96d = struct_value (%int_1.d9c, %tuple.b3d) [concrete]
// CHECK:STDOUT:   %.dd0: ref %i32 = struct_access file.%y.var, element0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.2f2: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.a64: %Int.as.Copy.impl.Op.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1a4: <witness> = impl_witness imports.%Copy.impl_witness_table.0ac, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.422: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.ce7: %Int.as.Copy.impl.Op.type.422 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.1a4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7e2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.604: type = fn_type_with_self_type %Copy.WithSelf.Op.type.7e2, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.ce7, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %.2f47: ref %tuple.type.cb3 = struct_access file.%y.var, element1 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.55f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.2f2) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.a64)]
// CHECK:STDOUT:   %Copy.impl_witness_table.0ac = impl_witness_table (%Core.import_ref.55f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.933 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.933 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.96d = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc15_28: type = splice_block %struct_type.a.b.loc15 [concrete = constants.%struct_type.a.b.96d] {
// CHECK:STDOUT:     %i32.loc15_13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc15_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15_27.1: %tuple.type.85c = tuple_literal (%i32.loc15_23) [concrete = constants.%tuple.4e6]
// CHECK:STDOUT:     %.loc15_27.2: type = converted %.loc15_27.1, constants.%tuple.type.cb3 [concrete = constants.%tuple.type.cb3]
// CHECK:STDOUT:     %struct_type.a.b.loc15: type = struct_type {.a: %i32, .b: %tuple.type.cb3} [concrete = constants.%struct_type.a.b.96d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.a.b.96d = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.933 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.933 = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %struct_type.a.b.96d = var %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc16_28: type = splice_block %struct_type.a.b.loc16 [concrete = constants.%struct_type.a.b.96d] {
// CHECK:STDOUT:     %i32.loc16_13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc16_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc16_27.1: %tuple.type.85c = tuple_literal (%i32.loc16_23) [concrete = constants.%tuple.4e6]
// CHECK:STDOUT:     %.loc16_27.2: type = converted %.loc16_27.1, constants.%tuple.type.cb3 [concrete = constants.%tuple.type.cb3]
// CHECK:STDOUT:     %struct_type.a.b.loc16: type = struct_type {.a: %i32, .b: %tuple.type.cb3} [concrete = constants.%struct_type.a.b.96d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %struct_type.a.b.96d = ref_binding y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc15_49.1: %tuple.type.985 = tuple_literal (%int_2) [concrete = constants.%tuple.e7e]
// CHECK:STDOUT:   %.loc15_50.1: %struct_type.a.b.057 = struct_literal (%int_1, %.loc15_49.1) [concrete = constants.%struct.60a]
// CHECK:STDOUT:   %impl.elem0.loc15_50: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc15_50.1: <bound method> = bound_method %int_1, %impl.elem0.loc15_50 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c1a]
// CHECK:STDOUT:   %specific_fn.loc15_50: <specific function> = specific_function %impl.elem0.loc15_50, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_50.2: <bound method> = bound_method %int_1, %specific_fn.loc15_50 [concrete = constants.%bound_method.e5b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15_50: init %i32 = call %bound_method.loc15_50.2(%int_1) [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %.loc15_50.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15_50 [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %.loc15_50.3: ref %i32 = struct_access file.%x.var, element0 [concrete = constants.%.2f43]
// CHECK:STDOUT:   %.loc15_50.4: init %i32 to %.loc15_50.3 = in_place_init %.loc15_50.2 [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %impl.elem0.loc15_49: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc15_49.1: <bound method> = bound_method %int_2, %impl.elem0.loc15_49 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.bf7]
// CHECK:STDOUT:   %specific_fn.loc15_49: <specific function> = specific_function %impl.elem0.loc15_49, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_49.2: <bound method> = bound_method %int_2, %specific_fn.loc15_49 [concrete = constants.%bound_method.7d5]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15_49: init %i32 = call %bound_method.loc15_49.2(%int_2) [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %.loc15_49.2: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15_49 [concrete = constants.%int_2.228]
// CHECK:STDOUT:   %.loc15_50.5: ref %tuple.type.cb3 = struct_access file.%x.var, element1 [concrete = constants.%.0f4]
// CHECK:STDOUT:   %.loc15_49.3: init %tuple.type.cb3 to %.loc15_50.5 = tuple_init (%.loc15_49.2) [concrete = constants.%tuple.b3d]
// CHECK:STDOUT:   %.loc15_50.6: init %tuple.type.cb3 = converted %.loc15_49.1, %.loc15_49.3 [concrete = constants.%tuple.b3d]
// CHECK:STDOUT:   %.loc15_50.7: init %tuple.type.cb3 to %.loc15_50.5 = in_place_init %.loc15_50.6 [concrete = constants.%tuple.b3d]
// CHECK:STDOUT:   %.loc15_50.8: init %struct_type.a.b.96d to file.%x.var = struct_init (%.loc15_50.4, %.loc15_50.7) [concrete = constants.%struct.86e]
// CHECK:STDOUT:   %.loc15_1: init %struct_type.a.b.96d = converted %.loc15_50.1, %.loc15_50.8 [concrete = constants.%struct.86e]
// CHECK:STDOUT:   assign file.%x.var, %.loc15_1
// CHECK:STDOUT:   %x.ref: ref %struct_type.a.b.96d = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   %.loc16_32.1: ref %i32 = struct_access %x.ref, element0 [concrete = constants.%.2f43]
// CHECK:STDOUT:   %.loc16_32.2: %i32 = acquire_value %.loc16_32.1
// CHECK:STDOUT:   %impl.elem0.loc16_32.1: %.604 = impl_witness_access constants.%Copy.impl_witness.1a4, element0 [concrete = constants.%Int.as.Copy.impl.Op.ce7]
// CHECK:STDOUT:   %bound_method.loc16_32.1: <bound method> = bound_method %.loc16_32.2, %impl.elem0.loc16_32.1
// CHECK:STDOUT:   %specific_fn.loc16_32.1: <specific function> = specific_function %impl.elem0.loc16_32.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_32.2: <bound method> = bound_method %.loc16_32.2, %specific_fn.loc16_32.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc16_32.1: init %i32 = call %bound_method.loc16_32.2(%.loc16_32.2)
// CHECK:STDOUT:   %.loc16_32.3: ref %i32 = struct_access file.%y.var, element0 [concrete = constants.%.dd0]
// CHECK:STDOUT:   %.loc16_32.4: init %i32 to %.loc16_32.3 = in_place_init %Int.as.Copy.impl.Op.call.loc16_32.1
// CHECK:STDOUT:   %.loc16_32.5: ref %tuple.type.cb3 = struct_access %x.ref, element1 [concrete = constants.%.0f4]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %.loc16_32.5, element0 [concrete = constants.%tuple.elem0.c57]
// CHECK:STDOUT:   %.loc16_32.6: %i32 = acquire_value %tuple.elem0
// CHECK:STDOUT:   %impl.elem0.loc16_32.2: %.604 = impl_witness_access constants.%Copy.impl_witness.1a4, element0 [concrete = constants.%Int.as.Copy.impl.Op.ce7]
// CHECK:STDOUT:   %bound_method.loc16_32.3: <bound method> = bound_method %.loc16_32.6, %impl.elem0.loc16_32.2
// CHECK:STDOUT:   %specific_fn.loc16_32.2: <specific function> = specific_function %impl.elem0.loc16_32.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_32.4: <bound method> = bound_method %.loc16_32.6, %specific_fn.loc16_32.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc16_32.2: init %i32 = call %bound_method.loc16_32.4(%.loc16_32.6)
// CHECK:STDOUT:   %.loc16_32.7: ref %tuple.type.cb3 = struct_access file.%y.var, element1 [concrete = constants.%.2f47]
// CHECK:STDOUT:   %.loc16_32.8: init %tuple.type.cb3 to %.loc16_32.7 = tuple_init (%Int.as.Copy.impl.Op.call.loc16_32.2)
// CHECK:STDOUT:   %.loc16_32.9: init %tuple.type.cb3 = converted %.loc16_32.5, %.loc16_32.8
// CHECK:STDOUT:   %.loc16_32.10: init %tuple.type.cb3 to %.loc16_32.7 = in_place_init %.loc16_32.9
// CHECK:STDOUT:   %.loc16_32.11: init %struct_type.a.b.96d to file.%y.var = struct_init (%.loc16_32.4, %.loc16_32.10)
// CHECK:STDOUT:   %.loc16_1: init %struct_type.a.b.96d = converted %x.ref, %.loc16_32.11
// CHECK:STDOUT:   assign file.%y.var, %.loc16_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
