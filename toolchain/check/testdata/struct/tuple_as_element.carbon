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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.135: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.58f: %tuple.type.135 = tuple_value (%i32) [concrete]
// CHECK:STDOUT:   %tuple.type.489: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cbe: type = struct_type {.a: %i32, .b: %tuple.type.489} [concrete]
// CHECK:STDOUT:   %pattern_type.c4d: type = pattern_type %struct_type.a.b.cbe [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.c4d = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.c4d = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.e7e: %tuple.type.985 = tuple_value (%int_2.ecc) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.057: type = struct_type {.a: Core.IntLiteral, .b: %tuple.type.985} [concrete]
// CHECK:STDOUT:   %struct.60a: %struct_type.a.b.057 = struct_value (%int_1.5b8, %tuple.e7e) [concrete]
// CHECK:STDOUT:   %.01e: ref %i32 = struct_access file.%x.var, element0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cb7: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e87: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.186: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.cb7 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e87) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ac8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.208: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ac8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a5f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.186, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.67d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.874: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %.f9e: ref %tuple.type.489 = struct_access file.%x.var, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem0.ea9: ref %i32 = tuple_access %.f9e, element0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.ca1: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.92b: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.e70: %tuple.type.489 = tuple_value (%int_2.92b) [concrete]
// CHECK:STDOUT:   %struct.de3: %struct_type.a.b.cbe = struct_value (%int_1.874, %tuple.e70) [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.c4d = ref_binding_pattern y [concrete]
// CHECK:STDOUT:   %y.var_patt: %pattern_type.c4d = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   %.9a7: ref %i32 = struct_access file.%y.var, element0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.4c3: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b4b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.175: %Int.as.Copy.impl.Op.type.b4b = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.4c3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.67d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.187: type = fn_type_with_self_type %Copy.WithSelf.Op.type.67d, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.175, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %.94a: ref %tuple.type.489 = struct_access file.%y.var, element1 [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.cbe = var_storage %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc15_28: type = splice_block %struct_type.a.b.loc15 [concrete = constants.%struct_type.a.b.cbe] {
// CHECK:STDOUT:     %i32.loc15_13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc15_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15_27.1: %tuple.type.135 = tuple_literal (%i32.loc15_23) [concrete = constants.%tuple.58f]
// CHECK:STDOUT:     %.loc15_27.2: type = converted %.loc15_27.1, constants.%tuple.type.489 [concrete = constants.%tuple.type.489]
// CHECK:STDOUT:     %struct_type.a.b.loc15: type = struct_type {.a: %i32, .b: %tuple.type.489} [concrete = constants.%struct_type.a.b.cbe]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.a.b.cbe = wrapper_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.c4d = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.c4d = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %struct_type.a.b.cbe = var_storage %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc16_28: type = splice_block %struct_type.a.b.loc16 [concrete = constants.%struct_type.a.b.cbe] {
// CHECK:STDOUT:     %i32.loc16_13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc16_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc16_27.1: %tuple.type.135 = tuple_literal (%i32.loc16_23) [concrete = constants.%tuple.58f]
// CHECK:STDOUT:     %.loc16_27.2: type = converted %.loc16_27.1, constants.%tuple.type.489 [concrete = constants.%tuple.type.489]
// CHECK:STDOUT:     %struct_type.a.b.loc16: type = struct_type {.a: %i32, .b: %tuple.type.489} [concrete = constants.%struct_type.a.b.cbe]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %struct_type.a.b.cbe = wrapper_binding y, %y.var [concrete = %y.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.c4d = ref_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.c4d = var_pattern %y.patt [concrete = constants.%y.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc15_49.1: %tuple.type.985 = tuple_literal (%int_2) [concrete = constants.%tuple.e7e]
// CHECK:STDOUT:   %.loc15_50.1: %struct_type.a.b.057 = struct_literal (%int_1, %.loc15_49.1) [concrete = constants.%struct.60a]
// CHECK:STDOUT:   %impl.elem0.loc15_50: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc15_50.1: <bound method> = bound_method %int_1, %impl.elem0.loc15_50 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a5f]
// CHECK:STDOUT:   %specific_fn.loc15_50: <specific function> = specific_function %impl.elem0.loc15_50, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_50.2: <bound method> = bound_method %int_1, %specific_fn.loc15_50 [concrete = constants.%bound_method.67d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15_50: init %i32 = call %bound_method.loc15_50.2(%int_1) [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %.loc15_50.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15_50 [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %.loc15_50.3: ref %i32 = struct_access file.%x.var, element0 [concrete = constants.%.01e]
// CHECK:STDOUT:   %.loc15_50.4: init %i32 to %.loc15_50.3 = in_place_init %.loc15_50.2 [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %impl.elem0.loc15_49: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc15_49.1: <bound method> = bound_method %int_2, %impl.elem0.loc15_49 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc15_49: <specific function> = specific_function %impl.elem0.loc15_49, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_49.2: <bound method> = bound_method %int_2, %specific_fn.loc15_49 [concrete = constants.%bound_method.ca1]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15_49: init %i32 = call %bound_method.loc15_49.2(%int_2) [concrete = constants.%int_2.92b]
// CHECK:STDOUT:   %.loc15_49.2: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15_49 [concrete = constants.%int_2.92b]
// CHECK:STDOUT:   %.loc15_50.5: ref %tuple.type.489 = struct_access file.%x.var, element1 [concrete = constants.%.f9e]
// CHECK:STDOUT:   %.loc15_49.3: init %tuple.type.489 to %.loc15_50.5 = tuple_init (%.loc15_49.2) [concrete = constants.%tuple.e70]
// CHECK:STDOUT:   %.loc15_50.6: init %tuple.type.489 = converted %.loc15_49.1, %.loc15_49.3 [concrete = constants.%tuple.e70]
// CHECK:STDOUT:   %.loc15_50.7: init %tuple.type.489 to %.loc15_50.5 = in_place_init %.loc15_50.6 [concrete = constants.%tuple.e70]
// CHECK:STDOUT:   %.loc15_50.8: init %struct_type.a.b.cbe to file.%x.var = struct_init (%.loc15_50.4, %.loc15_50.7) [concrete = constants.%struct.de3]
// CHECK:STDOUT:   %.loc15_1: init %struct_type.a.b.cbe = converted %.loc15_50.1, %.loc15_50.8 [concrete = constants.%struct.de3]
// CHECK:STDOUT:   assign file.%x.var, %.loc15_1
// CHECK:STDOUT:   %x.ref: ref %struct_type.a.b.cbe = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   %.loc16_32.1: ref %i32 = struct_access %x.ref, element0 [concrete = constants.%.01e]
// CHECK:STDOUT:   %.loc16_32.2: %i32 = acquire_value %.loc16_32.1
// CHECK:STDOUT:   %impl.elem0.loc16_32.1: %.187 = impl_witness_access constants.%Copy.impl_witness.4c3, element0 [concrete = constants.%Int.as.Copy.impl.Op.175]
// CHECK:STDOUT:   %bound_method.loc16_32.1: <bound method> = bound_method %.loc16_32.2, %impl.elem0.loc16_32.1
// CHECK:STDOUT:   %specific_fn.loc16_32.1: <specific function> = specific_function %impl.elem0.loc16_32.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_32.2: <bound method> = bound_method %.loc16_32.2, %specific_fn.loc16_32.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc16_32.1: init %i32 = call %bound_method.loc16_32.2(%.loc16_32.2)
// CHECK:STDOUT:   %.loc16_32.3: ref %i32 = struct_access file.%y.var, element0 [concrete = constants.%.9a7]
// CHECK:STDOUT:   %.loc16_32.4: init %i32 to %.loc16_32.3 = in_place_init %Int.as.Copy.impl.Op.call.loc16_32.1
// CHECK:STDOUT:   %.loc16_32.5: ref %tuple.type.489 = struct_access %x.ref, element1 [concrete = constants.%.f9e]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %.loc16_32.5, element0 [concrete = constants.%tuple.elem0.ea9]
// CHECK:STDOUT:   %.loc16_32.6: %i32 = acquire_value %tuple.elem0
// CHECK:STDOUT:   %impl.elem0.loc16_32.2: %.187 = impl_witness_access constants.%Copy.impl_witness.4c3, element0 [concrete = constants.%Int.as.Copy.impl.Op.175]
// CHECK:STDOUT:   %bound_method.loc16_32.3: <bound method> = bound_method %.loc16_32.6, %impl.elem0.loc16_32.2
// CHECK:STDOUT:   %specific_fn.loc16_32.2: <specific function> = specific_function %impl.elem0.loc16_32.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_32.4: <bound method> = bound_method %.loc16_32.6, %specific_fn.loc16_32.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc16_32.2: init %i32 = call %bound_method.loc16_32.4(%.loc16_32.6)
// CHECK:STDOUT:   %.loc16_32.7: ref %tuple.type.489 = struct_access file.%y.var, element1 [concrete = constants.%.94a]
// CHECK:STDOUT:   %.loc16_32.8: init %tuple.type.489 to %.loc16_32.7 = tuple_init (%Int.as.Copy.impl.Op.call.loc16_32.2)
// CHECK:STDOUT:   %.loc16_32.9: init %tuple.type.489 = converted %.loc16_32.5, %.loc16_32.8
// CHECK:STDOUT:   %.loc16_32.10: init %tuple.type.489 to %.loc16_32.7 = in_place_init %.loc16_32.9
// CHECK:STDOUT:   %.loc16_32.11: init %struct_type.a.b.cbe to file.%y.var = struct_init (%.loc16_32.4, %.loc16_32.10)
// CHECK:STDOUT:   %.loc16_1: init %struct_type.a.b.cbe = converted %x.ref, %.loc16_32.11
// CHECK:STDOUT:   assign file.%y.var, %.loc16_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
