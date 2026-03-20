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
// CHECK:STDOUT:   %tuple.896: %tuple.type.85c = tuple_value (%i32) [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.3d5: type = struct_type {.a: %i32, .b: %tuple.type.a1c} [concrete]
// CHECK:STDOUT:   %pattern_type.b4d: type = pattern_type %struct_type.a.b.3d5 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.e7e: %tuple.type.985 = tuple_value (%int_2.ecc) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.057: type = struct_type {.a: Core.IntLiteral, .b: %tuple.type.985} [concrete]
// CHECK:STDOUT:   %struct.60a: %struct_type.a.b.057 = struct_value (%int_1.5b8, %tuple.e7e) [concrete]
// CHECK:STDOUT:   %.ae3: ref %i32 = struct_access file.%x.var, element0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %.03e: ref %tuple.type.a1c = struct_access file.%x.var, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem0.83d: ref %i32 = tuple_access %.03e, element0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.092: %tuple.type.a1c = tuple_value (%int_2.ef8) [concrete]
// CHECK:STDOUT:   %struct.646: %struct_type.a.b.3d5 = struct_value (%int_1.5d2, %tuple.092) [concrete]
// CHECK:STDOUT:   %.a20: ref %i32 = struct_access file.%y.var, element0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %.07f: ref %tuple.type.a1c = struct_access file.%y.var, element1 [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc80, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %x.patt: %pattern_type.b4d = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.b4d = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.3d5 = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc17: type = splice_block %struct_type.a.b.loc17 [concrete = constants.%struct_type.a.b.3d5] {
// CHECK:STDOUT:     %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc16.1: %tuple.type.85c = tuple_literal (%i32.loc14) [concrete = constants.%tuple.896]
// CHECK:STDOUT:     %.loc16.2: type = converted %.loc16.1, constants.%tuple.type.a1c [concrete = constants.%tuple.type.a1c]
// CHECK:STDOUT:     %struct_type.a.b.loc17: type = struct_type {.a: %i32, .b: %tuple.type.a1c} [concrete = constants.%struct_type.a.b.3d5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.a.b.3d5 = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.b4d = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.b4d = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %struct_type.a.b.3d5 = var %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc50: type = splice_block %struct_type.a.b.loc50 [concrete = constants.%struct_type.a.b.3d5] {
// CHECK:STDOUT:     %i32.loc41: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc47: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc49.1: %tuple.type.85c = tuple_literal (%i32.loc47) [concrete = constants.%tuple.896]
// CHECK:STDOUT:     %.loc49.2: type = converted %.loc49.1, constants.%tuple.type.a1c [concrete = constants.%tuple.type.a1c]
// CHECK:STDOUT:     %struct_type.a.b.loc50: type = struct_type {.a: %i32, .b: %tuple.type.a1c} [concrete = constants.%struct_type.a.b.3d5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %struct_type.a.b.3d5 = ref_binding y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc31.1: %tuple.type.985 = tuple_literal (%int_2) [concrete = constants.%tuple.e7e]
// CHECK:STDOUT:   %.loc32.1: %struct_type.a.b.057 = struct_literal (%int_1, %.loc31.1) [concrete = constants.%struct.60a]
// CHECK:STDOUT:   %impl.elem0.loc32: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc32.1: <bound method> = bound_method %int_1, %impl.elem0.loc32 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc32: <specific function> = specific_function %impl.elem0.loc32, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc32.2: <bound method> = bound_method %int_1, %specific_fn.loc32 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32: init %i32 = call %bound_method.loc32.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc32.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc32.3: ref %i32 = struct_access file.%x.var, element0 [concrete = constants.%.ae3]
// CHECK:STDOUT:   %.loc32.4: init %i32 to %.loc32.3 = in_place_init %.loc32.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc31: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc31.1: <bound method> = bound_method %int_2, %impl.elem0.loc31 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc31: <specific function> = specific_function %impl.elem0.loc31, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc31.2: <bound method> = bound_method %int_2, %specific_fn.loc31 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc31: init %i32 = call %bound_method.loc31.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc31.2: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc31 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc32.5: ref %tuple.type.a1c = struct_access file.%x.var, element1 [concrete = constants.%.03e]
// CHECK:STDOUT:   %.loc31.3: init %tuple.type.a1c to %.loc32.5 = tuple_init (%.loc31.2) [concrete = constants.%tuple.092]
// CHECK:STDOUT:   %.loc32.6: init %tuple.type.a1c = converted %.loc31.1, %.loc31.3 [concrete = constants.%tuple.092]
// CHECK:STDOUT:   %.loc32.7: init %tuple.type.a1c to %.loc32.5 = in_place_init %.loc32.6 [concrete = constants.%tuple.092]
// CHECK:STDOUT:   %.loc32.8: init %struct_type.a.b.3d5 to file.%x.var = struct_init (%.loc32.4, %.loc32.7) [concrete = constants.%struct.646]
// CHECK:STDOUT:   %.loc1: init %struct_type.a.b.3d5 = converted %.loc32.1, %.loc32.8 [concrete = constants.%struct.646]
// CHECK:STDOUT:   assign file.%x.var, %.loc1
// CHECK:STDOUT:   %x.ref: ref %struct_type.a.b.3d5 = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   %.loc52.1: ref %i32 = struct_access %x.ref, element0 [concrete = constants.%.ae3]
// CHECK:STDOUT:   %.loc52.2: %i32 = acquire_value %.loc52.1
// CHECK:STDOUT:   %impl.elem0.loc52.1: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc52.1: <bound method> = bound_method %.loc52.2, %impl.elem0.loc52.1
// CHECK:STDOUT:   %specific_fn.loc52.1: <specific function> = specific_function %impl.elem0.loc52.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc52.2: <bound method> = bound_method %.loc52.2, %specific_fn.loc52.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc52.1: init %i32 = call %bound_method.loc52.2(%.loc52.2)
// CHECK:STDOUT:   %.loc52.3: ref %i32 = struct_access file.%y.var, element0 [concrete = constants.%.a20]
// CHECK:STDOUT:   %.loc52.4: init %i32 to %.loc52.3 = in_place_init %Int.as.Copy.impl.Op.call.loc52.1
// CHECK:STDOUT:   %.loc52.5: ref %tuple.type.a1c = struct_access %x.ref, element1 [concrete = constants.%.03e]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %.loc52.5, element0 [concrete = constants.%tuple.elem0.83d]
// CHECK:STDOUT:   %.loc52.6: %i32 = acquire_value %tuple.elem0
// CHECK:STDOUT:   %impl.elem0.loc52.2: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc52.3: <bound method> = bound_method %.loc52.6, %impl.elem0.loc52.2
// CHECK:STDOUT:   %specific_fn.loc52.2: <specific function> = specific_function %impl.elem0.loc52.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc52.4: <bound method> = bound_method %.loc52.6, %specific_fn.loc52.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc52.2: init %i32 = call %bound_method.loc52.4(%.loc52.6)
// CHECK:STDOUT:   %.loc52.7: ref %tuple.type.a1c = struct_access file.%y.var, element1 [concrete = constants.%.07f]
// CHECK:STDOUT:   %.loc52.8: init %tuple.type.a1c to %.loc52.7 = tuple_init (%Int.as.Copy.impl.Op.call.loc52.2)
// CHECK:STDOUT:   %.loc52.9: init %tuple.type.a1c = converted %.loc52.5, %.loc52.8
// CHECK:STDOUT:   %.loc52.10: init %tuple.type.a1c to %.loc52.7 = in_place_init %.loc52.9
// CHECK:STDOUT:   %.loc52.11: init %struct_type.a.b.3d5 to file.%y.var = struct_init (%.loc52.4, %.loc52.10)
// CHECK:STDOUT:   %.loc34: init %struct_type.a.b.3d5 = converted %x.ref, %.loc52.11
// CHECK:STDOUT:   assign file.%y.var, %.loc34
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
