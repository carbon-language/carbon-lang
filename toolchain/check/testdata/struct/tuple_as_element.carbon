// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.3d5: type = struct_type {.a: %i32, .b: %tuple.type.a1c} [concrete]
// CHECK:STDOUT:   %pattern_type.b4d: type = pattern_type %struct_type.a.b.3d5 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.057: type = struct_type {.a: Core.IntLiteral, .b: %tuple.type.985} [concrete]
// CHECK:STDOUT:   %.b14: ref %i32 = struct_access file.%x.var, element0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %.dd9: ref %tuple.type.a1c = struct_access file.%x.var, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem0.2a6: ref %i32 = tuple_access %.dd9, element0 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.a1c = tuple_value (%int_2.ef8) [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.3d5 = struct_value (%int_1.5d2, %tuple) [concrete]
// CHECK:STDOUT:   %.f15: ref %i32 = struct_access file.%y.var, element0 [concrete]
// CHECK:STDOUT:   %.413: ref %tuple.type.a1c = struct_access file.%y.var, element1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
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
// CHECK:STDOUT:     %x.patt: %pattern_type.b4d = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.b4d = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.3d5 = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc14_28: type = splice_block %struct_type.a.b.loc14 [concrete = constants.%struct_type.a.b.3d5] {
// CHECK:STDOUT:     %int_32.loc14_13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc14_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc14_27.1: %tuple.type.85c = tuple_literal (%i32.loc14_23)
// CHECK:STDOUT:     %.loc14_27.2: type = converted %.loc14_27.1, constants.%tuple.type.a1c [concrete = constants.%tuple.type.a1c]
// CHECK:STDOUT:     %struct_type.a.b.loc14: type = struct_type {.a: %i32, .b: %tuple.type.a1c} [concrete = constants.%struct_type.a.b.3d5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.a.b.3d5 = bind_name x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.b4d = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.b4d = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %struct_type.a.b.3d5 = var %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc15_28: type = splice_block %struct_type.a.b.loc15 [concrete = constants.%struct_type.a.b.3d5] {
// CHECK:STDOUT:     %int_32.loc15_13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc15_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15_27.1: %tuple.type.85c = tuple_literal (%i32.loc15_23)
// CHECK:STDOUT:     %.loc15_27.2: type = converted %.loc15_27.1, constants.%tuple.type.a1c [concrete = constants.%tuple.type.a1c]
// CHECK:STDOUT:     %struct_type.a.b.loc15: type = struct_type {.a: %i32, .b: %tuple.type.a1c} [concrete = constants.%struct_type.a.b.3d5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %struct_type.a.b.3d5 = bind_name y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc14_49.1: %tuple.type.985 = tuple_literal (%int_2)
// CHECK:STDOUT:   %.loc14_50.1: %struct_type.a.b.057 = struct_literal (%int_1, %.loc14_49.1)
// CHECK:STDOUT:   %impl.elem0.loc14_50: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_50.1: <bound method> = bound_method %int_1, %impl.elem0.loc14_50 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc14_50: <specific function> = specific_function %impl.elem0.loc14_50, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_50.2: <bound method> = bound_method %int_1, %specific_fn.loc14_50 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc14_50: init %i32 = call %bound_method.loc14_50.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_50.2: init %i32 = converted %int_1, %int.convert_checked.loc14_50 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_50.3: ref %i32 = struct_access file.%x.var, element0 [concrete = constants.%.b14]
// CHECK:STDOUT:   %.loc14_50.4: init %i32 = initialize_from %.loc14_50.2 to %.loc14_50.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc14_49: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_49.1: <bound method> = bound_method %int_2, %impl.elem0.loc14_49 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc14_49: <specific function> = specific_function %impl.elem0.loc14_49, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_49.2: <bound method> = bound_method %int_2, %specific_fn.loc14_49 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc14_49: init %i32 = call %bound_method.loc14_49.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc14_49.2: init %i32 = converted %int_2, %int.convert_checked.loc14_49 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc14_50.5: ref %tuple.type.a1c = struct_access file.%x.var, element1 [concrete = constants.%.dd9]
// CHECK:STDOUT:   %.loc14_49.3: init %tuple.type.a1c = tuple_init (%.loc14_49.2) to %.loc14_50.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc14_50.6: init %tuple.type.a1c = converted %.loc14_49.1, %.loc14_49.3 [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc14_50.7: init %tuple.type.a1c = initialize_from %.loc14_50.6 to %.loc14_50.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc14_50.8: init %struct_type.a.b.3d5 = struct_init (%.loc14_50.4, %.loc14_50.7) to file.%x.var [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc14_1: init %struct_type.a.b.3d5 = converted %.loc14_50.1, %.loc14_50.8 [concrete = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc14_1
// CHECK:STDOUT:   %x.ref: ref %struct_type.a.b.3d5 = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   %.loc15_32.1: ref %i32 = struct_access %x.ref, element0 [concrete = constants.%.b14]
// CHECK:STDOUT:   %.loc15_32.2: %i32 = bind_value %.loc15_32.1
// CHECK:STDOUT:   %.loc15_32.3: ref %i32 = struct_access file.%y.var, element0 [concrete = constants.%.f15]
// CHECK:STDOUT:   %.loc15_32.4: init %i32 = initialize_from %.loc15_32.2 to %.loc15_32.3
// CHECK:STDOUT:   %.loc15_32.5: ref %tuple.type.a1c = struct_access %x.ref, element1 [concrete = constants.%.dd9]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %.loc15_32.5, element0 [concrete = constants.%tuple.elem0.2a6]
// CHECK:STDOUT:   %.loc15_32.6: %i32 = bind_value %tuple.elem0
// CHECK:STDOUT:   %.loc15_32.7: ref %tuple.type.a1c = struct_access file.%y.var, element1 [concrete = constants.%.413]
// CHECK:STDOUT:   %.loc15_32.8: init %tuple.type.a1c = tuple_init (%.loc15_32.6) to %.loc15_32.7
// CHECK:STDOUT:   %.loc15_32.9: init %tuple.type.a1c = converted %.loc15_32.5, %.loc15_32.8
// CHECK:STDOUT:   %.loc15_32.10: init %tuple.type.a1c = initialize_from %.loc15_32.9 to %.loc15_32.7
// CHECK:STDOUT:   %.loc15_32.11: init %struct_type.a.b.3d5 = struct_init (%.loc15_32.4, %.loc15_32.10) to file.%y.var
// CHECK:STDOUT:   %.loc15_1: init %struct_type.a.b.3d5 = converted %x.ref, %.loc15_32.11
// CHECK:STDOUT:   assign file.%y.var, %.loc15_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
