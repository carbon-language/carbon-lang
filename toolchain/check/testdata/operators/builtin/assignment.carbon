// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/assignment.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/assignment.carbon

fn Main() {
  var a: i32 = 12;
  a = 9;

  var b: (i32, i32) = (1, 2);
  b.0 = 3;
  b.1 = 4;

  var c: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  c.a = 3;
  c.b = 4;

  var p: i32* = &a;
  *p = 5;

  *(if true then p else &a) = 10;
}

// CHECK:STDOUT: --- assignment.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_12.6a3: Core.IntLiteral = int_value 12 [concrete]
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
// CHECK:STDOUT:   %Convert.bound.221: <bound method> = bound_method %int_12.6a3, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.dae: <bound method> = bound_method %int_12.6a3, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_12.1e1: %i32 = int_value 12 [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %Convert.bound.9e2: <bound method> = bound_method %int_9.988, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.cd3: <bound method> = bound_method %int_9.988, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.f88: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.511: type = pattern_type %tuple.type.d07 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.1da: <bound method> = bound_method %int_4.0c1, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.501: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.851: type = pattern_type %struct_type.a.b.501 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.501 = struct_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %ptr.235 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Convert.bound.4e6: <bound method> = bound_method %int_5.64b, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.a25: <bound method> = bound_method %int_5.64b, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %int_10.64f: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %Convert.bound.491: <bound method> = bound_method %int_10.64f, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.5ba: <bound method> = bound_method %int_10.64f, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_10.265: %i32 = int_value 10 [concrete]
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
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.7ce = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %i32 = var %a.var_patt
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete = constants.%int_12.6a3]
// CHECK:STDOUT:   %impl.elem0.loc15: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc15_3.1: <bound method> = bound_method %int_12, %impl.elem0.loc15 [concrete = constants.%Convert.bound.221]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_3.2: <bound method> = bound_method %int_12, %specific_fn.loc15 [concrete = constants.%bound_method.dae]
// CHECK:STDOUT:   %int.convert_checked.loc15: init %i32 = call %bound_method.loc15_3.2(%int_12) [concrete = constants.%int_12.1e1]
// CHECK:STDOUT:   %.loc15_3: init %i32 = converted %int_12, %int.convert_checked.loc15 [concrete = constants.%int_12.1e1]
// CHECK:STDOUT:   assign %a.var, %.loc15_3
// CHECK:STDOUT:   %.loc15_10: type = splice_block %i32.loc15 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %i32 = bind_name a, %a.var
// CHECK:STDOUT:   %a.ref.loc16: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %impl.elem0.loc16: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc16_5.1: <bound method> = bound_method %int_9, %impl.elem0.loc16 [concrete = constants.%Convert.bound.9e2]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_5.2: <bound method> = bound_method %int_9, %specific_fn.loc16 [concrete = constants.%bound_method.cd3]
// CHECK:STDOUT:   %int.convert_checked.loc16: init %i32 = call %bound_method.loc16_5.2(%int_9) [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc16: init %i32 = converted %int_9, %int.convert_checked.loc16 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   assign %a.ref.loc16, %.loc16
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.511 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.511 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %tuple.type.d07 = var %b.var_patt
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc18: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc18_28.1: %tuple.type.f94 = tuple_literal (%int_1.loc18, %int_2.loc18)
// CHECK:STDOUT:   %impl.elem0.loc18_28.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc18_28.1: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_28.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc18_28.1: <specific function> = specific_function %impl.elem0.loc18_28.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_28.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18_28.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc18_28.1: init %i32 = call %bound_method.loc18_28.2(%int_1.loc18) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_28.2: init %i32 = converted %int_1.loc18, %int.convert_checked.loc18_28.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem0.loc18: ref %i32 = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc18_28.3: init %i32 = initialize_from %.loc18_28.2 to %tuple.elem0.loc18 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc18_28.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc18_28.3: <bound method> = bound_method %int_2.loc18, %impl.elem0.loc18_28.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc18_28.2: <specific function> = specific_function %impl.elem0.loc18_28.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_28.4: <bound method> = bound_method %int_2.loc18, %specific_fn.loc18_28.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc18_28.2: init %i32 = call %bound_method.loc18_28.4(%int_2.loc18) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc18_28.4: init %i32 = converted %int_2.loc18, %int.convert_checked.loc18_28.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.elem1.loc18: ref %i32 = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc18_28.5: init %i32 = initialize_from %.loc18_28.4 to %tuple.elem1.loc18 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc18_28.6: init %tuple.type.d07 = tuple_init (%.loc18_28.3, %.loc18_28.5) to %b.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc18_3: init %tuple.type.d07 = converted %.loc18_28.1, %.loc18_28.6 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %b.var, %.loc18_3
// CHECK:STDOUT:   %.loc18_19.1: type = splice_block %.loc18_19.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:     %int_32.loc18_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc18_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18_19.2: %tuple.type.24b = tuple_literal (%i32.loc18_11, %i32.loc18_16)
// CHECK:STDOUT:     %.loc18_19.3: type = converted %.loc18_19.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %tuple.type.d07 = bind_name b, %b.var
// CHECK:STDOUT:   %b.ref.loc19: ref %tuple.type.d07 = name_ref b, %b
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0.loc19: ref %i32 = tuple_access %b.ref.loc19, element0
// CHECK:STDOUT:   %int_3.loc19: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc19: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc19_7.1: <bound method> = bound_method %int_3.loc19, %impl.elem0.loc19 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc19: <specific function> = specific_function %impl.elem0.loc19, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19_7.2: <bound method> = bound_method %int_3.loc19, %specific_fn.loc19 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc19: init %i32 = call %bound_method.loc19_7.2(%int_3.loc19) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc19: init %i32 = converted %int_3.loc19, %int.convert_checked.loc19 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   assign %tuple.elem0.loc19, %.loc19
// CHECK:STDOUT:   %b.ref.loc20: ref %tuple.type.d07 = name_ref b, %b
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc20: ref %i32 = tuple_access %b.ref.loc20, element1
// CHECK:STDOUT:   %int_4.loc20: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc20: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc20_7.1: <bound method> = bound_method %int_4.loc20, %impl.elem0.loc20 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_7.2: <bound method> = bound_method %int_4.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc20: init %i32 = call %bound_method.loc20_7.2(%int_4.loc20) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc20: init %i32 = converted %int_4.loc20, %int.convert_checked.loc20 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %tuple.elem1.loc20, %.loc20
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.851 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.851 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %struct_type.a.b.501 = var %c.var_patt
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc22: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc22_46.1: %struct_type.a.b.cfd = struct_literal (%int_1.loc22, %int_2.loc22)
// CHECK:STDOUT:   %impl.elem0.loc22_46.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc22_46.1: <bound method> = bound_method %int_1.loc22, %impl.elem0.loc22_46.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc22_46.1: <specific function> = specific_function %impl.elem0.loc22_46.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_46.2: <bound method> = bound_method %int_1.loc22, %specific_fn.loc22_46.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc22_46.1: init %i32 = call %bound_method.loc22_46.2(%int_1.loc22) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc22_46.2: init %i32 = converted %int_1.loc22, %int.convert_checked.loc22_46.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc22_46.3: ref %i32 = struct_access %c.var, element0
// CHECK:STDOUT:   %.loc22_46.4: init %i32 = initialize_from %.loc22_46.2 to %.loc22_46.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc22_46.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc22_46.3: <bound method> = bound_method %int_2.loc22, %impl.elem0.loc22_46.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc22_46.2: <specific function> = specific_function %impl.elem0.loc22_46.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_46.4: <bound method> = bound_method %int_2.loc22, %specific_fn.loc22_46.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc22_46.2: init %i32 = call %bound_method.loc22_46.4(%int_2.loc22) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc22_46.5: init %i32 = converted %int_2.loc22, %int.convert_checked.loc22_46.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc22_46.6: ref %i32 = struct_access %c.var, element1
// CHECK:STDOUT:   %.loc22_46.7: init %i32 = initialize_from %.loc22_46.5 to %.loc22_46.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc22_46.8: init %struct_type.a.b.501 = struct_init (%.loc22_46.4, %.loc22_46.7) to %c.var [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc22_3: init %struct_type.a.b.501 = converted %.loc22_46.1, %.loc22_46.8 [concrete = constants.%struct]
// CHECK:STDOUT:   assign %c.var, %.loc22_3
// CHECK:STDOUT:   %.loc22_27: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b.501] {
// CHECK:STDOUT:     %int_32.loc22_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc22_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.501]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %struct_type.a.b.501 = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref.loc23: ref %struct_type.a.b.501 = name_ref c, %c
// CHECK:STDOUT:   %.loc23_4: ref %i32 = struct_access %c.ref.loc23, element0
// CHECK:STDOUT:   %int_3.loc23: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc23: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc23_7.1: <bound method> = bound_method %int_3.loc23, %impl.elem0.loc23 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_7.2: <bound method> = bound_method %int_3.loc23, %specific_fn.loc23 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc23: init %i32 = call %bound_method.loc23_7.2(%int_3.loc23) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc23_7: init %i32 = converted %int_3.loc23, %int.convert_checked.loc23 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   assign %.loc23_4, %.loc23_7
// CHECK:STDOUT:   %c.ref.loc24: ref %struct_type.a.b.501 = name_ref c, %c
// CHECK:STDOUT:   %.loc24_4: ref %i32 = struct_access %c.ref.loc24, element1
// CHECK:STDOUT:   %int_4.loc24: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc24: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc24_7.1: <bound method> = bound_method %int_4.loc24, %impl.elem0.loc24 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_7.2: <bound method> = bound_method %int_4.loc24, %specific_fn.loc24 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc24: init %i32 = call %bound_method.loc24_7.2(%int_4.loc24) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc24_7: init %i32 = converted %int_4.loc24, %int.convert_checked.loc24 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc24_4, %.loc24_7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.fe8 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.fe8 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.235 = var %p.var_patt
// CHECK:STDOUT:   %a.ref.loc26: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %addr.loc26: %ptr.235 = addr_of %a.ref.loc26
// CHECK:STDOUT:   assign %p.var, %addr.loc26
// CHECK:STDOUT:   %.loc26: type = splice_block %ptr [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc26: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc26: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32.loc26 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.235 = bind_name p, %p.var
// CHECK:STDOUT:   %p.ref.loc27: ref %ptr.235 = name_ref p, %p
// CHECK:STDOUT:   %.loc27_4: %ptr.235 = bind_value %p.ref.loc27
// CHECK:STDOUT:   %.loc27_3: ref %i32 = deref %.loc27_4
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0.loc27: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc27_6.1: <bound method> = bound_method %int_5, %impl.elem0.loc27 [concrete = constants.%Convert.bound.4e6]
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_6.2: <bound method> = bound_method %int_5, %specific_fn.loc27 [concrete = constants.%bound_method.a25]
// CHECK:STDOUT:   %int.convert_checked.loc27: init %i32 = call %bound_method.loc27_6.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc27_6: init %i32 = converted %int_5, %int.convert_checked.loc27 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   assign %.loc27_3, %.loc27_6
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %p.ref.loc29: ref %ptr.235 = name_ref p, %p
// CHECK:STDOUT:   %.loc29_18: %ptr.235 = bind_value %p.ref.loc29
// CHECK:STDOUT:   br !if.expr.result(%.loc29_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %a.ref.loc29: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %addr.loc29: %ptr.235 = addr_of %a.ref.loc29
// CHECK:STDOUT:   br !if.expr.result(%addr.loc29)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc29_5: %ptr.235 = block_arg !if.expr.result
// CHECK:STDOUT:   %.loc29_3: ref %i32 = deref %.loc29_5
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10.64f]
// CHECK:STDOUT:   %impl.elem0.loc29: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc29_29.1: <bound method> = bound_method %int_10, %impl.elem0.loc29 [concrete = constants.%Convert.bound.491]
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem0.loc29, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc29_29.2: <bound method> = bound_method %int_10, %specific_fn.loc29 [concrete = constants.%bound_method.5ba]
// CHECK:STDOUT:   %int.convert_checked.loc29: init %i32 = call %bound_method.loc29_29.2(%int_10) [concrete = constants.%int_10.265]
// CHECK:STDOUT:   %.loc29_29: init %i32 = converted %int_10, %int.convert_checked.loc29 [concrete = constants.%int_10.265]
// CHECK:STDOUT:   assign %.loc29_3, %.loc29_29
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
