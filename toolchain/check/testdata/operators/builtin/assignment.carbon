// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_12.6a3: Core.IntLiteral = int_value 12 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
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
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.501 = struct_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
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
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %.loc12_3.1: %i32 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %i32 = var a
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete = constants.%int_12.6a3]
// CHECK:STDOUT:   %impl.elem0.loc12: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc12_3.1: <bound method> = bound_method %int_12, %impl.elem0.loc12 [concrete = constants.%Convert.bound.221]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_3.2: <bound method> = bound_method %int_12, %specific_fn.loc12 [concrete = constants.%bound_method.dae]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %bound_method.loc12_3.2(%int_12) [concrete = constants.%int_12.1e1]
// CHECK:STDOUT:   %.loc12_3.2: init %i32 = converted %int_12, %int.convert_checked.loc12 [concrete = constants.%int_12.1e1]
// CHECK:STDOUT:   assign %a.var, %.loc12_3.2
// CHECK:STDOUT:   %.loc12_10: type = splice_block %i32.loc12 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %i32 = bind_name a, %a.var
// CHECK:STDOUT:   %a.ref.loc13: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %impl.elem0.loc13: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_5.1: <bound method> = bound_method %int_9, %impl.elem0.loc13 [concrete = constants.%Convert.bound.9e2]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_5.2: <bound method> = bound_method %int_9, %specific_fn.loc13 [concrete = constants.%bound_method.cd3]
// CHECK:STDOUT:   %int.convert_checked.loc13: init %i32 = call %bound_method.loc13_5.2(%int_9) [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc13: init %i32 = converted %int_9, %int.convert_checked.loc13 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   assign %a.ref.loc13, %.loc13
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %tuple.type.d07 = binding_pattern b
// CHECK:STDOUT:     %.loc15_3.1: %tuple.type.d07 = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %tuple.type.d07 = var b
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc15: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc15_28.1: %tuple.type.f94 = tuple_literal (%int_1.loc15, %int_2.loc15)
// CHECK:STDOUT:   %impl.elem0.loc15_28.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc15_28.1: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15_28.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc15_28.1: <specific function> = specific_function %impl.elem0.loc15_28.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_28.2: <bound method> = bound_method %int_1.loc15, %specific_fn.loc15_28.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc15_28.1: init %i32 = call %bound_method.loc15_28.2(%int_1.loc15) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc15_28.2: init %i32 = converted %int_1.loc15, %int.convert_checked.loc15_28.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem0.loc15: ref %i32 = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc15_28.3: init %i32 = initialize_from %.loc15_28.2 to %tuple.elem0.loc15 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc15_28.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc15_28.3: <bound method> = bound_method %int_2.loc15, %impl.elem0.loc15_28.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc15_28.2: <specific function> = specific_function %impl.elem0.loc15_28.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_28.4: <bound method> = bound_method %int_2.loc15, %specific_fn.loc15_28.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc15_28.2: init %i32 = call %bound_method.loc15_28.4(%int_2.loc15) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc15_28.4: init %i32 = converted %int_2.loc15, %int.convert_checked.loc15_28.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.elem1.loc15: ref %i32 = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc15_28.5: init %i32 = initialize_from %.loc15_28.4 to %tuple.elem1.loc15 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc15_28.6: init %tuple.type.d07 = tuple_init (%.loc15_28.3, %.loc15_28.5) to %b.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc15_3.2: init %tuple.type.d07 = converted %.loc15_28.1, %.loc15_28.6 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %b.var, %.loc15_3.2
// CHECK:STDOUT:   %.loc15_19.1: type = splice_block %.loc15_19.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:     %int_32.loc15_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc15_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15_19.2: %tuple.type.24b = tuple_literal (%i32.loc15_11, %i32.loc15_16)
// CHECK:STDOUT:     %.loc15_19.3: type = converted %.loc15_19.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %tuple.type.d07 = bind_name b, %b.var
// CHECK:STDOUT:   %b.ref.loc16: ref %tuple.type.d07 = name_ref b, %b
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0.loc16: ref %i32 = tuple_access %b.ref.loc16, element0
// CHECK:STDOUT:   %int_3.loc16: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc16: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc16_7.1: <bound method> = bound_method %int_3.loc16, %impl.elem0.loc16 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_7.2: <bound method> = bound_method %int_3.loc16, %specific_fn.loc16 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc16: init %i32 = call %bound_method.loc16_7.2(%int_3.loc16) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc16: init %i32 = converted %int_3.loc16, %int.convert_checked.loc16 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   assign %tuple.elem0.loc16, %.loc16
// CHECK:STDOUT:   %b.ref.loc17: ref %tuple.type.d07 = name_ref b, %b
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc17: ref %i32 = tuple_access %b.ref.loc17, element1
// CHECK:STDOUT:   %int_4.loc17: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc17: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc17_7.1: <bound method> = bound_method %int_4.loc17, %impl.elem0.loc17 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_7.2: <bound method> = bound_method %int_4.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc17: init %i32 = call %bound_method.loc17_7.2(%int_4.loc17) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc17: init %i32 = converted %int_4.loc17, %int.convert_checked.loc17 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %tuple.elem1.loc17, %.loc17
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %struct_type.a.b.501 = binding_pattern c
// CHECK:STDOUT:     %.loc19_3.1: %struct_type.a.b.501 = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %struct_type.a.b.501 = var c
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc19: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc19_46.1: %struct_type.a.b.cfd = struct_literal (%int_1.loc19, %int_2.loc19)
// CHECK:STDOUT:   %impl.elem0.loc19_46.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc19_46.1: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_46.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc19_46.1: <specific function> = specific_function %impl.elem0.loc19_46.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19_46.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19_46.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc19_46.1: init %i32 = call %bound_method.loc19_46.2(%int_1.loc19) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc19_46.2: init %i32 = converted %int_1.loc19, %int.convert_checked.loc19_46.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc19_46.3: ref %i32 = struct_access %c.var, element0
// CHECK:STDOUT:   %.loc19_46.4: init %i32 = initialize_from %.loc19_46.2 to %.loc19_46.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc19_46.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc19_46.3: <bound method> = bound_method %int_2.loc19, %impl.elem0.loc19_46.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc19_46.2: <specific function> = specific_function %impl.elem0.loc19_46.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19_46.4: <bound method> = bound_method %int_2.loc19, %specific_fn.loc19_46.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc19_46.2: init %i32 = call %bound_method.loc19_46.4(%int_2.loc19) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc19_46.5: init %i32 = converted %int_2.loc19, %int.convert_checked.loc19_46.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc19_46.6: ref %i32 = struct_access %c.var, element1
// CHECK:STDOUT:   %.loc19_46.7: init %i32 = initialize_from %.loc19_46.5 to %.loc19_46.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc19_46.8: init %struct_type.a.b.501 = struct_init (%.loc19_46.4, %.loc19_46.7) to %c.var [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc19_3.2: init %struct_type.a.b.501 = converted %.loc19_46.1, %.loc19_46.8 [concrete = constants.%struct]
// CHECK:STDOUT:   assign %c.var, %.loc19_3.2
// CHECK:STDOUT:   %.loc19_27: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b.501] {
// CHECK:STDOUT:     %int_32.loc19_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc19_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc19_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc19_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.501]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %struct_type.a.b.501 = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref.loc20: ref %struct_type.a.b.501 = name_ref c, %c
// CHECK:STDOUT:   %.loc20_4: ref %i32 = struct_access %c.ref.loc20, element0
// CHECK:STDOUT:   %int_3.loc20: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc20: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc20_7.1: <bound method> = bound_method %int_3.loc20, %impl.elem0.loc20 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_7.2: <bound method> = bound_method %int_3.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc20: init %i32 = call %bound_method.loc20_7.2(%int_3.loc20) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc20_7: init %i32 = converted %int_3.loc20, %int.convert_checked.loc20 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   assign %.loc20_4, %.loc20_7
// CHECK:STDOUT:   %c.ref.loc21: ref %struct_type.a.b.501 = name_ref c, %c
// CHECK:STDOUT:   %.loc21_4: ref %i32 = struct_access %c.ref.loc21, element1
// CHECK:STDOUT:   %int_4.loc21: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc21: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc21_7.1: <bound method> = bound_method %int_4.loc21, %impl.elem0.loc21 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_7.2: <bound method> = bound_method %int_4.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc21: init %i32 = call %bound_method.loc21_7.2(%int_4.loc21) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc21_7: init %i32 = converted %int_4.loc21, %int.convert_checked.loc21 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc21_4, %.loc21_7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %ptr.235 = binding_pattern p
// CHECK:STDOUT:     %.loc23_3: %ptr.235 = var_pattern %p.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.235 = var p
// CHECK:STDOUT:   %a.ref.loc23: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %addr.loc23: %ptr.235 = addr_of %a.ref.loc23
// CHECK:STDOUT:   assign %p.var, %addr.loc23
// CHECK:STDOUT:   %.loc23_13: type = splice_block %ptr [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.235 = bind_name p, %p.var
// CHECK:STDOUT:   %p.ref.loc24: ref %ptr.235 = name_ref p, %p
// CHECK:STDOUT:   %.loc24_4: %ptr.235 = bind_value %p.ref.loc24
// CHECK:STDOUT:   %.loc24_3: ref %i32 = deref %.loc24_4
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0.loc24: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc24_6.1: <bound method> = bound_method %int_5, %impl.elem0.loc24 [concrete = constants.%Convert.bound.4e6]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_6.2: <bound method> = bound_method %int_5, %specific_fn.loc24 [concrete = constants.%bound_method.a25]
// CHECK:STDOUT:   %int.convert_checked.loc24: init %i32 = call %bound_method.loc24_6.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc24_6: init %i32 = converted %int_5, %int.convert_checked.loc24 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   assign %.loc24_3, %.loc24_6
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %p.ref.loc26: ref %ptr.235 = name_ref p, %p
// CHECK:STDOUT:   %.loc26_18: %ptr.235 = bind_value %p.ref.loc26
// CHECK:STDOUT:   br !if.expr.result(%.loc26_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %a.ref.loc26: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %addr.loc26: %ptr.235 = addr_of %a.ref.loc26
// CHECK:STDOUT:   br !if.expr.result(%addr.loc26)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc26_5: %ptr.235 = block_arg !if.expr.result
// CHECK:STDOUT:   %.loc26_3: ref %i32 = deref %.loc26_5
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10.64f]
// CHECK:STDOUT:   %impl.elem0.loc26: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc26_29.1: <bound method> = bound_method %int_10, %impl.elem0.loc26 [concrete = constants.%Convert.bound.491]
// CHECK:STDOUT:   %specific_fn.loc26: <specific function> = specific_function %impl.elem0.loc26, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_29.2: <bound method> = bound_method %int_10, %specific_fn.loc26 [concrete = constants.%bound_method.5ba]
// CHECK:STDOUT:   %int.convert_checked.loc26: init %i32 = call %bound_method.loc26_29.2(%int_10) [concrete = constants.%int_10.265]
// CHECK:STDOUT:   %.loc26_29: init %i32 = converted %int_10, %int.convert_checked.loc26 [concrete = constants.%int_10.265]
// CHECK:STDOUT:   assign %.loc26_3, %.loc26_29
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
