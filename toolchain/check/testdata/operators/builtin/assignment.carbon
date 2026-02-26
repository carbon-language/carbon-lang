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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_12.6a3: Core.IntLiteral = int_value 12 [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.230: <bound method> = bound_method %int_12.6a3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.906: <bound method> = bound_method %int_12.6a3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_12.1e1: %i32 = int_value 12 [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87d: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.661: <bound method> = bound_method %int_9.988, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.f88: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.95a: %tuple.type.24b = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.511: type = pattern_type %tuple.type.d07 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.21c: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.6d7: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.501: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.851: type = pattern_type %struct_type.a.b.501 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.4aa: %struct_type.a.b.cfd = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %struct.ed5: %struct_type.a.b.501 = struct_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %ptr.235 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.843: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.c3c: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.011: %ptr.as.Copy.impl.Op.type.c3c = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.235, (%Copy.impl_witness.843) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e01: type = fn_type @Copy.WithSelf.Op, @Copy(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a62: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e01, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.011, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.e9d: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %int_10.64f: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2bf: <bound method> = bound_method %int_10.64f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.409: <bound method> = bound_method %int_10.64f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_10.265: %i32 = int_value 10 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.1: type = fn_type @DestroyOp.loc27 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.1: %DestroyOp.type.3e79c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.2: type = fn_type @DestroyOp.loc23 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.2: %DestroyOp.type.3e79c2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.3: type = fn_type @DestroyOp.loc19 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.3: %DestroyOp.type.3e79c2.3 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.4: type = fn_type @DestroyOp.loc16 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.4: %DestroyOp.type.3e79c2.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.7ce = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %i32 = var %a.var_patt
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete = constants.%int_12.6a3]
// CHECK:STDOUT:   %impl.elem0.loc16: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc16_3.1: <bound method> = bound_method %int_12, %impl.elem0.loc16 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.230]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_3.2: <bound method> = bound_method %int_12, %specific_fn.loc16 [concrete = constants.%bound_method.906]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16: init %i32 = call %bound_method.loc16_3.2(%int_12) [concrete = constants.%int_12.1e1]
// CHECK:STDOUT:   %.loc16: init %i32 = converted %int_12, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16 [concrete = constants.%int_12.1e1]
// CHECK:STDOUT:   assign %a.var, %.loc16
// CHECK:STDOUT:   %i32.loc16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: ref %i32 = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc17: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %impl.elem0.loc17: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc17_5.1: <bound method> = bound_method %int_9, %impl.elem0.loc17 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87d]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_5.2: <bound method> = bound_method %int_9, %specific_fn.loc17 [concrete = constants.%bound_method.661]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %i32 = call %bound_method.loc17_5.2(%int_9) [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc17: init %i32 = converted %int_9, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   assign %a.ref.loc17, %.loc17
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.511 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.511 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %tuple.type.d07 = var %b.var_patt
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc19: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc19_28.1: %tuple.type.f94 = tuple_literal (%int_1.loc19, %int_2.loc19) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %impl.elem0.loc19_28.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc19_28.1: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_28.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc19_28.1: <specific function> = specific_function %impl.elem0.loc19_28.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19_28.2: <bound method> = bound_method %int_1.loc19, %specific_fn.loc19_28.1 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_28.1: init %i32 = call %bound_method.loc19_28.2(%int_1.loc19) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc19_28.2: init %i32 = converted %int_1.loc19, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_28.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem0.loc19: ref %i32 = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc19_28.3: init %i32 to %tuple.elem0.loc19 = in_place_init %.loc19_28.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc19_28.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc19_28.3: <bound method> = bound_method %int_2.loc19, %impl.elem0.loc19_28.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc19_28.2: <specific function> = specific_function %impl.elem0.loc19_28.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19_28.4: <bound method> = bound_method %int_2.loc19, %specific_fn.loc19_28.2 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_28.2: init %i32 = call %bound_method.loc19_28.4(%int_2.loc19) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc19_28.4: init %i32 = converted %int_2.loc19, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc19_28.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.elem1.loc19: ref %i32 = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc19_28.5: init %i32 to %tuple.elem1.loc19 = in_place_init %.loc19_28.4 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc19_28.6: init %tuple.type.d07 to %b.var = tuple_init (%.loc19_28.3, %.loc19_28.5) [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %.loc19_3: init %tuple.type.d07 = converted %.loc19_28.1, %.loc19_28.6 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   assign %b.var, %.loc19_3
// CHECK:STDOUT:   %.loc19_19.1: type = splice_block %.loc19_19.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:     %i32.loc19_11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc19_16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc19_19.2: %tuple.type.24b = tuple_literal (%i32.loc19_11, %i32.loc19_16) [concrete = constants.%tuple.95a]
// CHECK:STDOUT:     %.loc19_19.3: type = converted %.loc19_19.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %tuple.type.d07 = ref_binding b, %b.var
// CHECK:STDOUT:   %b.ref.loc20: ref %tuple.type.d07 = name_ref b, %b
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0.loc20: ref %i32 = tuple_access %b.ref.loc20, element0
// CHECK:STDOUT:   %int_3.loc20: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc20: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc20_7.1: <bound method> = bound_method %int_3.loc20, %impl.elem0.loc20 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_7.2: <bound method> = bound_method %int_3.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %i32 = call %bound_method.loc20_7.2(%int_3.loc20) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc20: init %i32 = converted %int_3.loc20, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   assign %tuple.elem0.loc20, %.loc20
// CHECK:STDOUT:   %b.ref.loc21: ref %tuple.type.d07 = name_ref b, %b
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc21: ref %i32 = tuple_access %b.ref.loc21, element1
// CHECK:STDOUT:   %int_4.loc21: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc21: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc21_7.1: <bound method> = bound_method %int_4.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_7.2: <bound method> = bound_method %int_4.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.6d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %i32 = call %bound_method.loc21_7.2(%int_4.loc21) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc21: init %i32 = converted %int_4.loc21, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %tuple.elem1.loc21, %.loc21
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.851 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.851 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %struct_type.a.b.501 = var %c.var_patt
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc23: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc23_46.1: %struct_type.a.b.cfd = struct_literal (%int_1.loc23, %int_2.loc23) [concrete = constants.%struct.4aa]
// CHECK:STDOUT:   %impl.elem0.loc23_46.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc23_46.1: <bound method> = bound_method %int_1.loc23, %impl.elem0.loc23_46.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc23_46.1: <specific function> = specific_function %impl.elem0.loc23_46.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_46.2: <bound method> = bound_method %int_1.loc23, %specific_fn.loc23_46.1 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23_46.1: init %i32 = call %bound_method.loc23_46.2(%int_1.loc23) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc23_46.2: init %i32 = converted %int_1.loc23, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23_46.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc23_46.3: ref %i32 = struct_access %c.var, element0
// CHECK:STDOUT:   %.loc23_46.4: init %i32 to %.loc23_46.3 = in_place_init %.loc23_46.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc23_46.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc23_46.3: <bound method> = bound_method %int_2.loc23, %impl.elem0.loc23_46.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc23_46.2: <specific function> = specific_function %impl.elem0.loc23_46.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_46.4: <bound method> = bound_method %int_2.loc23, %specific_fn.loc23_46.2 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23_46.2: init %i32 = call %bound_method.loc23_46.4(%int_2.loc23) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc23_46.5: init %i32 = converted %int_2.loc23, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23_46.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc23_46.6: ref %i32 = struct_access %c.var, element1
// CHECK:STDOUT:   %.loc23_46.7: init %i32 to %.loc23_46.6 = in_place_init %.loc23_46.5 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc23_46.8: init %struct_type.a.b.501 to %c.var = struct_init (%.loc23_46.4, %.loc23_46.7) [concrete = constants.%struct.ed5]
// CHECK:STDOUT:   %.loc23_3: init %struct_type.a.b.501 = converted %.loc23_46.1, %.loc23_46.8 [concrete = constants.%struct.ed5]
// CHECK:STDOUT:   assign %c.var, %.loc23_3
// CHECK:STDOUT:   %.loc23_27: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b.501] {
// CHECK:STDOUT:     %i32.loc23_15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc23_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.501]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %struct_type.a.b.501 = ref_binding c, %c.var
// CHECK:STDOUT:   %c.ref.loc24: ref %struct_type.a.b.501 = name_ref c, %c
// CHECK:STDOUT:   %.loc24_4: ref %i32 = struct_access %c.ref.loc24, element0
// CHECK:STDOUT:   %int_3.loc24: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc24: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc24_7.1: <bound method> = bound_method %int_3.loc24, %impl.elem0.loc24 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_7.2: <bound method> = bound_method %int_3.loc24, %specific_fn.loc24 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24: init %i32 = call %bound_method.loc24_7.2(%int_3.loc24) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc24_7: init %i32 = converted %int_3.loc24, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   assign %.loc24_4, %.loc24_7
// CHECK:STDOUT:   %c.ref.loc25: ref %struct_type.a.b.501 = name_ref c, %c
// CHECK:STDOUT:   %.loc25_4: ref %i32 = struct_access %c.ref.loc25, element1
// CHECK:STDOUT:   %int_4.loc25: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc25: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc25_7.1: <bound method> = bound_method %int_4.loc25, %impl.elem0.loc25 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c]
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_7.2: <bound method> = bound_method %int_4.loc25, %specific_fn.loc25 [concrete = constants.%bound_method.6d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25: init %i32 = call %bound_method.loc25_7.2(%int_4.loc25) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc25_7: init %i32 = converted %int_4.loc25, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc25_4, %.loc25_7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.fe8 = ref_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.fe8 = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.235 = var %p.var_patt
// CHECK:STDOUT:   %a.ref.loc27: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %addr.loc27: %ptr.235 = addr_of %a.ref.loc27
// CHECK:STDOUT:   %impl.elem0.loc27: %.a62 = impl_witness_access constants.%Copy.impl_witness.843, element0 [concrete = constants.%ptr.as.Copy.impl.Op.011]
// CHECK:STDOUT:   %bound_method.loc27_17.1: <bound method> = bound_method %addr.loc27, %impl.elem0.loc27
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_17.2: <bound method> = bound_method %addr.loc27, %specific_fn.loc27
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.235 = call %bound_method.loc27_17.2(%addr.loc27)
// CHECK:STDOUT:   assign %p.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc27: type = splice_block %ptr [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %i32.loc27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32.loc27 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.235 = ref_binding p, %p.var
// CHECK:STDOUT:   %p.ref.loc28: ref %ptr.235 = name_ref p, %p
// CHECK:STDOUT:   %.loc28_4: %ptr.235 = acquire_value %p.ref.loc28
// CHECK:STDOUT:   %.loc28_3: ref %i32 = deref %.loc28_4
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0.loc28: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc28_6.1: <bound method> = bound_method %int_5, %impl.elem0.loc28 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005]
// CHECK:STDOUT:   %specific_fn.loc28: <specific function> = specific_function %impl.elem0.loc28, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc28_6.2: <bound method> = bound_method %int_5, %specific_fn.loc28 [concrete = constants.%bound_method.e9d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28: init %i32 = call %bound_method.loc28_6.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc28_6: init %i32 = converted %int_5, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   assign %.loc28_3, %.loc28_6
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %p.ref.loc30: ref %ptr.235 = name_ref p, %p
// CHECK:STDOUT:   %.loc30_18: %ptr.235 = acquire_value %p.ref.loc30
// CHECK:STDOUT:   br !if.expr.result(%.loc30_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %a.ref.loc30: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %addr.loc30: %ptr.235 = addr_of %a.ref.loc30
// CHECK:STDOUT:   br !if.expr.result(%addr.loc30)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc30_5: %ptr.235 = block_arg !if.expr.result
// CHECK:STDOUT:   %.loc30_3: ref %i32 = deref %.loc30_5
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10.64f]
// CHECK:STDOUT:   %impl.elem0.loc30: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc30_29.1: <bound method> = bound_method %int_10, %impl.elem0.loc30 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2bf]
// CHECK:STDOUT:   %specific_fn.loc30: <specific function> = specific_function %impl.elem0.loc30, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc30_29.2: <bound method> = bound_method %int_10, %specific_fn.loc30 [concrete = constants.%bound_method.409]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc30: init %i32 = call %bound_method.loc30_29.2(%int_10) [concrete = constants.%int_10.265]
// CHECK:STDOUT:   %.loc30_29: init %i32 = converted %int_10, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc30 [concrete = constants.%int_10.265]
// CHECK:STDOUT:   assign %.loc30_3, %.loc30_29
// CHECK:STDOUT:   %DestroyOp.bound.loc27: <bound method> = bound_method %p.var, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc27: init %empty_tuple.type = call %DestroyOp.bound.loc27(%p.var)
// CHECK:STDOUT:   %DestroyOp.bound.loc23: <bound method> = bound_method %c.var, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc23: init %empty_tuple.type = call %DestroyOp.bound.loc23(%c.var)
// CHECK:STDOUT:   %DestroyOp.bound.loc19: <bound method> = bound_method %b.var, constants.%DestroyOp.b0ebf8.3
// CHECK:STDOUT:   %DestroyOp.call.loc19: init %empty_tuple.type = call %DestroyOp.bound.loc19(%b.var)
// CHECK:STDOUT:   %DestroyOp.bound.loc16: <bound method> = bound_method %a.var, constants.%DestroyOp.b0ebf8.4
// CHECK:STDOUT:   %DestroyOp.call.loc16: init %empty_tuple.type = call %DestroyOp.bound.loc16(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc27(%self.param: ref %ptr.235) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc23(%self.param: ref %struct_type.a.b.501) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc19(%self.param: ref %tuple.type.d07) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc16(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
