// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/if_expr/struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/if_expr/struct.carbon

fn G(s: {.a: i32, .b: i32});

fn F(cond: bool) {
  var a: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  G(if cond then a else a);
}

// CHECK:STDOUT: --- struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.501: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.851: type = pattern_type %struct_type.a.b.501 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.3, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.3, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.921: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.921 [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.3(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.501 = struct_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/types/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %s.patt: %pattern_type.851 = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.param_patt: %pattern_type.851 = value_param_pattern %s.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %s.param: %struct_type.a.b.501 = value_param call_param0
// CHECK:STDOUT:     %.loc14: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b.501] {
// CHECK:STDOUT:       %int_32.loc14_14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc14_14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_32.loc14_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc14_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.501]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s: %struct_type.a.b.501 = bind_name s, %s.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %cond.patt: %pattern_type.831 = binding_pattern cond [concrete]
// CHECK:STDOUT:     %cond.param_patt: %pattern_type.831 = value_param_pattern %cond.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %cond.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc16_12.1: type = splice_block %.loc16_12.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc16_12.2: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:       %.loc16_12.3: type = converted %bool.make_type, %.loc16_12.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %cond: bool = bind_name cond, %cond.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%s.param: %struct_type.a.b.501);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%cond.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.851 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.851 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %struct_type.a.b.501 = var %a.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc17_46.1: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:   %impl.elem0.loc17_46.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc17_46.1: <bound method> = bound_method %int_1, %impl.elem0.loc17_46.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc17_46.1: <specific function> = specific_function %impl.elem0.loc17_46.1, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_46.2: <bound method> = bound_method %int_1, %specific_fn.loc17_46.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc17_46.1: init %i32 = call %bound_method.loc17_46.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc17_46.2: init %i32 = converted %int_1, %int.convert_checked.loc17_46.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc17_46.3: ref %i32 = struct_access %a.var, element0
// CHECK:STDOUT:   %.loc17_46.4: init %i32 = initialize_from %.loc17_46.2 to %.loc17_46.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc17_46.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc17_46.3: <bound method> = bound_method %int_2, %impl.elem0.loc17_46.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc17_46.2: <specific function> = specific_function %impl.elem0.loc17_46.2, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_46.4: <bound method> = bound_method %int_2, %specific_fn.loc17_46.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc17_46.2: init %i32 = call %bound_method.loc17_46.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc17_46.5: init %i32 = converted %int_2, %int.convert_checked.loc17_46.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc17_46.6: ref %i32 = struct_access %a.var, element1
// CHECK:STDOUT:   %.loc17_46.7: init %i32 = initialize_from %.loc17_46.5 to %.loc17_46.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc17_46.8: init %struct_type.a.b.501 = struct_init (%.loc17_46.4, %.loc17_46.7) to %a.var [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc17_3: init %struct_type.a.b.501 = converted %.loc17_46.1, %.loc17_46.8 [concrete = constants.%struct]
// CHECK:STDOUT:   assign %a.var, %.loc17_3
// CHECK:STDOUT:   %.loc17_27: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b.501] {
// CHECK:STDOUT:     %int_32.loc17_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc17_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc17_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc17_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.501]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %struct_type.a.b.501 = bind_name a, %a.var
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %cond.ref: bool = name_ref cond, %cond
// CHECK:STDOUT:   if %cond.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %a.ref.loc18_18: ref %struct_type.a.b.501 = name_ref a, %a
// CHECK:STDOUT:   %.loc18_18.1: ref %i32 = struct_access %a.ref.loc18_18, element0
// CHECK:STDOUT:   %.loc18_18.2: %i32 = bind_value %.loc18_18.1
// CHECK:STDOUT:   %.loc18_18.3: ref %i32 = struct_access %a.ref.loc18_18, element1
// CHECK:STDOUT:   %.loc18_18.4: %i32 = bind_value %.loc18_18.3
// CHECK:STDOUT:   %struct.loc18_18: %struct_type.a.b.501 = struct_value (%.loc18_18.2, %.loc18_18.4)
// CHECK:STDOUT:   %.loc18_18.5: %struct_type.a.b.501 = converted %a.ref.loc18_18, %struct.loc18_18
// CHECK:STDOUT:   br !if.expr.result(%.loc18_18.5)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %a.ref.loc18_25: ref %struct_type.a.b.501 = name_ref a, %a
// CHECK:STDOUT:   %.loc18_25.1: ref %i32 = struct_access %a.ref.loc18_25, element0
// CHECK:STDOUT:   %.loc18_25.2: %i32 = bind_value %.loc18_25.1
// CHECK:STDOUT:   %.loc18_25.3: ref %i32 = struct_access %a.ref.loc18_25, element1
// CHECK:STDOUT:   %.loc18_25.4: %i32 = bind_value %.loc18_25.3
// CHECK:STDOUT:   %struct.loc18_25: %struct_type.a.b.501 = struct_value (%.loc18_25.2, %.loc18_25.4)
// CHECK:STDOUT:   %.loc18_20: %struct_type.a.b.501 = converted %a.ref.loc18_25, %struct.loc18_25
// CHECK:STDOUT:   br !if.expr.result(%.loc18_20)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc18_5: %struct_type.a.b.501 = block_arg !if.expr.result
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.ref(%.loc18_5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
