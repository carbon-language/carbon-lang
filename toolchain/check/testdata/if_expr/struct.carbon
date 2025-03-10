// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.501: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
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
// CHECK:STDOUT:     %s.patt: %struct_type.a.b.501 = binding_pattern s
// CHECK:STDOUT:     %s.param_patt: %struct_type.a.b.501 = value_param_pattern %s.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %s.param: %struct_type.a.b.501 = value_param call_param0
// CHECK:STDOUT:     %.loc11: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b.501] {
// CHECK:STDOUT:       %int_32.loc11_14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_32.loc11_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.501]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s: %struct_type.a.b.501 = bind_name s, %s.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %cond.patt: bool = binding_pattern cond
// CHECK:STDOUT:     %cond.param_patt: bool = value_param_pattern %cond.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %cond.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc13_12.1: type = splice_block %.loc13_12.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc13_12.2: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:       %.loc13_12.3: type = converted %bool.make_type, %.loc13_12.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %cond: bool = bind_name cond, %cond.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%s.param_patt: %struct_type.a.b.501);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%cond.param_patt: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %struct_type.a.b.501 = binding_pattern a
// CHECK:STDOUT:     %.loc14_3.1: %struct_type.a.b.501 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %struct_type.a.b.501 = var a
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc14_46.1: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:   %impl.elem0.loc14_46.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_46.1: <bound method> = bound_method %int_1, %impl.elem0.loc14_46.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc14_46.1: <specific function> = specific_function %impl.elem0.loc14_46.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_46.2: <bound method> = bound_method %int_1, %specific_fn.loc14_46.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc14_46.1: init %i32 = call %bound_method.loc14_46.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_46.2: init %i32 = converted %int_1, %int.convert_checked.loc14_46.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_46.3: ref %i32 = struct_access %a.var, element0
// CHECK:STDOUT:   %.loc14_46.4: init %i32 = initialize_from %.loc14_46.2 to %.loc14_46.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc14_46.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_46.3: <bound method> = bound_method %int_2, %impl.elem0.loc14_46.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc14_46.2: <specific function> = specific_function %impl.elem0.loc14_46.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_46.4: <bound method> = bound_method %int_2, %specific_fn.loc14_46.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc14_46.2: init %i32 = call %bound_method.loc14_46.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc14_46.5: init %i32 = converted %int_2, %int.convert_checked.loc14_46.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc14_46.6: ref %i32 = struct_access %a.var, element1
// CHECK:STDOUT:   %.loc14_46.7: init %i32 = initialize_from %.loc14_46.5 to %.loc14_46.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc14_46.8: init %struct_type.a.b.501 = struct_init (%.loc14_46.4, %.loc14_46.7) to %a.var [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc14_3.2: init %struct_type.a.b.501 = converted %.loc14_46.1, %.loc14_46.8 [concrete = constants.%struct]
// CHECK:STDOUT:   assign %a.var, %.loc14_3.2
// CHECK:STDOUT:   %.loc14_27: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b.501] {
// CHECK:STDOUT:     %int_32.loc14_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc14_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.501]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %struct_type.a.b.501 = bind_name a, %a.var
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %cond.ref: bool = name_ref cond, %cond
// CHECK:STDOUT:   if %cond.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %a.ref.loc15_18: ref %struct_type.a.b.501 = name_ref a, %a
// CHECK:STDOUT:   %.loc15_18.1: ref %i32 = struct_access %a.ref.loc15_18, element0
// CHECK:STDOUT:   %.loc15_18.2: %i32 = bind_value %.loc15_18.1
// CHECK:STDOUT:   %.loc15_18.3: ref %i32 = struct_access %a.ref.loc15_18, element1
// CHECK:STDOUT:   %.loc15_18.4: %i32 = bind_value %.loc15_18.3
// CHECK:STDOUT:   %struct.loc15_18: %struct_type.a.b.501 = struct_value (%.loc15_18.2, %.loc15_18.4)
// CHECK:STDOUT:   %.loc15_18.5: %struct_type.a.b.501 = converted %a.ref.loc15_18, %struct.loc15_18
// CHECK:STDOUT:   br !if.expr.result(%.loc15_18.5)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %a.ref.loc15_25: ref %struct_type.a.b.501 = name_ref a, %a
// CHECK:STDOUT:   %.loc15_25.1: ref %i32 = struct_access %a.ref.loc15_25, element0
// CHECK:STDOUT:   %.loc15_25.2: %i32 = bind_value %.loc15_25.1
// CHECK:STDOUT:   %.loc15_25.3: ref %i32 = struct_access %a.ref.loc15_25, element1
// CHECK:STDOUT:   %.loc15_25.4: %i32 = bind_value %.loc15_25.3
// CHECK:STDOUT:   %struct.loc15_25: %struct_type.a.b.501 = struct_value (%.loc15_25.2, %.loc15_25.4)
// CHECK:STDOUT:   %.loc15_20: %struct_type.a.b.501 = converted %a.ref.loc15_25, %struct.loc15_25
// CHECK:STDOUT:   br !if.expr.result(%.loc15_20)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc15_5: %struct_type.a.b.501 = block_arg !if.expr.result
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.ref(%.loc15_5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
