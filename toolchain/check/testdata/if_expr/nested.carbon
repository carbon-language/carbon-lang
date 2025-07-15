// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/if_expr/nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/if_expr/nested.carbon

fn F(a: bool, b: bool, c: bool) -> i32 {
  return if a then if b then 1 else 2 else if c then 3 else 4;
}

// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.2, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.1, @ImplicitAs.impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.1, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.1da: <bound method> = bound_method %int_4.0c1, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @ImplicitAs.impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @ImplicitAs.impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @ImplicitAs.impl.4f9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.patt: %pattern_type.831 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.831 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.831 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.831 = value_param_pattern %c.patt, call_param2 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc15_9.1: type = splice_block %.loc15_9.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc15_9: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc15_9.2: type = value_of_initializer %bool.make_type.loc15_9 [concrete = bool]
// CHECK:STDOUT:       %.loc15_9.3: type = converted %bool.make_type.loc15_9, %.loc15_9.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: bool = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc15_18.1: type = splice_block %.loc15_18.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc15_18: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc15_18.2: type = value_of_initializer %bool.make_type.loc15_18 [concrete = bool]
// CHECK:STDOUT:       %.loc15_18.3: type = converted %bool.make_type.loc15_18, %.loc15_18.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %c.param: bool = value_param call_param2
// CHECK:STDOUT:     %.loc15_27.1: type = splice_block %.loc15_27.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc15_27: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc15_27.2: type = value_of_initializer %bool.make_type.loc15_27 [concrete = bool]
// CHECK:STDOUT:       %.loc15_27.3: type = converted %bool.make_type.loc15_27, %.loc15_27.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: bool = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param3
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: bool, %b.param: bool, %c.param: bool) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: bool = name_ref a, %a
// CHECK:STDOUT:   if %a.ref br !if.expr.then.loc16_10 else br !if.expr.else.loc16_10
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc16_10:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.expr.then.loc16_20 else br !if.expr.else.loc16_20
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc16_20:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_32.loc16_25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc16_25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc16_25: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc16_25.1: <bound method> = bound_method %int_1, %impl.elem0.loc16_25 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc16_25: <specific function> = specific_function %impl.elem0.loc16_25, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_25.2: <bound method> = bound_method %int_1, %specific_fn.loc16_25 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc16_25: init %i32 = call %bound_method.loc16_25.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc16_25.1: %i32 = value_of_initializer %int.convert_checked.loc16_25 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc16_25.2: %i32 = converted %int_1, %.loc16_25.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   br !if.expr.result.loc16_20(%.loc16_25.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc16_20:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc16_32: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc16_32.1: <bound method> = bound_method %int_2, %impl.elem0.loc16_32 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc16_32: <specific function> = specific_function %impl.elem0.loc16_32, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_32.2: <bound method> = bound_method %int_2, %specific_fn.loc16_32 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc16_32: init %i32 = call %bound_method.loc16_32.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc16_32.1: %i32 = value_of_initializer %int.convert_checked.loc16_32 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc16_32.2: %i32 = converted %int_2, %.loc16_32.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   br !if.expr.result.loc16_20(%.loc16_32.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc16_20:
// CHECK:STDOUT:   %.loc16_20: %i32 = block_arg !if.expr.result.loc16_20
// CHECK:STDOUT:   br !if.expr.result.loc16_10(%.loc16_20)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc16_10:
// CHECK:STDOUT:   %c.ref: bool = name_ref c, %c
// CHECK:STDOUT:   if %c.ref br !if.expr.then.loc16_44 else br !if.expr.else.loc16_44
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc16_44:
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_32.loc16_49: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc16_49: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc16_49: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc16_49.1: <bound method> = bound_method %int_3, %impl.elem0.loc16_49 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc16_49: <specific function> = specific_function %impl.elem0.loc16_49, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_49.2: <bound method> = bound_method %int_3, %specific_fn.loc16_49 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc16_49: init %i32 = call %bound_method.loc16_49.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc16_49.1: %i32 = value_of_initializer %int.convert_checked.loc16_49 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc16_49.2: %i32 = converted %int_3, %.loc16_49.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   br !if.expr.result.loc16_44(%.loc16_49.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc16_44:
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc16_56: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc16_56.1: <bound method> = bound_method %int_4, %impl.elem0.loc16_56 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc16_56: <specific function> = specific_function %impl.elem0.loc16_56, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_56.2: <bound method> = bound_method %int_4, %specific_fn.loc16_56 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc16_56: init %i32 = call %bound_method.loc16_56.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc16_56.1: %i32 = value_of_initializer %int.convert_checked.loc16_56 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc16_56.2: %i32 = converted %int_4, %.loc16_56.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   br !if.expr.result.loc16_44(%.loc16_56.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc16_44:
// CHECK:STDOUT:   %.loc16_44: %i32 = block_arg !if.expr.result.loc16_44
// CHECK:STDOUT:   br !if.expr.result.loc16_10(%.loc16_44)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc16_10:
// CHECK:STDOUT:   %.loc16_10: %i32 = block_arg !if.expr.result.loc16_10
// CHECK:STDOUT:   return %.loc16_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
