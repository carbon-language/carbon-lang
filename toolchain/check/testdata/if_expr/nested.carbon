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
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.204: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9e9, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.204) [concrete]
// CHECK:STDOUT:   %.1df: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.c11: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %bound_method.8bd: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %bound_method.f36: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5bb: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %bound_method.9cd: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.ee7: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340) = import_ref Core//prelude/parts/int, loc23_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9e9 = impl_witness_table (%Core.import_ref.ee7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc17_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:       %Bool.call.loc15_9: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc15_9.2: type = value_of_initializer %Bool.call.loc15_9 [concrete = bool]
// CHECK:STDOUT:       %.loc15_9.3: type = converted %Bool.call.loc15_9, %.loc15_9.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: bool = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc15_18.1: type = splice_block %.loc15_18.3 [concrete = bool] {
// CHECK:STDOUT:       %Bool.call.loc15_18: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc15_18.2: type = value_of_initializer %Bool.call.loc15_18 [concrete = bool]
// CHECK:STDOUT:       %.loc15_18.3: type = converted %Bool.call.loc15_18, %.loc15_18.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %c.param: bool = value_param call_param2
// CHECK:STDOUT:     %.loc15_27.1: type = splice_block %.loc15_27.3 [concrete = bool] {
// CHECK:STDOUT:       %Bool.call.loc15_27: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc15_27.2: type = value_of_initializer %Bool.call.loc15_27 [concrete = bool]
// CHECK:STDOUT:       %.loc15_27.3: type = converted %Bool.call.loc15_27, %.loc15_27.2 [concrete = bool]
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
// CHECK:STDOUT:   %impl.elem0.loc16_25: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc16_25.1: <bound method> = bound_method %int_1, %impl.elem0.loc16_25 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc16_25: <specific function> = specific_function %impl.elem0.loc16_25, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_25.2: <bound method> = bound_method %int_1, %specific_fn.loc16_25 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_25: init %i32 = call %bound_method.loc16_25.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc16_25.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_25 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc16_25.2: %i32 = converted %int_1, %.loc16_25.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   br !if.expr.result.loc16_20(%.loc16_25.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc16_20:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc16_32: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc16_32.1: <bound method> = bound_method %int_2, %impl.elem0.loc16_32 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82]
// CHECK:STDOUT:   %specific_fn.loc16_32: <specific function> = specific_function %impl.elem0.loc16_32, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_32.2: <bound method> = bound_method %int_2, %specific_fn.loc16_32 [concrete = constants.%bound_method.8bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_32: init %i32 = call %bound_method.loc16_32.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc16_32.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_32 [concrete = constants.%int_2.ef8]
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
// CHECK:STDOUT:   %impl.elem0.loc16_49: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc16_49.1: <bound method> = bound_method %int_3, %impl.elem0.loc16_49 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595]
// CHECK:STDOUT:   %specific_fn.loc16_49: <specific function> = specific_function %impl.elem0.loc16_49, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_49.2: <bound method> = bound_method %int_3, %specific_fn.loc16_49 [concrete = constants.%bound_method.f36]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_49: init %i32 = call %bound_method.loc16_49.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc16_49.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_49 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc16_49.2: %i32 = converted %int_3, %.loc16_49.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   br !if.expr.result.loc16_44(%.loc16_49.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc16_44:
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc16_56: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc16_56.1: <bound method> = bound_method %int_4, %impl.elem0.loc16_56 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5bb]
// CHECK:STDOUT:   %specific_fn.loc16_56: <specific function> = specific_function %impl.elem0.loc16_56, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_56.2: <bound method> = bound_method %int_4, %specific_fn.loc16_56 [concrete = constants.%bound_method.9cd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_56: init %i32 = call %bound_method.loc16_56.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc16_56.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16_56 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc16_56.2: %i32 = converted %int_4, %.loc16_56.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   br !if.expr.result.loc16_44(%.loc16_56.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc16_44:
// CHECK:STDOUT:   %.loc16_44: %i32 = block_arg !if.expr.result.loc16_44
// CHECK:STDOUT:   br !if.expr.result.loc16_10(%.loc16_44)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc16_10:
// CHECK:STDOUT:   %.loc16_10: %i32 = block_arg !if.expr.result.loc16_10
// CHECK:STDOUT:   %impl.elem0.loc16_10: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc16_10.1: <bound method> = bound_method %.loc16_10, %impl.elem0.loc16_10
// CHECK:STDOUT:   %specific_fn.loc16_10: <specific function> = specific_function %impl.elem0.loc16_10, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_10.2: <bound method> = bound_method %.loc16_10, %specific_fn.loc16_10
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc16_10.2(%.loc16_10)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
