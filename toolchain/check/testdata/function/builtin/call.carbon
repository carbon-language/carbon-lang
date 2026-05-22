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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/call.carbon

fn Add(a: i32, b: i32) -> i32 = "int.sadd";

var arr: array(i32, Add(1, 2));

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Add(a, b);
}

// CHECK:STDOUT: --- call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.70c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %Add.type: type = fn_type @Add [concrete]
// CHECK:STDOUT:   %Add: %Add.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4bd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2f8: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.7c2: %Int.as.ImplicitAs.impl.Convert.type.2f8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e78: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.e78) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet.e78 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.33d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d5e: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.22e: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.cc3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.4ff: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.49a: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d2b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.07d, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.73a: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.735: %Int.as.ImplicitAs.impl.Convert.type.73a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.4dd: %ImplicitAs.type.4bd = facet_value %i32, (%ImplicitAs.impl_witness.d2b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f57: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.4dd) [concrete]
// CHECK:STDOUT:   %.516: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.f57, %ImplicitAs.facet.4dd [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.49a, %Int.as.ImplicitAs.impl.Convert.735 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.735, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a4: <bound method> = bound_method %int_3.49a, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.215: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.d5a: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.9f4: %T.as.DefaultOrUnformed.impl.Op.type.d5a = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.343: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.f82, @T.as.DefaultOrUnformed.impl(%array_type) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.26c: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%array_type) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.4a5: %T.as.DefaultOrUnformed.impl.Op.type.26c = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %array_type, (%DefaultOrUnformed.impl_witness.343) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.4a5, @T.as.DefaultOrUnformed.impl.Op(%array_type) [concrete]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [concrete]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.dd0: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.2f8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.7c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.07d = impl_witness_table (%Core.import_ref.dd0), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.b30: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.d5a) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.9f4)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.f82 = impl_witness_table (%Core.import_ref.b30), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Add.decl: %Add.type = fn_decl @Add [concrete = constants.%Add] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c6b = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.c6b = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.c6b = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.c6b = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32.loc15_27 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc15_27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15: Core.Form = init_form %i32.loc15_27 [concrete = constants.%.70c]
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc15_11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %a: %i32 = value_binding a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc15_19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %b: %i32 = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %arr.patt: %pattern_type.215 = ref_binding_pattern arr [concrete]
// CHECK:STDOUT:     %arr.var_patt: %pattern_type.215 = var_pattern %arr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr.var: ref %array_type = var %arr.var_patt [concrete]
// CHECK:STDOUT:   %.loc17_30: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %Add.ref: %Add.type = name_ref Add, %Add.decl [concrete = constants.%Add]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0.loc17_25: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:     %bound_method.loc17_25.1: <bound method> = bound_method %int_1, %impl.elem0.loc17_25 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d]
// CHECK:STDOUT:     %specific_fn.loc17_25: <specific function> = specific_function %impl.elem0.loc17_25, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc17_25.2: <bound method> = bound_method %int_1, %specific_fn.loc17_25 [concrete = constants.%bound_method.33d]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_25: init %i32 = call %bound_method.loc17_25.2(%int_1) [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:     %.loc17_25.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_25 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:     %.loc17_25.2: %i32 = converted %int_1, %.loc17_25.1 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:     %impl.elem0.loc17_28: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:     %bound_method.loc17_28.1: <bound method> = bound_method %int_2, %impl.elem0.loc17_28 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.22e]
// CHECK:STDOUT:     %specific_fn.loc17_28: <specific function> = specific_function %impl.elem0.loc17_28, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc17_28.2: <bound method> = bound_method %int_2, %specific_fn.loc17_28 [concrete = constants.%bound_method.cc3]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_28: init %i32 = call %bound_method.loc17_28.2(%int_2) [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:     %.loc17_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17_28 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:     %.loc17_28.2: %i32 = converted %int_2, %.loc17_28.1 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:     %Add.call: init %i32 = call %Add.ref(%.loc17_25.2, %.loc17_28.2) [concrete = constants.%int_3.49a]
// CHECK:STDOUT:     %impl.elem0.loc17_29: %.516 = impl_witness_access constants.%ImplicitAs.impl_witness.d2b, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.735]
// CHECK:STDOUT:     %bound_method.loc17_29.1: <bound method> = bound_method %Add.call, %impl.elem0.loc17_29 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc17_29: <specific function> = specific_function %impl.elem0.loc17_29, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc17_29.2: <bound method> = bound_method %Add.call, %specific_fn.loc17_29 [concrete = constants.%bound_method.9a4]
// CHECK:STDOUT:     %.loc17_29.1: %i32 = value_of_initializer %Add.call [concrete = constants.%int_3.49a]
// CHECK:STDOUT:     %.loc17_29.2: %i32 = converted %Add.call, %.loc17_29.1 [concrete = constants.%int_3.49a]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc17_29.2(%.loc17_29.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc17_29.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc17_29.4: Core.IntLiteral = converted %Add.call, %.loc17_29.3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %.loc17_29.4, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr: ref %array_type = ref_binding arr, %arr.var [concrete = %arr.var]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [concrete = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c6b = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.c6b = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.c6b = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.c6b = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32.loc19_35 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc19_35: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc19: Core.Form = init_form %i32.loc19_35 [concrete = constants.%.70c]
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc19_19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %a: %i32 = value_binding a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc19_27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %b: %i32 = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add(%a.param: %i32, %b.param: %i32) -> out %return.param: %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param: %i32, %b.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Add.ref: %Add.type = name_ref Add, file.%Add.decl [concrete = constants.%Add]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %Add.call: init %i32 = call %Add.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   return %Add.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%array_type, (constants.%DefaultOrUnformed.impl_witness.343) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc17_31.1: %DefaultOrUnformed.type = converted constants.%array_type, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc17_31.1 [concrete = constants.%array_type]
// CHECK:STDOUT:   %.loc17_31.2: type = converted %.loc17_31.1, %as_type [concrete = constants.%array_type]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.4a5, @T.as.DefaultOrUnformed.impl.Op(constants.%array_type) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:   %.loc17_1: ref %array_type = splice_block file.%arr.var [concrete = file.%arr.var] {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %array_type to %.loc17_1 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign file.%arr.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
