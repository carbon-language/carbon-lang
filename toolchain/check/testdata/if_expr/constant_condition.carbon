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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/if_expr/constant_condition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/if_expr/constant_condition.carbon

fn A() -> i32 { return 1; }
fn B() -> i32 { return 2; }

fn F() -> i32 {
  return if true then A() else B();
}

fn G() -> i32 {
  return if false then A() else B();
}

fn Constant() -> i32 {
  var v: if true then i32 else i32* = 1;
  var w: if false then i32 else i32* = &v;
  return *w;
}

fn PartiallyConstant(t: type) -> i32 {
  var v: if true then i32 else t = 1;
  var w: if false then t else i32* = &v;
  return *w;
}

// CHECK:STDOUT: --- constant_condition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.265: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.dfc: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.393: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.9f7: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.24b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c95: %Int.as.Copy.impl.Op.type.24b = struct_value () [symbolic]
// CHECK:STDOUT:   %T.d9f: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.75b: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.d9f) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.692: %ptr.as.Copy.impl.Op.type.75b = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.fb7: <witness> = impl_witness imports.%Copy.impl_witness_table.b6a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.469: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.dfd: %Int.as.Copy.impl.Op.type.469 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.c50: %Copy.type = facet_value %i32, (%Copy.impl_witness.fb7) [concrete]
// CHECK:STDOUT:   %.65f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.c50 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.dfd, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %Constant.type: type = fn_type @Constant [concrete]
// CHECK:STDOUT:   %Constant: %Constant.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %ptr.235 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.be3: <witness> = impl_witness imports.%Copy.impl_witness_table.67d, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.d7c: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.687: %ptr.as.Copy.impl.Op.type.d7c = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.687: %Copy.type = facet_value %ptr.235, (%Copy.impl_witness.be3) [concrete]
// CHECK:STDOUT:   %.cbc: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.687 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.687, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.380: %type_where = facet_value %ptr.235, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.00f: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.380) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.6d0: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.00f = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5d5: type = ptr_type %ptr.235 [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.a6e: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.6d0, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.380) [concrete]
// CHECK:STDOUT:   %facet_value.d23: %type_where = facet_value %i32, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d4e: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.d23) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.424: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d4e = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.c20: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.424, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.d23) [concrete]
// CHECK:STDOUT:   %PartiallyConstant.type: type = fn_type @PartiallyConstant [concrete]
// CHECK:STDOUT:   %PartiallyConstant: %PartiallyConstant.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.d12: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.24b) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c95)]
// CHECK:STDOUT:   %Copy.impl_witness_table.b6a = impl_witness_table (%Core.import_ref.d12), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.659: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.75b) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.692)]
// CHECK:STDOUT:   %Copy.impl_witness_table.67d = impl_witness_table (%Core.import_ref.659), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .Constant = %Constant.decl
// CHECK:STDOUT:     .PartiallyConstant = %PartiallyConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Constant.decl: %Constant.type = fn_decl @Constant [concrete = constants.%Constant] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc26: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc26: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PartiallyConstant.decl: %PartiallyConstant.type = fn_decl @PartiallyConstant [concrete = constants.%PartiallyConstant] {
// CHECK:STDOUT:     %t.patt: %pattern_type.98f = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: %pattern_type.98f = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %t.param: type = value_param call_param0
// CHECK:STDOUT:     %t: type = value_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc15_25.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.265]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_25.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method.dfc]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc15_25.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc15: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   return %.loc15 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc16_25.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.393]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_25.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.9f7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc16_25.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc16: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   return %.loc16 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %i32 = call %A.ref()
// CHECK:STDOUT:   %.loc19_25.1: %i32 = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc19_25.2: %i32 = converted %A.call, %.loc19_25.1
// CHECK:STDOUT:   br !if.expr.result(%.loc19_25.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref()
// CHECK:STDOUT:   %.loc19_27.1: %i32 = value_of_initializer %B.call
// CHECK:STDOUT:   %.loc19_27.2: %i32 = converted %B.call, %.loc19_27.1
// CHECK:STDOUT:   br !if.expr.result(%.loc19_27.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc19_10: %i32 = block_arg !if.expr.result
// CHECK:STDOUT:   %impl.elem0: %.65f = impl_witness_access constants.%Copy.impl_witness.fb7, element0 [concrete = constants.%Int.as.Copy.impl.Op.dfd]
// CHECK:STDOUT:   %bound_method.loc19_10.1: <bound method> = bound_method %.loc19_10, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19_10.2: <bound method> = bound_method %.loc19_10, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc19_10.2(%.loc19_10)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %i32 = call %A.ref()
// CHECK:STDOUT:   %.loc23_26.1: %i32 = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc23_26.2: %i32 = converted %A.call, %.loc23_26.1
// CHECK:STDOUT:   br !if.expr.result(%.loc23_26.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref()
// CHECK:STDOUT:   %.loc23_28.1: %i32 = value_of_initializer %B.call
// CHECK:STDOUT:   %.loc23_28.2: %i32 = converted %B.call, %.loc23_28.1
// CHECK:STDOUT:   br !if.expr.result(%.loc23_28.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc23_10: %i32 = block_arg !if.expr.result
// CHECK:STDOUT:   %impl.elem0: %.65f = impl_witness_access constants.%Copy.impl_witness.fb7, element0 [concrete = constants.%Int.as.Copy.impl.Op.dfd]
// CHECK:STDOUT:   %bound_method.loc23_10.1: <bound method> = bound_method %.loc23_10, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_10.2: <bound method> = bound_method %.loc23_10, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc23_10.2(%.loc23_10)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Constant() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.7ce = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.7ce = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %i32 = var %v.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc27: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc27_3.1: <bound method> = bound_method %int_1, %impl.elem0.loc27 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.265]
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_3.2: <bound method> = bound_method %int_1, %specific_fn.loc27 [concrete = constants.%bound_method.dfc]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc27_3.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc27_3: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   assign %v.var, %.loc27_3
// CHECK:STDOUT:   br !.loc27_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc27_13:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.expr.then.loc27 else br !if.expr.else.loc27
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc27:
// CHECK:STDOUT:   %int_32.loc27_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc27_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result.loc27(%i32.loc27_23)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc27:
// CHECK:STDOUT:   %int_32.loc27_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc27_32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %ptr.loc27: type = ptr_type %i32.loc27_32 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   br !if.expr.result.loc27(%ptr.loc27)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc27:
// CHECK:STDOUT:   %.loc27_10: type = block_arg !if.expr.result.loc27 [concrete = constants.%i32]
// CHECK:STDOUT:   br !.loc27_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc27_7:
// CHECK:STDOUT:   %v: ref %i32 = ref_binding v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type.fe8 = ref_binding_pattern w [concrete]
// CHECK:STDOUT:     %w.var_patt: %pattern_type.fe8 = var_pattern %w.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.var: ref %ptr.235 = var %w.var_patt
// CHECK:STDOUT:   %v.ref: ref %i32 = name_ref v, %v
// CHECK:STDOUT:   %addr.loc28_40: %ptr.235 = addr_of %v.ref
// CHECK:STDOUT:   %impl.elem0.loc28: %.cbc = impl_witness_access constants.%Copy.impl_witness.be3, element0 [concrete = constants.%ptr.as.Copy.impl.Op.687]
// CHECK:STDOUT:   %bound_method.loc28_40.1: <bound method> = bound_method %addr.loc28_40, %impl.elem0.loc28
// CHECK:STDOUT:   %specific_fn.loc28: <specific function> = specific_function %impl.elem0.loc28, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc28_40.2: <bound method> = bound_method %addr.loc28_40, %specific_fn.loc28
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.235 = call %bound_method.loc28_40.2(%addr.loc28_40)
// CHECK:STDOUT:   assign %w.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   br !.loc28_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc28_13:
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false br !if.expr.then.loc28 else br !if.expr.else.loc28
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc28:
// CHECK:STDOUT:   %int_32.loc28_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc28_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result.loc28(%i32.loc28_24)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc28:
// CHECK:STDOUT:   %int_32.loc28_33: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc28_33: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %ptr.loc28: type = ptr_type %i32.loc28_33 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   br !if.expr.result.loc28(%ptr.loc28)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc28:
// CHECK:STDOUT:   %.loc28: type = block_arg !if.expr.result.loc28 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   br !.loc28_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc28_7:
// CHECK:STDOUT:   %w: ref %ptr.235 = ref_binding w, %w.var
// CHECK:STDOUT:   %w.ref: ref %ptr.235 = name_ref w, %w
// CHECK:STDOUT:   %.loc29_11: %ptr.235 = acquire_value %w.ref
// CHECK:STDOUT:   %.loc29_10.1: ref %i32 = deref %.loc29_11
// CHECK:STDOUT:   %.loc29_10.2: %i32 = acquire_value %.loc29_10.1
// CHECK:STDOUT:   %impl.elem0.loc29: %.65f = impl_witness_access constants.%Copy.impl_witness.fb7, element0 [concrete = constants.%Int.as.Copy.impl.Op.dfd]
// CHECK:STDOUT:   %bound_method.loc29_10.1: <bound method> = bound_method %.loc29_10.2, %impl.elem0.loc29
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem0.loc29, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc29_10.2: <bound method> = bound_method %.loc29_10.2, %specific_fn.loc29
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc29_10.2(%.loc29_10.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc28: <bound method> = bound_method %w.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.6d0
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.6d0, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.380) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.a6e]
// CHECK:STDOUT:   %bound_method.loc28_3: <bound method> = bound_method %w.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc28_3: %ptr.5d5 = addr_of %w.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc28: init %empty_tuple.type = call %bound_method.loc28_3(%addr.loc28_3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc27: <bound method> = bound_method %v.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.424
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.424, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.d23) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.c20]
// CHECK:STDOUT:   %bound_method.loc27_3.3: <bound method> = bound_method %v.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc27: %ptr.235 = addr_of %v.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc27: init %empty_tuple.type = call %bound_method.loc27_3.3(%addr.loc27)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartiallyConstant(%t.param: type) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.7ce = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.7ce = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %i32 = var %v.var_patt
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc33: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc33_3.1: <bound method> = bound_method %int_1, %impl.elem0.loc33 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.265]
// CHECK:STDOUT:   %specific_fn.loc33: <specific function> = specific_function %impl.elem0.loc33, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc33_3.2: <bound method> = bound_method %int_1, %specific_fn.loc33 [concrete = constants.%bound_method.dfc]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc33_3.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc33_3: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   assign %v.var, %.loc33_3
// CHECK:STDOUT:   br !.loc33_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc33_13:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.expr.then.loc33 else br !if.expr.else.loc33
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc33:
// CHECK:STDOUT:   %int_32.loc33: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc33: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result.loc33(%i32.loc33)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc33:
// CHECK:STDOUT:   %t.ref.loc33: type = name_ref t, %t
// CHECK:STDOUT:   br !if.expr.result.loc33(%t.ref.loc33)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc33:
// CHECK:STDOUT:   %.loc33_10: type = block_arg !if.expr.result.loc33 [concrete = constants.%i32]
// CHECK:STDOUT:   br !.loc33_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc33_7:
// CHECK:STDOUT:   %v: ref %i32 = ref_binding v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type.fe8 = ref_binding_pattern w [concrete]
// CHECK:STDOUT:     %w.var_patt: %pattern_type.fe8 = var_pattern %w.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.var: ref %ptr.235 = var %w.var_patt
// CHECK:STDOUT:   %v.ref: ref %i32 = name_ref v, %v
// CHECK:STDOUT:   %addr.loc34_38: %ptr.235 = addr_of %v.ref
// CHECK:STDOUT:   %impl.elem0.loc34: %.cbc = impl_witness_access constants.%Copy.impl_witness.be3, element0 [concrete = constants.%ptr.as.Copy.impl.Op.687]
// CHECK:STDOUT:   %bound_method.loc34_38.1: <bound method> = bound_method %addr.loc34_38, %impl.elem0.loc34
// CHECK:STDOUT:   %specific_fn.loc34: <specific function> = specific_function %impl.elem0.loc34, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_38.2: <bound method> = bound_method %addr.loc34_38, %specific_fn.loc34
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.235 = call %bound_method.loc34_38.2(%addr.loc34_38)
// CHECK:STDOUT:   assign %w.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   br !.loc34_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc34_13:
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false br !if.expr.then.loc34 else br !if.expr.else.loc34
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc34:
// CHECK:STDOUT:   %t.ref.loc34: type = name_ref t, %t
// CHECK:STDOUT:   br !if.expr.result.loc34(%t.ref.loc34)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc34:
// CHECK:STDOUT:   %int_32.loc34: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc34: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %ptr: type = ptr_type %i32.loc34 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   br !if.expr.result.loc34(%ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc34:
// CHECK:STDOUT:   %.loc34: type = block_arg !if.expr.result.loc34 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   br !.loc34_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc34_7:
// CHECK:STDOUT:   %w: ref %ptr.235 = ref_binding w, %w.var
// CHECK:STDOUT:   %w.ref: ref %ptr.235 = name_ref w, %w
// CHECK:STDOUT:   %.loc35_11: %ptr.235 = acquire_value %w.ref
// CHECK:STDOUT:   %.loc35_10.1: ref %i32 = deref %.loc35_11
// CHECK:STDOUT:   %.loc35_10.2: %i32 = acquire_value %.loc35_10.1
// CHECK:STDOUT:   %impl.elem0.loc35: %.65f = impl_witness_access constants.%Copy.impl_witness.fb7, element0 [concrete = constants.%Int.as.Copy.impl.Op.dfd]
// CHECK:STDOUT:   %bound_method.loc35_10.1: <bound method> = bound_method %.loc35_10.2, %impl.elem0.loc35
// CHECK:STDOUT:   %specific_fn.loc35: <specific function> = specific_function %impl.elem0.loc35, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc35_10.2: <bound method> = bound_method %.loc35_10.2, %specific_fn.loc35
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc35_10.2(%.loc35_10.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc34: <bound method> = bound_method %w.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.6d0
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.6d0, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.380) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.a6e]
// CHECK:STDOUT:   %bound_method.loc34_3: <bound method> = bound_method %w.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc34_3: %ptr.5d5 = addr_of %w.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc34: init %empty_tuple.type = call %bound_method.loc34_3(%addr.loc34_3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc33: <bound method> = bound_method %v.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.424
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.424, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.d23) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.c20]
// CHECK:STDOUT:   %bound_method.loc33_3.3: <bound method> = bound_method %v.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc33: %ptr.235 = addr_of %v.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc33: init %empty_tuple.type = call %bound_method.loc33_3.3(%addr.loc33)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
