// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/and.carbon

fn F() -> bool { return true; }
fn G() -> bool { return true; }

fn And() -> bool {
  return F() and G();
}

fn Constant() {
  var a: if true and true then bool else () = true;
  var b: if true and false then bool else () = ();
  var c: if false and true then bool else () = ();
  var d: if false and false then bool else () = ();
}

fn PartialConstant(x: bool) {
  var a: if false and x then bool else () = ();
}

// CHECK:STDOUT: --- and.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.a56: <witness> = impl_witness imports.%Copy.impl_witness_table.189 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value bool, (%Copy.impl_witness.a56) [concrete]
// CHECK:STDOUT:   %.05c: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound: <bound method> = bound_method %true, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %And.type: type = fn_type @And [concrete]
// CHECK:STDOUT:   %And: %And.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %Constant.type: type = fn_type @Constant [concrete]
// CHECK:STDOUT:   %Constant: %Constant.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.ff9: %type_where = facet_value %empty_tuple.type, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.bb8: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.ff9) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.72f: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.bb8 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.843: type = ptr_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.b2f: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.72f, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.ff9) [concrete]
// CHECK:STDOUT:   %facet_value.3f5: %type_where = facet_value bool, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.694: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.3f5) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.2db: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.694 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.8f1: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.2db, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.3f5) [concrete]
// CHECK:STDOUT:   %PartialConstant.type: type = fn_type @PartialConstant [concrete]
// CHECK:STDOUT:   %PartialConstant: %PartialConstant.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.cc5: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.189 = impl_witness_table (%Core.import_ref.cc5), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .And = %And.decl
// CHECK:STDOUT:     .Constant = %Constant.decl
// CHECK:STDOUT:     .PartialConstant = %PartialConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc15_11.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc15_11.2: type = converted %Bool.call, %.loc15_11.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc16_11.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc16_11.2: type = converted %Bool.call, %.loc16_11.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %And.decl: %And.type = fn_decl @And [concrete = constants.%And] {
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc18_13.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc18_13.2: type = converted %Bool.call, %.loc18_13.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Constant.decl: %Constant.type = fn_decl @Constant [concrete = constants.%Constant] {} {}
// CHECK:STDOUT:   %PartialConstant.decl: %PartialConstant.type = fn_decl @PartialConstant [concrete = constants.%PartialConstant] {
// CHECK:STDOUT:     %x.patt: %pattern_type.831 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.831 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc29_23.1: type = splice_block %.loc29_23.3 [concrete = bool] {
// CHECK:STDOUT:       %Bool.call.loc29: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc29_23.2: type = value_of_initializer %Bool.call.loc29 [concrete = bool]
// CHECK:STDOUT:       %.loc29_23.3: type = converted %Bool.call.loc29, %.loc29_23.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: bool = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %impl.elem0: %.05c = impl_witness_access constants.%Copy.impl_witness.a56, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %true, %impl.elem0 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method(%true) [concrete = constants.%true]
// CHECK:STDOUT:   return %bool.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %impl.elem0: %.05c = impl_witness_access constants.%Copy.impl_witness.a56, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %true, %impl.elem0 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method(%true) [concrete = constants.%true]
// CHECK:STDOUT:   return %bool.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @And() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init bool = call %F.ref()
// CHECK:STDOUT:   %.loc19_14.1: bool = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc19_14.2: bool = converted %F.call, %.loc19_14.1
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %.loc19_14.2 br !and.rhs else br !and.result(%false)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init bool = call %G.ref()
// CHECK:STDOUT:   %.loc19_14.3: bool = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc19_14.4: bool = converted %G.call, %.loc19_14.3
// CHECK:STDOUT:   br !and.result(%.loc19_14.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc19_14.5: bool = block_arg !and.result
// CHECK:STDOUT:   %impl.elem0: %.05c = impl_witness_access constants.%Copy.impl_witness.a56, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc19_14.5, %impl.elem0
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method(%.loc19_14.5)
// CHECK:STDOUT:   return %bool.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Constant() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.831 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.831 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref bool = var %a.var_patt
// CHECK:STDOUT:   %true.loc23_47: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %impl.elem0: %.05c = impl_witness_access constants.%Copy.impl_witness.a56, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc23_47: <bound method> = bound_method %true.loc23_47, %impl.elem0 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method.loc23_47(%true.loc23_47) [concrete = constants.%true]
// CHECK:STDOUT:   assign %a.var, %bool.as.Copy.impl.Op.call
// CHECK:STDOUT:   br !.loc23_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc23_13:
// CHECK:STDOUT:   %true.loc23_13: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %false.loc23: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %true.loc23_13 br !and.rhs.loc23 else br !and.result.loc23(%false.loc23)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc23:
// CHECK:STDOUT:   %true.loc23_22: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   br !and.result.loc23(%true.loc23_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc23:
// CHECK:STDOUT:   %.loc23_18: bool = block_arg !and.result.loc23 [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc23_18 br !if.expr.then.loc23 else br !if.expr.else.loc23
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc23:
// CHECK:STDOUT:   %Bool.call.loc23: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc23_32.1: type = value_of_initializer %Bool.call.loc23 [concrete = bool]
// CHECK:STDOUT:   %.loc23_32.2: type = converted %Bool.call.loc23, %.loc23_32.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc23(%.loc23_32.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc23:
// CHECK:STDOUT:   %.loc23_43: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc23_37: type = converted %.loc23_43, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc23(%.loc23_37)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc23:
// CHECK:STDOUT:   %.loc23_10: type = block_arg !if.expr.result.loc23 [concrete = bool]
// CHECK:STDOUT:   br !.loc23_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc23_7:
// CHECK:STDOUT:   %a: ref bool = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.cb1 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %empty_tuple.type = var %b.var_patt
// CHECK:STDOUT:   %.loc24_49.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc24_49.2: init %empty_tuple.type = tuple_init () to %b.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc24_3: init %empty_tuple.type = converted %.loc24_49.1, %.loc24_49.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %b.var, %.loc24_3
// CHECK:STDOUT:   br !.loc24_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc24_13:
// CHECK:STDOUT:   %true.loc24: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %false.loc24_18: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %true.loc24 br !and.rhs.loc24 else br !and.result.loc24(%false.loc24_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc24:
// CHECK:STDOUT:   %false.loc24_22: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   br !and.result.loc24(%false.loc24_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc24:
// CHECK:STDOUT:   %.loc24_18: bool = block_arg !and.result.loc24 [concrete = constants.%false]
// CHECK:STDOUT:   if %.loc24_18 br !if.expr.then.loc24 else br !if.expr.else.loc24
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc24:
// CHECK:STDOUT:   %Bool.call.loc24: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc24_33.1: type = value_of_initializer %Bool.call.loc24 [concrete = bool]
// CHECK:STDOUT:   %.loc24_33.2: type = converted %Bool.call.loc24, %.loc24_33.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc24(%.loc24_33.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc24:
// CHECK:STDOUT:   %.loc24_44: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc24_38: type = converted %.loc24_44, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc24(%.loc24_38)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc24:
// CHECK:STDOUT:   %.loc24_10: type = block_arg !if.expr.result.loc24 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !.loc24_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc24_7:
// CHECK:STDOUT:   %b: ref %empty_tuple.type = ref_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.cb1 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.cb1 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %empty_tuple.type = var %c.var_patt
// CHECK:STDOUT:   %.loc25_49.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc25_49.2: init %empty_tuple.type = tuple_init () to %c.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc25_3: init %empty_tuple.type = converted %.loc25_49.1, %.loc25_49.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %c.var, %.loc25_3
// CHECK:STDOUT:   br !.loc25_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc25_13:
// CHECK:STDOUT:   %false.loc25_13: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %false.loc25_19: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc25_13 br !and.rhs.loc25 else br !and.result.loc25(%false.loc25_19)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc25:
// CHECK:STDOUT:   %true.loc25: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   br !and.result.loc25(%true.loc25)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc25:
// CHECK:STDOUT:   %.loc25_19: bool = block_arg !and.result.loc25 [concrete = constants.%false]
// CHECK:STDOUT:   if %.loc25_19 br !if.expr.then.loc25 else br !if.expr.else.loc25
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc25:
// CHECK:STDOUT:   %Bool.call.loc25: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc25_33.1: type = value_of_initializer %Bool.call.loc25 [concrete = bool]
// CHECK:STDOUT:   %.loc25_33.2: type = converted %Bool.call.loc25, %.loc25_33.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc25(%.loc25_33.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc25:
// CHECK:STDOUT:   %.loc25_44: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc25_38: type = converted %.loc25_44, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc25(%.loc25_38)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc25:
// CHECK:STDOUT:   %.loc25_10: type = block_arg !if.expr.result.loc25 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !.loc25_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc25_7:
// CHECK:STDOUT:   %c: ref %empty_tuple.type = ref_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.cb1 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.cb1 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var %d.var_patt
// CHECK:STDOUT:   %.loc26_50.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc26_50.2: init %empty_tuple.type = tuple_init () to %d.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc26_3: init %empty_tuple.type = converted %.loc26_50.1, %.loc26_50.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %d.var, %.loc26_3
// CHECK:STDOUT:   br !.loc26_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc26_13:
// CHECK:STDOUT:   %false.loc26_13: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %false.loc26_19: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc26_13 br !and.rhs.loc26 else br !and.result.loc26(%false.loc26_19)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc26:
// CHECK:STDOUT:   %false.loc26_23: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   br !and.result.loc26(%false.loc26_23)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc26:
// CHECK:STDOUT:   %.loc26_19: bool = block_arg !and.result.loc26 [concrete = constants.%false]
// CHECK:STDOUT:   if %.loc26_19 br !if.expr.then.loc26 else br !if.expr.else.loc26
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc26:
// CHECK:STDOUT:   %Bool.call.loc26: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc26_34.1: type = value_of_initializer %Bool.call.loc26 [concrete = bool]
// CHECK:STDOUT:   %.loc26_34.2: type = converted %Bool.call.loc26, %.loc26_34.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc26(%.loc26_34.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc26:
// CHECK:STDOUT:   %.loc26_45: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc26_39: type = converted %.loc26_45, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc26(%.loc26_39)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc26:
// CHECK:STDOUT:   %.loc26_10: type = block_arg !if.expr.result.loc26 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !.loc26_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc26_7:
// CHECK:STDOUT:   %d: ref %empty_tuple.type = ref_binding d, %d.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc26: <bound method> = bound_method %d.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72f
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72f, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.ff9) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.b2f]
// CHECK:STDOUT:   %bound_method.loc26: <bound method> = bound_method %d.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc26: %ptr.843 = addr_of %d.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc26: init %empty_tuple.type = call %bound_method.loc26(%addr.loc26)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc25: <bound method> = bound_method %c.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72f
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72f, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.ff9) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.b2f]
// CHECK:STDOUT:   %bound_method.loc25: <bound method> = bound_method %c.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc25: %ptr.843 = addr_of %c.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc25: init %empty_tuple.type = call %bound_method.loc25(%addr.loc25)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc24: <bound method> = bound_method %b.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72f
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72f, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.ff9) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.b2f]
// CHECK:STDOUT:   %bound_method.loc24: <bound method> = bound_method %b.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc24: %ptr.843 = addr_of %b.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc24: init %empty_tuple.type = call %bound_method.loc24(%addr.loc24)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc23: <bound method> = bound_method %a.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2db
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.2db, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.3f5) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.8f1]
// CHECK:STDOUT:   %bound_method.loc23_3: <bound method> = bound_method %a.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4
// CHECK:STDOUT:   %addr.loc23: %ptr.bb2 = addr_of %a.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc23: init %empty_tuple.type = call %bound_method.loc23_3(%addr.loc23)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartialConstant(%x.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.cb1 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var %a.var_patt
// CHECK:STDOUT:   %.loc30_46.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc30_46.2: init %empty_tuple.type = tuple_init () to %a.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc30_3: init %empty_tuple.type = converted %.loc30_46.1, %.loc30_46.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %a.var, %.loc30_3
// CHECK:STDOUT:   br !.loc30_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc30_13:
// CHECK:STDOUT:   %false.loc30_13: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %false.loc30_19: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc30_13 br !and.rhs else br !and.result(%false.loc30_19)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %x.ref: bool = name_ref x, %x
// CHECK:STDOUT:   br !and.result(%x.ref)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc30_19: bool = block_arg !and.result [concrete = constants.%false]
// CHECK:STDOUT:   if %.loc30_19 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %Bool.call.loc30: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc30_30.1: type = value_of_initializer %Bool.call.loc30 [concrete = bool]
// CHECK:STDOUT:   %.loc30_30.2: type = converted %Bool.call.loc30, %.loc30_30.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result(%.loc30_30.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %.loc30_41: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc30_35: type = converted %.loc30_41, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result(%.loc30_35)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc30_10: type = block_arg !if.expr.result [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !.loc30_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc30_7:
// CHECK:STDOUT:   %a: ref %empty_tuple.type = ref_binding a, %a.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %a.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72f
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72f, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.ff9) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.b2f]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.843 = addr_of %a.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
