// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/or.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/or.carbon

fn F() -> bool { return true; }
fn G() -> bool { return true; }

fn Or() -> bool {
  return F() or G();
}

fn Constant() {
  var a: if true or true then bool else () = true;
  var b: if true or false then bool else () = true;
  var c: if false or true then bool else () = true;
  var d: if false or false then bool else () = ();
}

fn PartialConstant(x: bool) {
  var a: if true or x then bool else () = true;
}

// CHECK:STDOUT: --- or.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Or.type: type = fn_type @Or [concrete]
// CHECK:STDOUT:   %Or: %Or.type = struct_value () [concrete]
// CHECK:STDOUT:   %Constant.type: type = fn_type @Constant [concrete]
// CHECK:STDOUT:   %Constant: %Constant.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.1dc: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Op.type.a63: type = fn_type @Op.2, @impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Op.ea3: %Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.926: %Destroy.type = facet_value %empty_tuple.type, (%Destroy.impl_witness.1dc) [concrete]
// CHECK:STDOUT:   %.346: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.926 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.393: <specific function> = specific_function %Op.ea3, @Op.2(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.862: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl(bool) [concrete]
// CHECK:STDOUT:   %Op.type.655: type = fn_type @Op.2, @impl(bool) [concrete]
// CHECK:STDOUT:   %Op.8b7: %Op.type.655 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.2aa: %Destroy.type = facet_value bool, (%Destroy.impl_witness.862) [concrete]
// CHECK:STDOUT:   %.0d4: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.2aa [concrete]
// CHECK:STDOUT:   %Op.specific_fn.c3f: <specific function> = specific_function %Op.8b7, @Op.2(bool) [concrete]
// CHECK:STDOUT:   %PartialConstant.type: type = fn_type @PartialConstant [concrete]
// CHECK:STDOUT:   %PartialConstant: %PartialConstant.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/types/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.%Op.type (%Op.type.bc9) = import_ref Core//prelude/destroy, loc15_23, loaded [symbolic = @impl.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .Or = %Or.decl
// CHECK:STDOUT:     .Constant = %Constant.decl
// CHECK:STDOUT:     .PartialConstant = %PartialConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc14_11.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc14_11.2: type = converted %bool.make_type, %.loc14_11.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc15_11.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc15_11.2: type = converted %bool.make_type, %.loc15_11.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Or.decl: %Or.type = fn_decl @Or [concrete = constants.%Or] {
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc17_12.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc17_12.2: type = converted %bool.make_type, %.loc17_12.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Constant.decl: %Constant.type = fn_decl @Constant [concrete = constants.%Constant] {} {}
// CHECK:STDOUT:   %PartialConstant.decl: %PartialConstant.type = fn_decl @PartialConstant [concrete = constants.%PartialConstant] {
// CHECK:STDOUT:     %x.patt: %pattern_type.831 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.831 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc28_23.1: type = splice_block %.loc28_23.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc28: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc28_23.2: type = value_of_initializer %bool.make_type.loc28 [concrete = bool]
// CHECK:STDOUT:       %.loc28_23.3: type = converted %bool.make_type.loc28, %.loc28_23.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: bool = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   return %true
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   return %true
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Or() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init bool = call %F.ref()
// CHECK:STDOUT:   %.loc18_14.1: bool = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc18_14.2: bool = converted %F.call, %.loc18_14.1
// CHECK:STDOUT:   %.loc18_14.3: bool = not %.loc18_14.2
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc18_14.3 br !or.rhs else br !or.result(%true)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init bool = call %G.ref()
// CHECK:STDOUT:   %.loc18_14.4: bool = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc18_14.5: bool = converted %G.call, %.loc18_14.4
// CHECK:STDOUT:   br !or.result(%.loc18_14.5)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc18_14.6: bool = block_arg !or.result
// CHECK:STDOUT:   return %.loc18_14.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Constant() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.831 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.831 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref bool = var %a.var_patt
// CHECK:STDOUT:   %true.loc22_46: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   assign %a.var, %true.loc22_46
// CHECK:STDOUT:   br !.loc22_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc22_13:
// CHECK:STDOUT:   %true.loc22_13: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc22_18.1: bool = not %true.loc22_13 [concrete = constants.%false]
// CHECK:STDOUT:   %true.loc22_18: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc22_18.1 br !or.rhs.loc22 else br !or.result.loc22(%true.loc22_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs.loc22:
// CHECK:STDOUT:   %true.loc22_21: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   br !or.result.loc22(%true.loc22_21)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result.loc22:
// CHECK:STDOUT:   %.loc22_18.2: bool = block_arg !or.result.loc22 [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc22_18.2 br !if.expr.then.loc22 else br !if.expr.else.loc22
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc22:
// CHECK:STDOUT:   %bool.make_type.loc22: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc22_31.1: type = value_of_initializer %bool.make_type.loc22 [concrete = bool]
// CHECK:STDOUT:   %.loc22_31.2: type = converted %bool.make_type.loc22, %.loc22_31.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc22(%.loc22_31.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc22:
// CHECK:STDOUT:   %.loc22_42: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_36: type = converted %.loc22_42, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc22(%.loc22_36)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc22:
// CHECK:STDOUT:   %.loc22_10: type = block_arg !if.expr.result.loc22 [concrete = bool]
// CHECK:STDOUT:   br !.loc22_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc22_7:
// CHECK:STDOUT:   %a: ref bool = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.831 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref bool = var %b.var_patt
// CHECK:STDOUT:   %true.loc23_47: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   assign %b.var, %true.loc23_47
// CHECK:STDOUT:   br !.loc23_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc23_13:
// CHECK:STDOUT:   %true.loc23_13: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc23_18.1: bool = not %true.loc23_13 [concrete = constants.%false]
// CHECK:STDOUT:   %true.loc23_18: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc23_18.1 br !or.rhs.loc23 else br !or.result.loc23(%true.loc23_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs.loc23:
// CHECK:STDOUT:   %false.loc23: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   br !or.result.loc23(%false.loc23)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result.loc23:
// CHECK:STDOUT:   %.loc23_18.2: bool = block_arg !or.result.loc23 [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc23_18.2 br !if.expr.then.loc23 else br !if.expr.else.loc23
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc23:
// CHECK:STDOUT:   %bool.make_type.loc23: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc23_32.1: type = value_of_initializer %bool.make_type.loc23 [concrete = bool]
// CHECK:STDOUT:   %.loc23_32.2: type = converted %bool.make_type.loc23, %.loc23_32.1 [concrete = bool]
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
// CHECK:STDOUT:   %b: ref bool = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.831 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.831 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref bool = var %c.var_patt
// CHECK:STDOUT:   %true.loc24_47: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   assign %c.var, %true.loc24_47
// CHECK:STDOUT:   br !.loc24_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc24_13:
// CHECK:STDOUT:   %false.loc24: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc24_19.1: bool = not %false.loc24 [concrete = constants.%true]
// CHECK:STDOUT:   %true.loc24_19: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc24_19.1 br !or.rhs.loc24 else br !or.result.loc24(%true.loc24_19)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs.loc24:
// CHECK:STDOUT:   %true.loc24_22: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   br !or.result.loc24(%true.loc24_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result.loc24:
// CHECK:STDOUT:   %.loc24_19.2: bool = block_arg !or.result.loc24 [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc24_19.2 br !if.expr.then.loc24 else br !if.expr.else.loc24
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc24:
// CHECK:STDOUT:   %bool.make_type.loc24: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc24_32.1: type = value_of_initializer %bool.make_type.loc24 [concrete = bool]
// CHECK:STDOUT:   %.loc24_32.2: type = converted %bool.make_type.loc24, %.loc24_32.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc24(%.loc24_32.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc24:
// CHECK:STDOUT:   %.loc24_43: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc24_37: type = converted %.loc24_43, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc24(%.loc24_37)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc24:
// CHECK:STDOUT:   %.loc24_10: type = block_arg !if.expr.result.loc24 [concrete = bool]
// CHECK:STDOUT:   br !.loc24_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc24_7:
// CHECK:STDOUT:   %c: ref bool = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.cb1 = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.cb1 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var %d.var_patt
// CHECK:STDOUT:   %.loc25_49.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc25_49.2: init %empty_tuple.type = tuple_init () to %d.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc25_3.1: init %empty_tuple.type = converted %.loc25_49.1, %.loc25_49.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %d.var, %.loc25_3.1
// CHECK:STDOUT:   br !.loc25_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc25_13:
// CHECK:STDOUT:   %false.loc25_13: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc25_19.1: bool = not %false.loc25_13 [concrete = constants.%true]
// CHECK:STDOUT:   %true.loc25: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc25_19.1 br !or.rhs.loc25 else br !or.result.loc25(%true.loc25)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs.loc25:
// CHECK:STDOUT:   %false.loc25_22: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   br !or.result.loc25(%false.loc25_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result.loc25:
// CHECK:STDOUT:   %.loc25_19.2: bool = block_arg !or.result.loc25 [concrete = constants.%false]
// CHECK:STDOUT:   if %.loc25_19.2 br !if.expr.then.loc25 else br !if.expr.else.loc25
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc25:
// CHECK:STDOUT:   %bool.make_type.loc25: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc25_33.1: type = value_of_initializer %bool.make_type.loc25 [concrete = bool]
// CHECK:STDOUT:   %.loc25_33.2: type = converted %bool.make_type.loc25, %.loc25_33.1 [concrete = bool]
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
// CHECK:STDOUT:   %d: ref %empty_tuple.type = bind_name d, %d.var
// CHECK:STDOUT:   %impl.elem0.loc25: %.346 = impl_witness_access constants.%Destroy.impl_witness.1dc, element0 [concrete = constants.%Op.ea3]
// CHECK:STDOUT:   %bound_method.loc25_3.1: <bound method> = bound_method %d.var, %impl.elem0.loc25
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Op.2(constants.%empty_tuple.type) [concrete = constants.%Op.specific_fn.393]
// CHECK:STDOUT:   %bound_method.loc25_3.2: <bound method> = bound_method %d.var, %specific_fn.loc25
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc25_3.2: %empty_tuple.type = converted %d.var, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %no_op.loc25: init %empty_tuple.type = call %bound_method.loc25_3.2(%.loc25_3.2)
// CHECK:STDOUT:   %impl.elem0.loc24: %.0d4 = impl_witness_access constants.%Destroy.impl_witness.862, element0 [concrete = constants.%Op.8b7]
// CHECK:STDOUT:   %bound_method.loc24_3.1: <bound method> = bound_method %c.var, %impl.elem0.loc24
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24, @Op.2(bool) [concrete = constants.%Op.specific_fn.c3f]
// CHECK:STDOUT:   %bound_method.loc24_3.2: <bound method> = bound_method %c.var, %specific_fn.loc24
// CHECK:STDOUT:   %.loc24_3: bool = bind_value %c.var
// CHECK:STDOUT:   %no_op.loc24: init %empty_tuple.type = call %bound_method.loc24_3.2(%.loc24_3)
// CHECK:STDOUT:   %impl.elem0.loc23: %.0d4 = impl_witness_access constants.%Destroy.impl_witness.862, element0 [concrete = constants.%Op.8b7]
// CHECK:STDOUT:   %bound_method.loc23_3.1: <bound method> = bound_method %b.var, %impl.elem0.loc23
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Op.2(bool) [concrete = constants.%Op.specific_fn.c3f]
// CHECK:STDOUT:   %bound_method.loc23_3.2: <bound method> = bound_method %b.var, %specific_fn.loc23
// CHECK:STDOUT:   %.loc23_3: bool = bind_value %b.var
// CHECK:STDOUT:   %no_op.loc23: init %empty_tuple.type = call %bound_method.loc23_3.2(%.loc23_3)
// CHECK:STDOUT:   %impl.elem0.loc22: %.0d4 = impl_witness_access constants.%Destroy.impl_witness.862, element0 [concrete = constants.%Op.8b7]
// CHECK:STDOUT:   %bound_method.loc22_3.1: <bound method> = bound_method %a.var, %impl.elem0.loc22
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem0.loc22, @Op.2(bool) [concrete = constants.%Op.specific_fn.c3f]
// CHECK:STDOUT:   %bound_method.loc22_3.2: <bound method> = bound_method %a.var, %specific_fn.loc22
// CHECK:STDOUT:   %.loc22_3: bool = bind_value %a.var
// CHECK:STDOUT:   %no_op.loc22: init %empty_tuple.type = call %bound_method.loc22_3.2(%.loc22_3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartialConstant(%x.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.831 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.831 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref bool = var %a.var_patt
// CHECK:STDOUT:   %true.loc29_43: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   assign %a.var, %true.loc29_43
// CHECK:STDOUT:   br !.loc29_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc29_13:
// CHECK:STDOUT:   %true.loc29_13: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc29_18.1: bool = not %true.loc29_13 [concrete = constants.%false]
// CHECK:STDOUT:   %true.loc29_18: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc29_18.1 br !or.rhs else br !or.result(%true.loc29_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %x.ref: bool = name_ref x, %x
// CHECK:STDOUT:   br !or.result(%x.ref)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc29_18.2: bool = block_arg !or.result [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc29_18.2 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %bool.make_type.loc29: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc29_28.1: type = value_of_initializer %bool.make_type.loc29 [concrete = bool]
// CHECK:STDOUT:   %.loc29_28.2: type = converted %bool.make_type.loc29, %.loc29_28.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result(%.loc29_28.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %.loc29_39: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc29_33: type = converted %.loc29_39, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result(%.loc29_33)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc29_10: type = block_arg !if.expr.result [concrete = bool]
// CHECK:STDOUT:   br !.loc29_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc29_7:
// CHECK:STDOUT:   %a: ref bool = bind_name a, %a.var
// CHECK:STDOUT:   %impl.elem0: %.0d4 = impl_witness_access constants.%Destroy.impl_witness.862, element0 [concrete = constants.%Op.8b7]
// CHECK:STDOUT:   %bound_method.loc29_3.1: <bound method> = bound_method %a.var, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Op.2(bool) [concrete = constants.%Op.specific_fn.c3f]
// CHECK:STDOUT:   %bound_method.loc29_3.2: <bound method> = bound_method %a.var, %specific_fn
// CHECK:STDOUT:   %.loc29_3: bool = bind_value %a.var
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method.loc29_3.2(%.loc29_3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
