// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/unary_op.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/unary_op.carbon

fn Not(b: bool) -> bool {
  return not b;
}

let not_true: bool = not true;
let not_false: bool = not false;

fn Constant() {
  var a: if not true then bool else () = ();
  var b: if not false then bool else () = true;
}

// CHECK:STDOUT: --- unary_op.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %Not.type: type = fn_type @Not [concrete]
// CHECK:STDOUT:   %Not: %Not.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %Constant.type: type = fn_type @Constant [concrete]
// CHECK:STDOUT:   %Constant: %Constant.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.862: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl(bool) [concrete]
// CHECK:STDOUT:   %Op.type.655: type = fn_type @Op.2, @impl(bool) [concrete]
// CHECK:STDOUT:   %Op.8b7: %Op.type.655 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.2aa: %Destroy.type = facet_value bool, (%Destroy.impl_witness.862) [concrete]
// CHECK:STDOUT:   %.0d4: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.2aa [concrete]
// CHECK:STDOUT:   %Op.specific_fn.c3f: <specific function> = specific_function %Op.8b7, @Op.2(bool) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.1dc: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Op.type.a63: type = fn_type @Op.2, @impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Op.ea3: %Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.926: %Destroy.type = facet_value %empty_tuple.type, (%Destroy.impl_witness.1dc) [concrete]
// CHECK:STDOUT:   %.346: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.926 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.393: <specific function> = specific_function %Op.ea3, @Op.2(%empty_tuple.type) [concrete]
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
// CHECK:STDOUT:     .Not = %Not.decl
// CHECK:STDOUT:     .not_true = %not_true
// CHECK:STDOUT:     .not_false = %not_false
// CHECK:STDOUT:     .Constant = %Constant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Not.decl: %Not.type = fn_decl @Not [concrete = constants.%Not] {
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc14_20: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc14_20.1: type = value_of_initializer %bool.make_type.loc14_20 [concrete = bool]
// CHECK:STDOUT:     %.loc14_20.2: type = converted %bool.make_type.loc14_20, %.loc14_20.1 [concrete = bool]
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc14_11.1: type = splice_block %.loc14_11.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc14_11: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc14_11.2: type = value_of_initializer %bool.make_type.loc14_11 [concrete = bool]
// CHECK:STDOUT:       %.loc14_11.3: type = converted %bool.make_type.loc14_11, %.loc14_11.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_true.patt: %pattern_type.831 = binding_pattern not_true [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18_15.1: type = splice_block %.loc18_15.3 [concrete = bool] {
// CHECK:STDOUT:     %bool.make_type.loc18: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc18_15.2: type = value_of_initializer %bool.make_type.loc18 [concrete = bool]
// CHECK:STDOUT:     %.loc18_15.3: type = converted %bool.make_type.loc18, %.loc18_15.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %not_true: bool = bind_name not_true, @__global_init.%.loc18
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_false.patt: %pattern_type.831 = binding_pattern not_false [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_16.1: type = splice_block %.loc19_16.3 [concrete = bool] {
// CHECK:STDOUT:     %bool.make_type.loc19: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc19_16.2: type = value_of_initializer %bool.make_type.loc19 [concrete = bool]
// CHECK:STDOUT:     %.loc19_16.3: type = converted %bool.make_type.loc19, %.loc19_16.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %not_false: bool = bind_name not_false, @__global_init.%.loc19
// CHECK:STDOUT:   %Constant.decl: %Constant.type = fn_decl @Constant [concrete = constants.%Constant] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Not(%b.param: bool) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   %.loc15: bool = not %b.ref
// CHECK:STDOUT:   return %.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Constant() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.cb1 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var %a.var_patt
// CHECK:STDOUT:   %.loc22_43.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_43.2: init %empty_tuple.type = tuple_init () to %a.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc22_3.1: init %empty_tuple.type = converted %.loc22_43.1, %.loc22_43.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %a.var, %.loc22_3.1
// CHECK:STDOUT:   br !.loc22_17
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc22_17:
// CHECK:STDOUT:   %true.loc22: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc22_13: bool = not %true.loc22 [concrete = constants.%false]
// CHECK:STDOUT:   if %.loc22_13 br !if.expr.then.loc22 else br !if.expr.else.loc22
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc22:
// CHECK:STDOUT:   %bool.make_type.loc22: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc22_27.1: type = value_of_initializer %bool.make_type.loc22 [concrete = bool]
// CHECK:STDOUT:   %.loc22_27.2: type = converted %bool.make_type.loc22, %.loc22_27.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc22(%.loc22_27.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc22:
// CHECK:STDOUT:   %.loc22_38: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_32: type = converted %.loc22_38, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc22(%.loc22_32)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc22:
// CHECK:STDOUT:   %.loc22_10: type = block_arg !if.expr.result.loc22 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !.loc22_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc22_7:
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.831 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref bool = var %b.var_patt
// CHECK:STDOUT:   %true.loc23: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   assign %b.var, %true.loc23
// CHECK:STDOUT:   br !.loc23_17
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc23_17:
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc23_13: bool = not %false [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc23_13 br !if.expr.then.loc23 else br !if.expr.else.loc23
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc23:
// CHECK:STDOUT:   %bool.make_type.loc23: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc23_28.1: type = value_of_initializer %bool.make_type.loc23 [concrete = bool]
// CHECK:STDOUT:   %.loc23_28.2: type = converted %bool.make_type.loc23, %.loc23_28.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc23(%.loc23_28.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc23:
// CHECK:STDOUT:   %.loc23_39: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc23_33: type = converted %.loc23_39, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc23(%.loc23_33)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc23:
// CHECK:STDOUT:   %.loc23_10: type = block_arg !if.expr.result.loc23 [concrete = bool]
// CHECK:STDOUT:   br !.loc23_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc23_7:
// CHECK:STDOUT:   %b: ref bool = bind_name b, %b.var
// CHECK:STDOUT:   %impl.elem0.loc23: %.0d4 = impl_witness_access constants.%Destroy.impl_witness.862, element0 [concrete = constants.%Op.8b7]
// CHECK:STDOUT:   %bound_method.loc23_3.1: <bound method> = bound_method %b.var, %impl.elem0.loc23
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Op.2(bool) [concrete = constants.%Op.specific_fn.c3f]
// CHECK:STDOUT:   %bound_method.loc23_3.2: <bound method> = bound_method %b.var, %specific_fn.loc23
// CHECK:STDOUT:   %.loc23_3: bool = bind_value %b.var
// CHECK:STDOUT:   %no_op.loc23: init %empty_tuple.type = call %bound_method.loc23_3.2(%.loc23_3)
// CHECK:STDOUT:   %impl.elem0.loc22: %.346 = impl_witness_access constants.%Destroy.impl_witness.1dc, element0 [concrete = constants.%Op.ea3]
// CHECK:STDOUT:   %bound_method.loc22_3.1: <bound method> = bound_method %a.var, %impl.elem0.loc22
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem0.loc22, @Op.2(constants.%empty_tuple.type) [concrete = constants.%Op.specific_fn.393]
// CHECK:STDOUT:   %bound_method.loc22_3.2: <bound method> = bound_method %a.var, %specific_fn.loc22
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc22_3.2: %empty_tuple.type = converted %a.var, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %no_op.loc22: init %empty_tuple.type = call %bound_method.loc22_3.2(%.loc22_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc18: bool = not %true [concrete = constants.%false]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc19: bool = not %false [concrete = constants.%true]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
